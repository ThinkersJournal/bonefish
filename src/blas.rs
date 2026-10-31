//! BLAS through oneMKL. Every function bounds-checks its buffers in Rust before calling the
//! device, so a safe call cannot read or write outside a [`UsmBuffer`].

use crate::error::{Error, Result};
use crate::queue::Queue;
use crate::scalar::Scalar;
use crate::usm::UsmBuffer;

/// How a matrix operand is used. For real types `ConjTrans` is the same as `Trans`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transpose {
    /// `op(A) = A`.
    No,
    /// `op(A) = A^T`.
    Trans,
    /// `op(A) = A^H`.
    ConjTrans,
}

impl Transpose {
    pub(crate) fn code(self) -> i32 {
        match self {
            Transpose::No => 0,
            Transpose::Trans => 1,
            Transpose::ConjTrans => 2,
        }
    }
}

/// Check that a column-major `rows x cols` matrix with leading dimension `ld` fits in `len`
/// elements. An empty matrix touches nothing, but `ld` must still be at least `max(1, rows)`.
pub(crate) fn check_matrix(
    name: &str,
    len: usize,
    ld: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if ld < rows.max(1) {
        return Err(Error::InvalidArgument(format!(
            "{name}: leading dimension {ld} is below the row count {rows}"
        )));
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    // Last element touched is at (rows - 1) + (cols - 1) * ld.
    let needed = (cols - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(rows))
        .ok_or_else(|| Error::InvalidArgument(format!("{name}: extent overflows")))?;
    if needed > len {
        return Err(Error::InvalidArgument(format!(
            "{name}: a {rows} x {cols} matrix with ld {ld} needs {needed} elements, buffer has {len}"
        )));
    }
    Ok(())
}

pub(crate) fn dim(name: &str, v: usize) -> Result<i64> {
    i64::try_from(v).map_err(|_| Error::InvalidArgument(format!("{name} = {v} exceeds i64")))
}

/// `C = alpha * op(A) * op(B) + beta * C`, column-major, blocking.
///
/// `op(A)` is `m x k`, `op(B)` is `k x n`, `C` is `m x n`. When `C` is the destination,
/// `beta = 0` ignores whatever (including NaN) it held.
///
/// # Errors
///
/// [`Error::QueueMismatch`] if a buffer was allocated from another queue;
/// [`Error::InvalidArgument`] if a leading dimension is below its row count or a buffer is too
/// short for the matrix it must hold; [`Error::Sycl`] / [`Error::Mkl`] from the device.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: Scalar>(
    queue: &Queue,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &UsmBuffer<T>,
    lda: usize,
    b: &UsmBuffer<T>,
    ldb: usize,
    beta: T,
    c: &mut UsmBuffer<T>,
    ldc: usize,
) -> Result<()> {
    if !(queue.same_as(a.queue()) && queue.same_as(b.queue()) && queue.same_as(c.queue())) {
        return Err(Error::QueueMismatch);
    }
    let (a_rows, a_cols) = if transa == Transpose::No {
        (m, k)
    } else {
        (k, m)
    };
    let (b_rows, b_cols) = if transb == Transpose::No {
        (k, n)
    } else {
        (n, k)
    };
    check_matrix("A", a.len(), lda, a_rows, a_cols)?;
    check_matrix("B", b.len(), ldb, b_rows, b_cols)?;
    check_matrix("C", c.len(), ldc, m, n)?;
    // SAFETY: every pointer is USM from `queue` (checked above) and the extents above fit inside
    // the buffers, so the device reads and writes stay in bounds. `c` is exclusively borrowed, so
    // it cannot alias `a` or `b`.
    let rc = unsafe {
        T::gemm(
            queue.raw(),
            transa.code(),
            transb.code(),
            dim("m", m)?,
            dim("n", n)?,
            dim("k", k)?,
            alpha,
            a.raw(),
            dim("lda", lda)?,
            b.raw(),
            dim("ldb", ldb)?,
            beta,
            c.raw(),
            dim("ldc", ldc)?,
        )
    };
    Queue::status(rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_that_fits_is_accepted() {
        // 2 x 3, ld 2: touches elements 0..6.
        assert!(check_matrix("A", 6, 2, 2, 3).is_ok());
        // Padded: ld 4 needs (3 - 1) * 4 + 2 = 10.
        assert!(check_matrix("A", 10, 4, 2, 3).is_ok());
    }

    #[test]
    fn a_short_buffer_is_rejected_with_the_needed_size() {
        // Positive control above: 6 fits. One fewer does not, and the message says what is needed.
        let e = check_matrix("A", 5, 2, 2, 3).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("needs 6")),
            "{e}"
        );
        assert!(check_matrix("A", 9, 4, 2, 3).is_err());
    }

    #[test]
    fn a_leading_dimension_below_the_rows_is_rejected() {
        let e = check_matrix("B", 100, 1, 2, 2).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidArgument(m) if m.contains("leading dimension")),
            "{e}"
        );
    }

    #[test]
    fn empty_matrices_touch_nothing_but_still_need_a_sane_ld() {
        assert!(check_matrix("A", 0, 1, 0, 5).is_ok());
        // 5 rows, 0 columns: no element is touched, and ld = 5 is valid for 5 rows.
        assert!(check_matrix("A", 0, 5, 5, 0).is_ok());
        // Positive/negative pair: the same shape with ld = 1 < 5 rows is rejected.
        assert!(check_matrix("A", 0, 1, 5, 0).is_err());
        assert!(
            check_matrix("A", 0, 0, 0, 5).is_err(),
            "ld 0 is below max(1, rows)"
        );
    }

    #[test]
    fn extent_overflow_is_an_error_not_a_wrap() {
        assert!(check_matrix("A", usize::MAX, usize::MAX, 2, usize::MAX).is_err());
    }
}
