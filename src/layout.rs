//! Shapes shared by the BLAS layers: how a matrix operand is used, and the extent checks every
//! safe call makes before it reaches the device. This sits below `blas` and `scope` so neither has
//! to import the other (ROADMAP R14, MOD-002).

use crate::error::{Error, Result};

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
