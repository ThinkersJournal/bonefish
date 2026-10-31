//! `Queue::scope`: non-blocking operations, events, and the guarantees the scope gives
//! (`docs/ASYNC-DESIGN.md`). Needs a device; without oneAPI the tests skip loudly unless
//! `BONEFISH_REQUIRE_SYCL=1`. The compile-time guarantees (`E0502`, `E0505`, no escape) are
//! doctests on `Queue::scope`. CPU/OpenCL device only; GPU UNVERIFIED.

use std::panic::{AssertUnwindSafe, catch_unwind};

use bonefish::{DeviceKind, Error, Queue, SYCL_AVAILABLE, Transpose, UsmBuffer, UsmKind};

fn queue(test: &str) -> Option<Queue> {
    if !SYCL_AVAILABLE {
        assert!(
            !std::env::var("BONEFISH_REQUIRE_SYCL").is_ok_and(|v| v == "1"),
            "{test}: BONEFISH_REQUIRE_SYCL=1 but bonefish was built without oneAPI"
        );
        eprintln!("SKIP {test}: built without oneAPI. Nothing ran on a device.");
        return None;
    }
    Some(Queue::new(DeviceKind::Cpu).expect("a CPU SYCL device"))
}

fn buf(q: &Queue, data: Vec<f32>) -> UsmBuffer<f32> {
    UsmBuffer::from_slice(q, UsmKind::Shared, &data).unwrap()
}

fn two_identity(n: usize) -> Vec<f32> {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 2.0;
    }
    v
}

/// Ten dependent products ping-ponging between two buffers, every operand a bound handle and the
/// intermediate results read back as inputs of the next operation. The answer is a power of two.
#[test]
fn a_chain_of_dependent_products_gives_the_right_answer() {
    let Some(q) = queue("chain") else { return };
    let n = 128;
    let a = buf(&q, two_identity(n));
    let mut x = buf(&q, vec![1.0; n * n]);
    let mut y = buf(&q, vec![0.0; n * n]);
    let steps = 10;
    q.scope(|s| {
        let a = s.read(&a)?;
        let x = s.write(&mut x)?;
        let y = s.write(&mut y)?;
        for step in 0..steps {
            let (src, dst) = if step % 2 == 0 { (&x, &y) } else { (&y, &x) };
            s.gemm(
                Transpose::No,
                Transpose::No,
                n,
                n,
                n,
                1.0,
                &a,
                n,
                src,
                n,
                0.0,
                dst,
                n,
            )?;
        }
        Ok(())
    })
    .unwrap();
    let want = (1u32 << steps) as f32;
    let result = if steps % 2 == 0 { &x } else { &y };
    let v = result.to_vec().unwrap();
    assert!(v.iter().all(|e| *e == want), "first element {}", v[0]);
}

/// The same chain, each step stating its dependency explicitly with an event.
#[test]
fn explicit_event_dependencies_are_accepted() {
    let Some(q) = queue("deps") else { return };
    let n = 64;
    let a = buf(&q, two_identity(n));
    let mut x = buf(&q, vec![1.0; n * n]);
    let mut y = buf(&q, vec![0.0; n * n]);
    let mut z = buf(&q, vec![0.0; n * n]);
    q.scope(|s| {
        let a = s.read(&a)?;
        let (x, y, z) = (s.write(&mut x)?, s.write(&mut y)?, s.write(&mut z)?);
        let e1 = s.gemm(
            Transpose::No,
            Transpose::No,
            n,
            n,
            n,
            1.0,
            &a,
            n,
            &x,
            n,
            0.0,
            &y,
            n,
        )?;
        let e2 = s.gemm_after(
            &[&e1],
            Transpose::No,
            Transpose::No,
            n,
            n,
            n,
            1.0,
            &a,
            n,
            &y,
            n,
            0.0,
            &z,
            n,
        )?;
        e2.wait()?; // waiting early is allowed
        Ok(())
    })
    .unwrap();
    assert!(z.to_vec().unwrap().iter().all(|e| *e == 4.0));
}

/// The closure panics with work in flight. The scope must still wait before the buffers' borrows
/// end, so the result is complete afterwards. The result is read through the host pointer
/// (`as_slice`, shared USM) rather than `to_vec`, because `to_vec` queues a copy behind the work
/// and would hide a missing wait.
#[test]
fn a_panic_inside_the_scope_still_waits_for_the_device() {
    let Some(q) = queue("panic") else { return };
    let n = 512;
    let a = buf(&q, vec![1.0; n * n]);
    let b = buf(&q, vec![1.0; n * n]);
    let mut c = buf(&q, vec![0.0; n * n]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = q.scope(|s| {
            let (a, b) = (s.read(&a)?, s.read(&b)?);
            let c = s.write(&mut c)?;
            s.gemm(
                Transpose::No,
                Transpose::No,
                n,
                n,
                n,
                1.0,
                &a,
                n,
                &b,
                n,
                0.0,
                &c,
                n,
            )?;
            panic!("closure panics with the product in flight");
            #[allow(unreachable_code)]
            Ok(())
        });
    }));
    assert!(r.is_err(), "the panic must propagate");
    let host = c.as_slice().expect("shared USM is host readable");
    assert!(
        host.iter().all(|e| *e == n as f32),
        "first element {}, last {}: the device was still running after the unwind",
        host[0],
        host[host.len() - 1]
    );
}

#[test]
fn a_buffer_from_another_queue_is_rejected_at_binding() {
    let Some(q) = queue("mismatch") else { return };
    let other = Queue::new(DeviceKind::Cpu).unwrap();
    let foreign = buf(&other, vec![0.0; 4]);
    let own = buf(&q, vec![0.0; 4]);
    q.scope(|s| {
        // Control: a buffer from this queue binds.
        assert!(s.read(&own).is_ok());
        assert!(matches!(s.read(&foreign), Err(Error::QueueMismatch)));
        Ok(())
    })
    .unwrap();
}

#[test]
fn an_output_that_is_also_an_input_is_rejected() {
    let Some(q) = queue("alias") else { return };
    let mut c = buf(&q, vec![1.0; 4]);
    let b = buf(&q, vec![1.0; 4]);
    q.scope(|s| {
        let b = s.read(&b)?;
        let c = s.write(&mut c)?;
        // c as its own input: refused.
        let r = s.gemm(
            Transpose::No,
            Transpose::No,
            2,
            2,
            2,
            1.0,
            &c,
            2,
            &b,
            2,
            0.0,
            &c,
            2,
        );
        assert!(matches!(&r, Err(Error::InvalidArgument(m)) if m.contains("also an input")));
        Ok(())
    })
    .unwrap();
}

#[test]
fn a_short_buffer_is_rejected_before_anything_is_submitted() {
    let Some(q) = queue("bounds") else { return };
    let a = buf(&q, vec![1.0; 4]);
    let small = buf(&q, vec![1.0; 3]); // a 2 x 2 operand needs 4
    let mut c = buf(&q, vec![0.0; 4]);
    q.scope(|s| {
        let (a, small) = (s.read(&a)?, s.read(&small)?);
        let c = s.write(&mut c)?;
        let bad = s.gemm(
            Transpose::No,
            Transpose::No,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &small,
            2,
            0.0,
            &c,
            2,
        );
        assert!(matches!(&bad, Err(Error::InvalidArgument(m)) if m.contains("needs 4")));
        // Control: the same call with a buffer of the right size is accepted.
        Ok(())
    })
    .unwrap();
    let a2 = buf(&q, vec![1.0; 4]);
    let b2 = buf(&q, vec![1.0; 4]);
    q.scope(|s| {
        let (a2, b2) = (s.read(&a2)?, s.read(&b2)?);
        let c = s.write(&mut c)?;
        s.gemm(
            Transpose::No,
            Transpose::No,
            2,
            2,
            2,
            1.0,
            &a2,
            2,
            &b2,
            2,
            0.0,
            &c,
            2,
        )?;
        Ok(())
    })
    .unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![2.0; 4]);
}

#[test]
fn without_sycl_scope_reports_unavailable_through_bindings() {
    if SYCL_AVAILABLE {
        return;
    }
    // No queue can exist, so there is nothing to scope; the type must still compile and the
    // constructor must fail cleanly (the compile check is this whole file building).
    assert!(matches!(
        Queue::new(DeviceKind::Cpu),
        Err(Error::Unavailable(_))
    ));
}
