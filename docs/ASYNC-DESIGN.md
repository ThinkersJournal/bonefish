# Non-blocking operations and `Event` (R3b): design note

Status: **design, no code**. Bonefish lane, 2026-10-30. Decisions already taken by the PM (2026-10-30): scoped API (A) built on an internal unsafe layer (C); queues in-order by default with explicit dependency lists still exposed; shim ops return a `sycl::event*` and take a dependency array. This note says how those fit together and what is still unproven.

Checked against the oneAPI 2026.1 headers on the laptop: `sycl::property::queue::in_order` exists (`properties/queue_properties.def`), as do `ext::oneapi::property::queue::discard_events` and `queue::is_in_order()`; the oneMKL BLAS entry points return a `sycl::event` and take `const std::vector<sycl::event>&` dependencies (`blas/usm_decls.hpp`). Behaviour that has not been run is marked **UNVERIFIED**.

## 1. The problem

Every shim call today ends in `.wait()`, so a safe call returns with the device finished and nothing is ever in flight. A non-blocking call returns while the device still reads and writes the operand buffers. Rust must stop the host from dropping, reading or mutating those buffers until the work is done.

"Wait in `Event::drop`" is not enough: `mem::forget(event)` ends the borrow with the work still running. **Soundness must not depend on a handle's `Drop`.** Hence the scope: the thing that waits is the scope, which cannot be forgotten from outside because the caller's closure runs inside it.

## 2. API sketch (Rust)

```rust
queue.scope(|s| -> Result<()> {
    let a = s.read(&a_buf);        // Read<'scope, T>   : holds &'scope UsmBuffer
    let b = s.read(&b_buf);
    let c = s.write(&mut c_buf);   // Write<'scope, T>  : holds &'scope mut UsmBuffer
    let e1 = s.gemm(Transpose::No, Transpose::No, m, n, k, 1.0, &a, lda, &b, ldb, 0.0, &c, ldc)?;
    let e2 = s.gemm_after(&[&e1], /* same arguments, next product into c */ ..)?;
    e2.wait()?;                    // optional: wait for one event early
    Ok(())
})?;                               // scope end: queue wait, then the first asynchronous error
```

- `Queue::scope<'env, F, R>(&self, f: F) -> Result<R> where F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> Result<R>`, the same shape as `std::thread::scope`, so `'scope` is invariant and nothing borrowed can escape.
- `Scope::read` / `write` **bind** a buffer once into a handle. Operations take handles, not buffers. This is deliberate: if each operation borrowed its buffer for the whole scope, a second operation chaining on the first one's output could never name that buffer again (the borrow checker would see two overlapping `&mut`). Binding once and passing the handle by `&` lets many operations reference one buffer.
- `Event<'scope>` wraps a `sycl::event*`; it cannot outlive the scope; dropping it only frees the handle (a leak costs memory, never safety).
- The blocking functions stay (`blas::gemm`, `Dft::forward_*`), implemented as "submit, wait".

**Implemented (0.0.12), differences from the sketch below:** operations take any `Operand` (a `Read` or a `Write`) as input and a `&Write` as output, so a result can feed the next operation; an output that is also an input of the same call is refused at run time (`InvalidArgument`), since the types alone cannot say two handles name one buffer; `gemm_after(&[&event], ...)` is the explicit-dependency form and `gemm` is `gemm_after(&[], ...)`; the scope's wait result is combined as "the closure's error, else the first asynchronous error, else the value".

## 3. Why this is sound, and what it depends on

1. **Host cannot touch a bound buffer.** `read` keeps `&'scope UsmBuffer` alive and `write` keeps `&'scope mut`; the host's `as_slice`/`to_vec`/`copy_from_slice` need `&`/`&mut` on the same buffer and are rejected by the borrow checker until the scope ends. Pinned by `compile_fail,E0502` / `E0505` doctests (read or drop while bound).
2. **Scope waits on every exit path.** A guard's `Drop` calls `queue.wait()`. It runs on normal return, on `?`, and on panic unwinding. Tested with `catch_unwind` (a kernel in flight, closure panics, the buffer is intact afterwards).
3. **`mem::forget` cannot skip the wait**: there is nothing outside the closure to forget. Forgetting a `Read`/`Write` handle inside the closure only leaks the *borrow bookkeeping*, which the scope's end still closes.
4. **Aliasing between in-flight operations.** Two operations naming the same `Write` handle are only race-free if the device never runs them concurrently. **That is why queues are created in-order** (`sycl::property::queue::in_order`): device work on one queue is serialised, so chained operations on a bound buffer cannot overlap. In-order is therefore a *soundness* requirement of this API, not only a convenience. Cross-queue ordering is the job of `gemm_after(&[&e])`.
5. **Queue outlives the work.** Handles hold the scope, which holds `&Queue`; `Queue` is dropped only after the scope returned.

What this does **not** give: concurrent operations on one queue (in-order forbids it by design), or sharing a `Scope` between threads (`Scope` is `!Sync`; one scope per thread, matching the `Queue` thread-safety finding in `ROADMAP.md` R3b).

## 4. Shim ABI changes (`bonefish-sys`)

All new functions are additive; the blocking ones keep working.

```c
// Opaque event handle: a heap-allocated sycl::event. Null on failure (see bf_last_error).
void*   bf_gemm_s_async(void* q, ..same args as bf_sgemm.., void* const* deps, int64_t n_deps);  // returns event*
void*   bf_gemm_d_async(...);
void*   bf_dft_compute_async(void* plan, int32_t dir, void* in, void* out, void* const* deps, int64_t n_deps);
int32_t bf_event_wait(void* event);     // blocking; surfaces an asynchronous error as a status
void    bf_event_free(void* event);
int32_t bf_queue_wait(void* q);         // wait_and_throw semantics, see section 5
void*   bf_queue_new_in_order(int32_t kind);   // or an `in_order` flag on bf_queue_new
```

`oneapi::mkl::blas::column_major::gemm` already takes `const std::vector<sycl::event>& dependencies` and returns the event, so the shim only stops calling `.wait()` and builds the vector from `deps`.

## 5. Asynchronous errors

A SYCL queue constructed without an `async_handler` reports *asynchronous* exceptions (raised after submission) through the implementation's default behaviour, which can be to terminate the process. **UNVERIFIED on 2026.1**: I have not run an asynchronously failing kernel. Plan: install an `async_handler` on every queue the shim creates that stores the exceptions in a per-queue list; `bf_queue_wait` / `bf_event_wait` call `wait_and_throw`/drain that list and return the usual status codes (1 SYCL, 2 oneMKL). The first implementation row is a test that **forces** an asynchronous failure and shows the status arrives instead of an abort; if it cannot be forced on the CPU device, that is reported as unverified rather than assumed.

**Measured since (0.0.11, shim row; Windows, oneAPI 2026.1, CPU/OpenCL; GPU UNVERIFIED):** the worry was real. With no `async_handler` on the queue, an asynchronous failure (a host task that throws) made SYCL's default handler print the exception and the process exited (code 127). With the handler installed, the failure comes back from `bf_event_wait` and from `bf_queue_wait` as status 1 with its message, and is not reported a second time. The handler runs on the thread that calls `wait_and_throw`, so the shim collects into a per-thread list and drains it right after the wait; a queue destroyed with uncollected errors drops them rather than blaming the next queue waited on from that thread. Not measured: a *device-side* kernel failure (the test forces the error with a host task, which is the portable way to raise an asynchronous error).

Also measured: with the `in_order` property removed, an out-of-order queue still passed the dependent-chain test 15 of 15 on this device. So section 3.4's "in-order is a soundness requirement" is a requirement of the *specification*, which the property test (`is_in_order()`) pins; the chain test cannot show it on the CPU device.

## 6. Other risks and non-goals

- **`discard_events`**: Intel's oneMKL known-limitations page says queues created with the `discard_events` property are unsupported and "may lead to unexpected exceptions". I saw that through a search summary, not the page itself, so treat the wording as secondhand. The property exists in the headers (above). The shim never sets it, and a test asserts `is_in_order()` and, by construction, that only `in_order` was requested.
- **Out-of-order queues** are not offered in v1 (see 3.4). A later row could add them with a stricter, per-buffer-token API.
- **DFT plans** are class-based oneMKL objects, not shareable between threads (R3b); an async `compute` borrows the plan `&mut` for the scope.
- **GPU**: every statement here was reasoned from the headers and, once implemented, is tested on the CPU/OpenCL device only. **GPU behaviour is UNVERIFIED**; an in-order queue's ordering guarantee on a GPU is the SYCL specification's, not something measured here.

## 7. Rows, in order

1. Shim: in-order queue, async `gemm` (s, d), `bf_event_wait/free`, `bf_queue_wait`, the `async_handler`; tests in `bonefish-sys` including the forced asynchronous failure (or its honest absence).
2. `bonefish`: `Scope`, `Read`/`Write`, `Event<'scope>`, async `gemm`; tests: result correctness after a chain, `compile_fail,E0502/E0505` hazards, `catch_unwind` wait-on-panic.
3. Async `Dft::compute`, async level-1/`gemv`.
