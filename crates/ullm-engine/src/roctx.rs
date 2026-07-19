// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal, opt-in ROCTx range support for local GPU diagnostics.
//!
//! The SDK ROCTx library is loaded only when a diagnostic calls [`enable`].  Until then a range
//! is one atomic flag check, so ordinary serving does not load a profiler library or emit marker
//! events.  We intentionally use the ROCprofiler SDK implementation rather than the legacy
//! `libroctx64` compatibility library: rocprofv3 observes the former's range events.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

const RTLD_NOW: c_int = 2;
const ROCTX_SDK_LIBRARY: &[u8] = b"librocprofiler-sdk-roctx.so.1\0";
const ROCTX_RANGE_PUSH_A: &[u8] = b"roctxRangePushA\0";
const ROCTX_RANGE_POP: &[u8] = b"roctxRangePop\0";

type RangePush = unsafe extern "C" fn(*const c_char) -> c_int;
type RangePop = unsafe extern "C" fn() -> c_int;

struct RoctxApi {
    // The DSO is deliberately retained for the rest of the process: these function pointers
    // must remain valid while a profile range can be open.
    _handle: *mut c_void,
    push: RangePush,
    pop: RangePop,
}

// A dynamically loaded DSO handle and its immutable function pointers are safe to share.
unsafe impl Send for RoctxApi {}
unsafe impl Sync for RoctxApi {}

static ROCTX_API: OnceLock<Result<RoctxApi, String>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// Enables ROCTx emission for this process.
///
/// This is deliberately explicit: shared inference code remains inert unless a dedicated
/// diagnostic opts in.  The SDK library and both required range symbols must be available.
pub fn enable() -> Result<(), String> {
    match ROCTX_API.get_or_init(load_roctx_api) {
        Ok(_) => {
            ENABLED.store(true, Ordering::Release);
            Ok(())
        }
        Err(error) => Err(error.clone()),
    }
}

/// Opens a thread-local, balanced ROCTx range when [`enable`] has been called.
pub fn range(label: &str) -> RoctxRange {
    if !ENABLED.load(Ordering::Acquire) {
        return RoctxRange { active: false };
    }
    let Some(api) = ROCTX_API.get().and_then(|result| result.as_ref().ok()) else {
        return RoctxRange { active: false };
    };
    let Ok(label) = CString::new(label) else {
        return RoctxRange { active: false };
    };
    // SAFETY: `enable` resolved the SDK function pointer, and `label` remains alive throughout
    // this synchronous C call.
    unsafe { (api.push)(label.as_ptr()) };
    RoctxRange { active: true }
}

/// RAII guard that balances a successful [`range`] call, including on error returns.
pub struct RoctxRange {
    active: bool,
}

impl Drop for RoctxRange {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Some(Ok(api)) = ROCTX_API.get() {
            // SAFETY: `enable` resolved the SDK function pointer and intentionally retains its
            // DSO handle until process exit.
            unsafe { (api.pop)() };
        }
    }
}

fn load_roctx_api() -> Result<RoctxApi, String> {
    // SAFETY: the library name is a static NUL-terminated string and RTLD_NOW is a valid dlopen
    // flag.  The handle is intentionally never closed (see `RoctxApi`).
    let handle = unsafe { dlopen(ROCTX_SDK_LIBRARY.as_ptr().cast::<c_char>(), RTLD_NOW) };
    if handle.is_null() {
        return Err(format!(
            "failed to load ROCprofiler SDK ROCTx library: {}",
            dl_error_message()
        ));
    }
    let push = load_symbol::<RangePush>(handle, ROCTX_RANGE_PUSH_A)?;
    let pop = load_symbol::<RangePop>(handle, ROCTX_RANGE_POP)?;
    Ok(RoctxApi {
        _handle: handle,
        push,
        pop,
    })
}

fn load_symbol<T>(handle: *mut c_void, name: &[u8]) -> Result<T, String> {
    // SAFETY: `handle` was returned by dlopen, and `name` is a static NUL-terminated symbol.
    let symbol = unsafe { dlsym(handle, name.as_ptr().cast::<c_char>()) };
    if symbol.is_null() {
        return Err(format!(
            "ROCprofiler SDK ROCTx is missing {}: {}",
            String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]),
            dl_error_message()
        ));
    }
    // SAFETY: the caller supplies the exact C ABI signature for the named ROCTx symbol.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) })
}

fn dl_error_message() -> String {
    // SAFETY: dlerror returns either null or a NUL-terminated error string owned by the loader.
    let error = unsafe { dlerror() };
    if error.is_null() {
        "dynamic loader returned no detail".to_string()
    } else {
        // SAFETY: checked non-null above; POSIX guarantees the string is NUL-terminated.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}
