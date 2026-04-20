// shim.h — C-linkage surface over XRT's C++ API.
//
// We expose exactly what 1bit-xdna's Rust side needs: open a device
// by BDF index, load an xclbin, run a named kernel with one input + one
// output buffer. Every function returns a small int32 status (0 = OK, <0
// = error) to keep the FFI boundary simple — errors are logged on the C++
// side; callers can print them by enabling XRT_INI=xrt.log.
//
// Lifetime contract:
//   - `onebit_xrt_open` allocates an opaque device handle. Caller MUST
//     eventually free it with `onebit_xrt_close`.
//   - The handle is not thread-safe; external synchronization required if
//     shared between threads.

#ifndef HALO_XRT_SHIM_H
#define HALO_XRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes returned by every shim call.
#define HALO_XRT_OK             0
#define HALO_XRT_E_INVALID     -1
#define HALO_XRT_E_NOT_FOUND   -2
#define HALO_XRT_E_DEVICE      -3
#define HALO_XRT_E_KERNEL      -4
#define HALO_XRT_E_INTERNAL    -5

// Opaque device handle. Defined in shim.cpp.
typedef struct HaloXrtDevice HaloXrtDevice;

// Open the XRT device at BDF index `bdf_idx`. Returns NULL on error (bad
// index, no XDNA silicon, XRT runtime init failed).
HaloXrtDevice* onebit_xrt_open(uint32_t bdf_idx);

// Free a handle returned by onebit_xrt_open. No-op on NULL.
void onebit_xrt_close(HaloXrtDevice* dev);

// Load an xclbin from disk. The xclbin is registered with the device and
// stays resident until the device handle is closed. Returns:
//   HALO_XRT_OK         on success
//   HALO_XRT_E_INVALID  if dev is NULL or path is NULL
//   HALO_XRT_E_NOT_FOUND if the file doesn't exist or can't be read
//   HALO_XRT_E_DEVICE   if XRT rejected the xclbin (wrong target, etc.)
int32_t onebit_xrt_load_xclbin(HaloXrtDevice* dev, const char* path);

// Run kernel `name` on this device with a single input buffer + single
// output buffer. The shim allocates XRT BOs, memcpy's the input in, runs
// the kernel to completion, memcpy's the output back, and frees the BOs.
//
// This is the "stub dispatch" surface — when a real xclbin lands the shim
// can grow a richer multi-buffer API, but one-in/one-out is enough to wire
// up 1bit-router's NPU fallback path today.
//
// Returns HALO_XRT_OK on success, HALO_XRT_E_KERNEL if the kernel name
// isn't exported by the loaded xclbin, HALO_XRT_E_INTERNAL on any other
// XRT failure during dispatch.
int32_t onebit_xrt_run_kernel(
    HaloXrtDevice* dev,
    const char*    name,
    const uint8_t* in,
    size_t         in_len,
    uint8_t*       out,
    size_t         out_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HALO_XRT_SHIM_H
