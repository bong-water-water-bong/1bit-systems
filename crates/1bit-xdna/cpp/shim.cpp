// shim.cpp — thin C++ → C bridge over XRT's C++ API.
//
// This file is compiled ONLY when the `real-xrt` feature is enabled.
// Rust's `build.rs` gates the cc::Build invocation on CARGO_FEATURE_REAL_XRT.
//
// The file is intentionally small. XRT's C++ API is exception-throwing;
// we catch everything at the extern-C boundary and return status codes so
// Rust never sees a C++ exception unwind through FFI (which would be UB).

#include "shim.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>

// Opaque handle layout. Keeping the xrt::device + currently-loaded uuid
// inside the struct means the Rust side doesn't need to care about XRT's
// move-only types.
struct HaloXrtDevice {
    xrt::device        device;
    xrt::uuid          xclbin_uuid;
    bool               xclbin_loaded;

    HaloXrtDevice()
        : device{}, xclbin_uuid{}, xclbin_loaded{false} {}
};

extern "C" HaloXrtDevice* onebit_xrt_open(uint32_t bdf_idx) {
    try {
        auto* d = new HaloXrtDevice();
        d->device = xrt::device(bdf_idx);
        return d;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "onebit_xrt_open(%u) failed: %s\n", bdf_idx, e.what());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "onebit_xrt_open(%u) failed: unknown exception\n", bdf_idx);
        return nullptr;
    }
}

extern "C" void onebit_xrt_close(HaloXrtDevice* dev) {
    if (dev == nullptr) return;
    try {
        delete dev;
    } catch (...) {
        // Destructor swallows — nothing meaningful we can report through
        // a void return.
    }
}

extern "C" int32_t onebit_xrt_load_xclbin(HaloXrtDevice* dev, const char* path) {
    if (dev == nullptr || path == nullptr) {
        return HALO_XRT_E_INVALID;
    }
    // Pre-check existence so we return E_NOT_FOUND instead of the generic
    // E_DEVICE that XRT's internal error would otherwise surface.
    {
        std::ifstream probe(path, std::ios::binary);
        if (!probe.good()) {
            return HALO_XRT_E_NOT_FOUND;
        }
    }
    try {
        auto uuid = dev->device.load_xclbin(path);
        dev->xclbin_uuid = uuid;
        dev->xclbin_loaded = true;
        return HALO_XRT_OK;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "onebit_xrt_load_xclbin(%s) failed: %s\n", path, e.what());
        return HALO_XRT_E_DEVICE;
    } catch (...) {
        return HALO_XRT_E_DEVICE;
    }
}

extern "C" int32_t onebit_xrt_run_kernel(
    HaloXrtDevice* dev,
    const char*    name,
    const uint8_t* in,
    size_t         in_len,
    uint8_t*       out,
    size_t         out_len)
{
    if (dev == nullptr || name == nullptr) {
        return HALO_XRT_E_INVALID;
    }
    if (!dev->xclbin_loaded) {
        return HALO_XRT_E_KERNEL;
    }
    if ((in == nullptr && in_len > 0) || (out == nullptr && out_len > 0)) {
        return HALO_XRT_E_INVALID;
    }
    try {
        auto kernel = xrt::kernel(dev->device, dev->xclbin_uuid, name);

        // Group IDs 0 and 1 are conventional input/output on AIE kernels
        // generated via Peano / mlir-aie. If the kernel signature differs,
        // the caller's xclbin needs a richer shim — this is the MVP path.
        auto bo_in = (in_len > 0)
            ? xrt::bo(dev->device, in_len, xrt::bo::flags::normal, kernel.group_id(0))
            : xrt::bo();
        auto bo_out = (out_len > 0)
            ? xrt::bo(dev->device, out_len, xrt::bo::flags::normal, kernel.group_id(1))
            : xrt::bo();

        if (in_len > 0) {
            bo_in.write(in, in_len, 0);
            bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE, in_len, 0);
        }

        auto run = kernel(bo_in, bo_out);
        run.wait();

        if (out_len > 0) {
            bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, out_len, 0);
            bo_out.read(out, out_len, 0);
        }
        return HALO_XRT_OK;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "onebit_xrt_run_kernel(%s) failed: %s\n", name, e.what());
        return HALO_XRT_E_INTERNAL;
    } catch (...) {
        return HALO_XRT_E_INTERNAL;
    }
}
