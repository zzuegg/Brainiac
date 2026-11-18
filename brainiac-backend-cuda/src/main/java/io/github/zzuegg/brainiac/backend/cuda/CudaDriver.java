package io.github.zzuegg.brainiac.backend.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.List;

/**
 * Bindings to the CUDA driver API ({@code nvcuda.dll}, {@code libcuda.so.1}), which comes
 * with NVIDIA's display driver: no toolkit is needed for it. Only the functions the backend
 * uses are bound. Every call checks its result and throws a {@link CudaException} that names
 * the call and the error.
 *
 * <p>Device pointers ({@code CUdeviceptr}) and handles are {@code long} values.
 */
@SuppressWarnings("restricted")
final class CudaDriver {

    // error codes the backend reacts to
    static final int CUDA_SUCCESS = 0;
    static final int CUDA_ERROR_OUT_OF_MEMORY = 2;
    static final int CUDA_ERROR_NOT_READY = 600;

    // device attributes
    static final int ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8;
    static final int ATTR_MULTIPROCESSOR_COUNT = 16;
    static final int ATTR_COMPUTE_CAPABILITY_MAJOR = 75;
    static final int ATTR_COMPUTE_CAPABILITY_MINOR = 76;

    static final int CU_STREAM_NON_BLOCKING = 1;
    static final int CU_JIT_ERROR_LOG_BUFFER = 5;
    static final int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

    private static final AddressLayout PTR = ADDRESS;

    final String library;

    private final MethodHandle cuInit;
    private final MethodHandle cuDriverGetVersion;
    private final MethodHandle cuDeviceGetCount;
    private final MethodHandle cuDeviceGet;
    private final MethodHandle cuDeviceGetName;
    private final MethodHandle cuDeviceTotalMem;
    private final MethodHandle cuDeviceGetAttribute;
    private final MethodHandle cuDevicePrimaryCtxRetain;
    private final MethodHandle cuCtxSetCurrent;
    private final MethodHandle cuCtxSynchronize;
    private final MethodHandle cuStreamCreate;
    private final MethodHandle cuStreamSynchronize;
    private final MethodHandle cuMemAlloc;
    private final MethodHandle cuMemFree;
    private final MethodHandle cuMemGetInfo;
    private final MethodHandle cuMemAllocHost;
    private final MethodHandle cuMemcpyHtoDAsync;
    private final MethodHandle cuMemcpyDtoHAsync;
    private final MethodHandle cuMemcpyDtoDAsync;
    private final MethodHandle cuMemsetD8Async;
    private final MethodHandle cuMemsetD16Async;
    private final MethodHandle cuMemsetD32Async;
    private final MethodHandle cuModuleLoadDataEx;
    private final MethodHandle cuModuleGetFunction;
    private final MethodHandle cuLaunchKernel;
    private final MethodHandle cuGetErrorName;
    private final MethodHandle cuGetErrorString;

    /**
     * Loads the driver library and binds its functions. Does not call into it.
     *
     * @throws CudaException if there is no NVIDIA driver
     */
    CudaDriver() {
        Binder b = Binder.load("the NVIDIA CUDA driver", "brainiac.cuda.driver",
                Binder.WINDOWS ? List.of("nvcuda.dll", "nvcuda") : List.of("libcuda.so.1", "libcuda.so"), null, null);
        library = b.library();
        cuInit = b.bind("cuInit", FunctionDescriptor.of(JAVA_INT, JAVA_INT), false);
        cuDriverGetVersion = b.bind("cuDriverGetVersion", FunctionDescriptor.of(JAVA_INT, PTR), false);
        cuDeviceGetCount = b.bind("cuDeviceGetCount", FunctionDescriptor.of(JAVA_INT, PTR), false);
        cuDeviceGet = b.bind("cuDeviceGet", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        cuDeviceGetName = b.bind("cuDeviceGetName", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT), false);
        cuDeviceTotalMem = b.bind("cuDeviceTotalMem_v2", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        cuDeviceGetAttribute = b.bind("cuDeviceGetAttribute",
                FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT), false);
        cuDevicePrimaryCtxRetain = b.bind("cuDevicePrimaryCtxRetain",
                FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        cuCtxSetCurrent = b.bind("cuCtxSetCurrent", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        cuCtxSynchronize = b.bind("cuCtxSynchronize", FunctionDescriptor.of(JAVA_INT), false);
        cuStreamCreate = b.bind("cuStreamCreate", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        cuStreamSynchronize = b.bind("cuStreamSynchronize", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        cuMemAlloc = b.bind("cuMemAlloc_v2", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG), false);
        cuMemFree = b.bind("cuMemFree_v2", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        cuMemGetInfo = b.bind("cuMemGetInfo_v2", FunctionDescriptor.of(JAVA_INT, PTR, PTR), false);
        cuMemAllocHost = b.bind("cuMemAllocHost_v2", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG), false);
        cuMemcpyHtoDAsync = b.bind("cuMemcpyHtoDAsync_v2",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, PTR, JAVA_LONG, JAVA_LONG), false);
        cuMemcpyDtoHAsync = b.bind("cuMemcpyDtoHAsync_v2",
                FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG, JAVA_LONG, JAVA_LONG), false);
        cuMemcpyDtoDAsync = b.bind("cuMemcpyDtoDAsync_v2",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG), true);
        cuMemsetD8Async = b.bind("cuMemsetD8Async",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE, JAVA_LONG, JAVA_LONG), true);
        cuMemsetD16Async = b.bind("cuMemsetD16Async",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_SHORT, JAVA_LONG, JAVA_LONG), true);
        cuMemsetD32Async = b.bind("cuMemsetD32Async",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG), true);
        cuModuleLoadDataEx = b.bind("cuModuleLoadDataEx",
                FunctionDescriptor.of(JAVA_INT, PTR, PTR, JAVA_INT, PTR, PTR), false);
        cuModuleGetFunction = b.bind("cuModuleGetFunction", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG, PTR),
                false);
        cuLaunchKernel = b.bind("cuLaunchKernel", FunctionDescriptor.of(JAVA_INT, JAVA_LONG,
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, PTR, PTR), true);
        cuGetErrorName = b.bind("cuGetErrorName", FunctionDescriptor.of(JAVA_INT, JAVA_INT, PTR), false);
        cuGetErrorString = b.bind("cuGetErrorString", FunctionDescriptor.of(JAVA_INT, JAVA_INT, PTR), false);
    }

    // ------------------------------------------------------------------ errors

    /** Throws if {@code result} is an error, naming the call and the error. */
    void check(int result, String call) {
        if (result != CUDA_SUCCESS) {
            throw error(result, call);
        }
    }

    CudaException error(int result, String call) {
        return new CudaException(call + ": " + errorName(result) + " (" + result + "): " + errorString(result), result);
    }

    /** For example {@code CUDA_ERROR_OUT_OF_MEMORY}. */
    String errorName(int result) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS);
            int r = (int) cuGetErrorName.invokeExact(result, p);
            return r == CUDA_SUCCESS ? string(p) : "CUDA error";
        } catch (Throwable t) {
            return "CUDA error";
        }
    }

    private String errorString(int result) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(ADDRESS);
            int r = (int) cuGetErrorString.invokeExact(result, p);
            return r == CUDA_SUCCESS ? string(p) : "unknown error";
        } catch (Throwable t) {
            return "unknown error";
        }
    }

    /** The C string a {@code const char**} out-parameter points to. */
    static String string(MemorySegment pointerToString) {
        MemorySegment s = pointerToString.get(ADDRESS, 0);
        return s.equals(MemorySegment.NULL) ? "" : s.reinterpret(Long.MAX_VALUE).getString(0);
    }

    // ------------------------------------------------------------------ initialization and devices

    void init() {
        int r;
        try {
            r = (int) cuInit.invokeExact(0);
        } catch (Throwable t) {
            throw Binder.fail("cuInit", t);
        }
        check(r, "cuInit");
    }

    /** For example 12040 for CUDA 12.4. */
    int driverVersion() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) cuDriverGetVersion.invokeExact(v), "cuDriverGetVersion");
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuDriverGetVersion", t);
        }
    }

    int deviceCount() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) cuDeviceGetCount.invokeExact(v), "cuDeviceGetCount");
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuDeviceGetCount", t);
        }
    }

    int device(int ordinal) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) cuDeviceGet.invokeExact(v, ordinal), "cuDeviceGet(" + ordinal + ")");
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuDeviceGet", t);
        }
    }

    String deviceName(int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(256);
            check((int) cuDeviceGetName.invokeExact(v, 256, device), "cuDeviceGetName");
            return v.getString(0);
        } catch (Throwable t) {
            throw Binder.fail("cuDeviceGetName", t);
        }
    }

    long totalMemory(int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_LONG);
            check((int) cuDeviceTotalMem.invokeExact(v, device), "cuDeviceTotalMem");
            return v.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuDeviceTotalMem", t);
        }
    }

    int attribute(int device, int attribute) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) cuDeviceGetAttribute.invokeExact(v, attribute, device), "cuDeviceGetAttribute(" + attribute + ")");
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuDeviceGetAttribute", t);
        }
    }

    /** The device's primary context, shared with any other CUDA user in the process. */
    long primaryContext(int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_LONG);
            check((int) cuDevicePrimaryCtxRetain.invokeExact(v, device), "cuDevicePrimaryCtxRetain");
            return v.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuDevicePrimaryCtxRetain", t);
        }
    }

    void setCurrent(long context) {
        int r;
        try {
            r = (int) cuCtxSetCurrent.invokeExact(context);
        } catch (Throwable t) {
            throw Binder.fail("cuCtxSetCurrent", t);
        }
        check(r, "cuCtxSetCurrent");
    }

    void synchronizeContext() {
        int r;
        try {
            r = (int) cuCtxSynchronize.invokeExact();
        } catch (Throwable t) {
            throw Binder.fail("cuCtxSynchronize", t);
        }
        check(r, "cuCtxSynchronize");
    }

    // ------------------------------------------------------------------ streams

    long createStream(int flags) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_LONG);
            check((int) cuStreamCreate.invokeExact(v, flags), "cuStreamCreate");
            return v.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuStreamCreate", t);
        }
    }

    /** Waits for the stream; returns the error code instead of throwing, so callers can add what ran. */
    int synchronize(long stream) {
        try {
            return (int) cuStreamSynchronize.invokeExact(stream);
        } catch (Throwable t) {
            throw Binder.fail("cuStreamSynchronize", t);
        }
    }

    // ------------------------------------------------------------------ memory

    /** Allocates device memory; returns the error code (out of memory is expected) and the pointer in {@code out}. */
    int allocate(long bytes, MemorySegment out) {
        try {
            return (int) cuMemAlloc.invokeExact(out, bytes);
        } catch (Throwable t) {
            throw Binder.fail("cuMemAlloc", t);
        }
    }

    void free(long pointer) {
        int r;
        try {
            r = (int) cuMemFree.invokeExact(pointer);
        } catch (Throwable t) {
            throw Binder.fail("cuMemFree", t);
        }
        check(r, "cuMemFree");
    }

    /** Free and total device memory in bytes. */
    long[] memoryInfo() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment free = a.allocate(JAVA_LONG);
            MemorySegment total = a.allocate(JAVA_LONG);
            check((int) cuMemGetInfo.invokeExact(free, total), "cuMemGetInfo");
            return new long[] {free.get(JAVA_LONG, 0), total.get(JAVA_LONG, 0)};
        } catch (Throwable t) {
            throw Binder.fail("cuMemGetInfo", t);
        }
    }

    /** Page-locked host memory, which the device copies to and from at full speed; never freed. */
    MemorySegment allocateHost(long bytes) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(ADDRESS);
            check((int) cuMemAllocHost.invokeExact(v, bytes), "cuMemAllocHost(" + bytes + " bytes)");
            return v.get(ADDRESS, 0).reinterpret(bytes);
        } catch (Throwable t) {
            throw Binder.fail("cuMemAllocHost", t);
        }
    }

    void copyToDevice(long dst, MemorySegment src, long bytes, long stream) {
        int r;
        try {
            r = (int) cuMemcpyHtoDAsync.invokeExact(dst, src, bytes, stream);
        } catch (Throwable t) {
            throw Binder.fail("cuMemcpyHtoDAsync", t);
        }
        check(r, "cuMemcpyHtoDAsync(" + bytes + " bytes)");
    }

    void copyToHost(MemorySegment dst, long src, long bytes, long stream) {
        int r;
        try {
            r = (int) cuMemcpyDtoHAsync.invokeExact(dst, src, bytes, stream);
        } catch (Throwable t) {
            throw Binder.fail("cuMemcpyDtoHAsync", t);
        }
        check(r, "cuMemcpyDtoHAsync(" + bytes + " bytes)");
    }

    void copy(long dst, long src, long bytes, long stream) {
        int r;
        try {
            r = (int) cuMemcpyDtoDAsync.invokeExact(dst, src, bytes, stream);
        } catch (Throwable t) {
            throw Binder.fail("cuMemcpyDtoDAsync", t);
        }
        check(r, "cuMemcpyDtoDAsync(" + bytes + " bytes)");
    }

    void fill8(long dst, byte value, long count, long stream) {
        int r;
        try {
            r = (int) cuMemsetD8Async.invokeExact(dst, value, count, stream);
        } catch (Throwable t) {
            throw Binder.fail("cuMemsetD8Async", t);
        }
        check(r, "cuMemsetD8Async");
    }

    void fill16(long dst, short value, long count, long stream) {
        int r;
        try {
            r = (int) cuMemsetD16Async.invokeExact(dst, value, count, stream);
        } catch (Throwable t) {
            throw Binder.fail("cuMemsetD16Async", t);
        }
        check(r, "cuMemsetD16Async");
    }

    void fill32(long dst, int value, long count, long stream) {
        int r;
        try {
            r = (int) cuMemsetD32Async.invokeExact(dst, value, count, stream);
        } catch (Throwable t) {
            throw Binder.fail("cuMemsetD32Async", t);
        }
        check(r, "cuMemsetD32Async");
    }

    // ------------------------------------------------------------------ modules and kernels

    /**
     * Loads PTX (compiled by the driver for this device) or a cubin.
     *
     * @param image the PTX text, ending in a zero byte, or the cubin
     * @return the module
     */
    long loadModule(MemorySegment image) {
        try (Arena a = Arena.ofConfined()) {
            int logSize = 16 * 1024;
            MemorySegment log = a.allocate(logSize);
            MemorySegment module = a.allocate(JAVA_LONG);
            MemorySegment options = a.allocate(JAVA_INT, 2);
            options.setAtIndex(JAVA_INT, 0, CU_JIT_ERROR_LOG_BUFFER);
            options.setAtIndex(JAVA_INT, 1, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
            MemorySegment values = a.allocate(ADDRESS, 2);
            values.setAtIndex(ADDRESS, 0, log);
            values.setAtIndex(ADDRESS, 1, MemorySegment.ofAddress(logSize));
            int r = (int) cuModuleLoadDataEx.invokeExact(module, image, 2, options, values);
            if (r != CUDA_SUCCESS) {
                String text = log.getString(0);
                throw new CudaException("cuModuleLoadDataEx: " + errorName(r) + " (" + r + ")"
                        + (text.isBlank() ? "" : ": " + text.strip()), r);
            }
            return module.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuModuleLoadDataEx", t);
        }
    }

    long function(long module, String name) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment f = a.allocate(JAVA_LONG);
            check((int) cuModuleGetFunction.invokeExact(f, module, a.allocateFrom(name)), "cuModuleGetFunction(" + name + ")");
            return f.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("cuModuleGetFunction", t);
        }
    }

    /**
     * Queues a kernel. Returns the error code, so the caller can name the kernel.
     *
     * @param params a {@code void*[]} with a pointer to each argument's value; the values are
     *        copied before this returns
     */
    int launch(long function, int gridX, int gridY, int gridZ, int blockX, int sharedBytes, long stream,
            MemorySegment params) {
        try {
            return (int) cuLaunchKernel.invokeExact(function, gridX, gridY, gridZ, blockX, 1, 1, sharedBytes, stream,
                    params, MemorySegment.NULL);
        } catch (Throwable t) {
            throw Binder.fail("cuLaunchKernel", t);
        }
    }
}
