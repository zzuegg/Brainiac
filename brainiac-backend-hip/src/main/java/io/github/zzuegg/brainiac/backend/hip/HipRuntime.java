package io.github.zzuegg.brainiac.backend.hip;

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
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Bindings to the HIP runtime ({@code amdhip64_7.dll}, {@code libamdhip64.so.7}), AMD's
 * counterpart of the CUDA driver API. It comes with AMD's display driver and with the HIP SDK
 * (ROCm on Linux). Only the functions the backend uses are bound. Every call checks its result
 * and throws a {@link HipException} that names the call and the error.
 *
 * <p>Device pointers and handles (streams, modules, functions) are {@code long} values.
 */
@SuppressWarnings("restricted")
final class HipRuntime {

    // error codes the backend reacts to
    static final int HIP_SUCCESS = 0;
    static final int HIP_ERROR_OUT_OF_MEMORY = 2;
    static final int HIP_ERROR_NOT_READY = 600;

    static final int HIP_STREAM_NON_BLOCKING = 1;

    private static final AddressLayout PTR = ADDRESS;

    final String library;

    private final MethodHandle hipInit;
    private final MethodHandle hipDriverGetVersion;
    private final MethodHandle hipRuntimeGetVersion;
    private final MethodHandle hipGetDeviceCount;
    private final MethodHandle hipDeviceGet;
    private final MethodHandle hipDeviceGetName;
    private final MethodHandle hipDeviceTotalMem;
    private final MethodHandle hipGetDeviceProperties;
    private final MethodHandle hipSetDevice;
    private final MethodHandle hipDeviceSynchronize;
    private final MethodHandle hipStreamCreateWithFlags;
    private final MethodHandle hipStreamSynchronize;
    private final MethodHandle hipMalloc;
    private final MethodHandle hipFree;
    private final MethodHandle hipMemGetInfo;
    private final MethodHandle hipHostMalloc;
    private final MethodHandle hipMemcpyHtoDAsync;
    private final MethodHandle hipMemcpyDtoHAsync;
    private final MethodHandle hipMemcpyDtoDAsync;
    private final MethodHandle hipMemsetD8Async;
    private final MethodHandle hipMemsetD16Async;
    private final MethodHandle hipMemsetD32Async;
    private final MethodHandle hipModuleLoadData;
    private final MethodHandle hipModuleGetFunction;
    private final MethodHandle hipModuleLaunchKernel;
    private final MethodHandle hipGetErrorName;
    private final MethodHandle hipGetErrorString;

    /**
     * Loads the HIP runtime and binds its functions. Does not call into it.
     *
     * @throws HipException if there is no HIP runtime
     */
    HipRuntime() {
        Binder b = Binder.load("the AMD HIP runtime", "brainiac.hip.runtime",
                Binder.WINDOWS ? List.of("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll")
                        : List.of("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"),
                Binder.WINDOWS ? Pattern.compile("amdhip64_(\\d+)\\.dll") : Pattern.compile("libamdhip64\\.so\\.([0-9.]+)"),
                null);
        library = b.library();
        hipInit = b.bind("hipInit", FunctionDescriptor.of(JAVA_INT, JAVA_INT), false);
        hipDriverGetVersion = b.bind("hipDriverGetVersion", FunctionDescriptor.of(JAVA_INT, PTR), false);
        hipRuntimeGetVersion = b.bind("hipRuntimeGetVersion", FunctionDescriptor.of(JAVA_INT, PTR), false);
        hipGetDeviceCount = b.bind("hipGetDeviceCount", FunctionDescriptor.of(JAVA_INT, PTR), false);
        hipDeviceGet = b.bind("hipDeviceGet", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        hipDeviceGetName = b.bind("hipDeviceGetName", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT, JAVA_INT), false);
        hipDeviceTotalMem = b.bind("hipDeviceTotalMem", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        MethodHandle props = b.optional("hipGetDevicePropertiesR0600", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT));
        hipGetDeviceProperties = props != null ? props
                : b.bind("hipGetDeviceProperties", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT), false);
        hipSetDevice = b.bind("hipSetDevice", FunctionDescriptor.of(JAVA_INT, JAVA_INT), false);
        hipDeviceSynchronize = b.bind("hipDeviceSynchronize", FunctionDescriptor.of(JAVA_INT), false);
        hipStreamCreateWithFlags = b.bind("hipStreamCreateWithFlags", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_INT),
                false);
        hipStreamSynchronize = b.bind("hipStreamSynchronize", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        hipMalloc = b.bind("hipMalloc", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG), false);
        hipFree = b.bind("hipFree", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        hipMemGetInfo = b.bind("hipMemGetInfo", FunctionDescriptor.of(JAVA_INT, PTR, PTR), false);
        MethodHandle host = b.optional("hipHostMalloc", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG, JAVA_INT));
        hipHostMalloc = host != null ? host
                : b.bind("hipHostAlloc", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG, JAVA_INT), false);
        hipMemcpyHtoDAsync = b.bind("hipMemcpyHtoDAsync",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, PTR, JAVA_LONG, JAVA_LONG), false);
        hipMemcpyDtoHAsync = b.bind("hipMemcpyDtoHAsync",
                FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG, JAVA_LONG, JAVA_LONG), false);
        hipMemcpyDtoDAsync = b.bind("hipMemcpyDtoDAsync",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG), true);
        hipMemsetD8Async = b.bind("hipMemsetD8Async",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_BYTE, JAVA_LONG, JAVA_LONG), true);
        hipMemsetD16Async = b.bind("hipMemsetD16Async",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_SHORT, JAVA_LONG, JAVA_LONG), true);
        hipMemsetD32Async = b.bind("hipMemsetD32Async",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG), true);
        hipModuleLoadData = b.bind("hipModuleLoadData", FunctionDescriptor.of(JAVA_INT, PTR, PTR), false);
        hipModuleGetFunction = b.bind("hipModuleGetFunction", FunctionDescriptor.of(JAVA_INT, PTR, JAVA_LONG, PTR),
                false);
        hipModuleLaunchKernel = b.bind("hipModuleLaunchKernel", FunctionDescriptor.of(JAVA_INT, JAVA_LONG,
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, PTR, PTR), true);
        hipGetErrorName = b.bind("hipGetErrorName", FunctionDescriptor.of(PTR, JAVA_INT), false);
        hipGetErrorString = b.bind("hipGetErrorString", FunctionDescriptor.of(PTR, JAVA_INT), false);
    }

    // ------------------------------------------------------------------ errors

    /** Throws if {@code result} is an error, naming the call and the error. */
    void check(int result, String call) {
        if (result != HIP_SUCCESS) {
            throw error(result, call);
        }
    }

    HipException error(int result, String call) {
        return new HipException(call + ": " + errorName(result) + " (" + result + "): " + errorString(result), result);
    }

    /** For example {@code hipErrorOutOfMemory}. */
    String errorName(int result) {
        try {
            return cString((MemorySegment) hipGetErrorName.invokeExact(result), "HIP error");
        } catch (Throwable t) {
            return "HIP error";
        }
    }

    private String errorString(int result) {
        try {
            return cString((MemorySegment) hipGetErrorString.invokeExact(result), "unknown error");
        } catch (Throwable t) {
            return "unknown error";
        }
    }

    private static String cString(MemorySegment s, String otherwise) {
        return s.equals(MemorySegment.NULL) ? otherwise : s.reinterpret(Long.MAX_VALUE).getString(0);
    }

    // ------------------------------------------------------------------ initialization and devices

    void init() {
        int r;
        try {
            r = (int) hipInit.invokeExact(0);
        } catch (Throwable t) {
            throw Binder.fail("hipInit", t);
        }
        check(r, "hipInit");
    }

    /** The driver's version number, for example 60443483. */
    int driverVersion() {
        return version(hipDriverGetVersion, "hipDriverGetVersion");
    }

    /** The runtime's version number: major * 10000000 + minor * 100000 + patch. */
    int runtimeVersion() {
        return version(hipRuntimeGetVersion, "hipRuntimeGetVersion");
    }

    private int version(MethodHandle h, String call) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) h.invokeExact(v), call);
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail(call, t);
        }
    }

    /** A HIP version number as text, for example "7.2.26015". */
    static String versionText(int v) {
        return v / 10_000_000 + "." + v / 100_000 % 100 + "." + v % 100_000;
    }

    int deviceCount() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) hipGetDeviceCount.invokeExact(v), "hipGetDeviceCount");
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("hipGetDeviceCount", t);
        }
    }

    int device(int ordinal) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) hipDeviceGet.invokeExact(v, ordinal), "hipDeviceGet(" + ordinal + ")");
            return v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("hipDeviceGet", t);
        }
    }

    String deviceName(int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(256);
            check((int) hipDeviceGetName.invokeExact(v, 256, device), "hipDeviceGetName");
            return v.getString(0);
        } catch (Throwable t) {
            throw Binder.fail("hipDeviceGetName", t);
        }
    }

    long totalMemory(int device) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_LONG);
            check((int) hipDeviceTotalMem.invokeExact(v, device), "hipDeviceTotalMem");
            return v.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("hipDeviceTotalMem", t);
        }
    }

    /**
     * What the backend needs from {@code hipDeviceProp_t}.
     *
     * @param architecture the GPU's target name for the compiler, such as {@code gfx1201}
     * @param warpSize lanes per wavefront (32 on RDNA), or 0 if unknown
     * @param multiprocessors compute units, or 0 if unknown
     */
    record Properties(String architecture, int warpSize, int multiprocessors) {
    }

    // offsets in hipDeviceProp_t (R0600, laid out like CUDA 12's cudaDeviceProp)
    private static final int PROP_TOTAL_GLOBAL_MEM = 288;
    private static final int PROP_WARP_SIZE = 308;
    private static final int PROP_MULTIPROCESSORS = 388;
    private static final Pattern GFX = Pattern.compile("gfx[0-9a-f]+(:[a-z]+[+-])*");

    /**
     * Reads the device's properties. The structure changes between HIP versions, so only what
     * can be checked is used: the architecture name is found by its {@code gfx} prefix, and the
     * numeric fields only when the total memory sits where the expected layout puts it.
     */
    Properties properties(int device, long totalMemory) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = a.allocate(8192, 16);
            check((int) hipGetDeviceProperties.invokeExact(p, device), "hipGetDeviceProperties");
            String arch = null;
            byte[] bytes = p.toArray(JAVA_BYTE);
            for (int i = 256; i + 3 < bytes.length && arch == null; i++) {
                if (bytes[i] == 'g' && bytes[i + 1] == 'f' && bytes[i + 2] == 'x' && bytes[i - 1] == 0) {
                    int end = i;
                    while (end < bytes.length && bytes[end] != 0) {
                        end++;
                    }
                    String s = new String(bytes, i, end - i, StandardCharsets.US_ASCII);
                    if (GFX.matcher(s).matches()) {
                        arch = s;
                    }
                }
            }
            int warp = 0;
            int units = 0;
            if (p.get(JAVA_LONG, PROP_TOTAL_GLOBAL_MEM) == totalMemory) {
                warp = p.get(JAVA_INT, PROP_WARP_SIZE);
                units = p.get(JAVA_INT, PROP_MULTIPROCESSORS);
                if (warp != 32 && warp != 64) {
                    warp = 0;
                }
                if (units <= 0 || units > 4096) {
                    units = 0;
                }
            }
            String named = System.getProperty("brainiac.hip.arch");
            if (named != null && !named.isBlank()) {
                arch = named.strip();
            }
            if (arch == null) {
                throw new HipException("cannot tell the GPU's architecture from hipGetDeviceProperties; name it with "
                        + "-Dbrainiac.hip.arch (such as gfx1201)", 0);
            }
            return new Properties(arch, warp, units);
        } catch (Throwable t) {
            throw Binder.fail("hipGetDeviceProperties", t);
        }
    }

    /** Makes the device current on the calling thread. */
    void setDevice(int ordinal) {
        int r;
        try {
            r = (int) hipSetDevice.invokeExact(ordinal);
        } catch (Throwable t) {
            throw Binder.fail("hipSetDevice", t);
        }
        check(r, "hipSetDevice");
    }

    void synchronizeDevice() {
        int r;
        try {
            r = (int) hipDeviceSynchronize.invokeExact();
        } catch (Throwable t) {
            throw Binder.fail("hipDeviceSynchronize", t);
        }
        check(r, "hipDeviceSynchronize");
    }

    // ------------------------------------------------------------------ streams

    long createStream(int flags) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_LONG);
            check((int) hipStreamCreateWithFlags.invokeExact(v, flags), "hipStreamCreateWithFlags");
            return v.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("hipStreamCreateWithFlags", t);
        }
    }

    /** Waits for the stream; returns the error code instead of throwing, so callers can add what ran. */
    int synchronize(long stream) {
        try {
            return (int) hipStreamSynchronize.invokeExact(stream);
        } catch (Throwable t) {
            throw Binder.fail("hipStreamSynchronize", t);
        }
    }

    // ------------------------------------------------------------------ memory

    /** Allocates device memory; returns the error code (out of memory is expected) and the pointer in {@code out}. */
    int allocate(long bytes, MemorySegment out) {
        try {
            return (int) hipMalloc.invokeExact(out, bytes);
        } catch (Throwable t) {
            throw Binder.fail("hipMalloc", t);
        }
    }

    void free(long pointer) {
        int r;
        try {
            r = (int) hipFree.invokeExact(pointer);
        } catch (Throwable t) {
            throw Binder.fail("hipFree", t);
        }
        check(r, "hipFree");
    }

    /** Free and total device memory in bytes. */
    long[] memoryInfo() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment free = a.allocate(JAVA_LONG);
            MemorySegment total = a.allocate(JAVA_LONG);
            check((int) hipMemGetInfo.invokeExact(free, total), "hipMemGetInfo");
            return new long[] {free.get(JAVA_LONG, 0), total.get(JAVA_LONG, 0)};
        } catch (Throwable t) {
            throw Binder.fail("hipMemGetInfo", t);
        }
    }

    /** Page-locked host memory, which the device copies to and from at full speed; never freed. */
    MemorySegment allocateHost(long bytes) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(ADDRESS);
            check((int) hipHostMalloc.invokeExact(v, bytes, 0), "hipHostMalloc(" + bytes + " bytes)");
            return v.get(ADDRESS, 0).reinterpret(bytes);
        } catch (Throwable t) {
            throw Binder.fail("hipHostMalloc", t);
        }
    }

    void copyToDevice(long dst, MemorySegment src, long bytes, long stream) {
        int r;
        try {
            r = (int) hipMemcpyHtoDAsync.invokeExact(dst, src, bytes, stream);
        } catch (Throwable t) {
            throw Binder.fail("hipMemcpyHtoDAsync", t);
        }
        check(r, "hipMemcpyHtoDAsync(" + bytes + " bytes)");
    }

    void copyToHost(MemorySegment dst, long src, long bytes, long stream) {
        int r;
        try {
            r = (int) hipMemcpyDtoHAsync.invokeExact(dst, src, bytes, stream);
        } catch (Throwable t) {
            throw Binder.fail("hipMemcpyDtoHAsync", t);
        }
        check(r, "hipMemcpyDtoHAsync(" + bytes + " bytes)");
    }

    void copy(long dst, long src, long bytes, long stream) {
        int r;
        try {
            r = (int) hipMemcpyDtoDAsync.invokeExact(dst, src, bytes, stream);
        } catch (Throwable t) {
            throw Binder.fail("hipMemcpyDtoDAsync", t);
        }
        check(r, "hipMemcpyDtoDAsync(" + bytes + " bytes)");
    }

    void fill8(long dst, byte value, long count, long stream) {
        int r;
        try {
            r = (int) hipMemsetD8Async.invokeExact(dst, value, count, stream);
        } catch (Throwable t) {
            throw Binder.fail("hipMemsetD8Async", t);
        }
        check(r, "hipMemsetD8Async");
    }

    void fill16(long dst, short value, long count, long stream) {
        int r;
        try {
            r = (int) hipMemsetD16Async.invokeExact(dst, value, count, stream);
        } catch (Throwable t) {
            throw Binder.fail("hipMemsetD16Async", t);
        }
        check(r, "hipMemsetD16Async");
    }

    void fill32(long dst, int value, long count, long stream) {
        int r;
        try {
            r = (int) hipMemsetD32Async.invokeExact(dst, value, count, stream);
        } catch (Throwable t) {
            throw Binder.fail("hipMemsetD32Async", t);
        }
        check(r, "hipMemsetD32Async");
    }

    // ------------------------------------------------------------------ modules and kernels

    /**
     * Loads a code object compiled by hipRTC for this device.
     *
     * @param image the code object
     * @return the module
     */
    long loadModule(MemorySegment image) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment module = a.allocate(JAVA_LONG);
            check((int) hipModuleLoadData.invokeExact(module, image), "hipModuleLoadData");
            return module.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("hipModuleLoadData", t);
        }
    }

    long function(long module, String name) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment f = a.allocate(JAVA_LONG);
            check((int) hipModuleGetFunction.invokeExact(f, module, a.allocateFrom(name)),
                    "hipModuleGetFunction(" + name + ")");
            return f.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw Binder.fail("hipModuleGetFunction", t);
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
            return (int) hipModuleLaunchKernel.invokeExact(function, gridX, gridY, gridZ, blockX, 1, 1, sharedBytes,
                    stream, params, MemorySegment.NULL);
        } catch (Throwable t) {
            throw Binder.fail("hipModuleLaunchKernel", t);
        }
    }
}
