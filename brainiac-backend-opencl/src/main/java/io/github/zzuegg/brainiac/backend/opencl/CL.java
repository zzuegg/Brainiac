package io.github.zzuegg.brainiac.backend.opencl;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The few OpenCL 1.2 functions the backend needs, called through the Foreign Function and
 * Memory API. Handles ({@code cl_context}, {@code cl_mem}...) are {@link MemorySegment}
 * addresses; {@code size_t} is 64 bits.
 */
final class CL {

    static final int SUCCESS = 0;
    static final long DEVICE_TYPE_CPU = 1L << 1;
    static final long DEVICE_TYPE_GPU = 1L << 2;
    static final long DEVICE_TYPE_ACCELERATOR = 1L << 3;
    static final long DEVICE_TYPE_ALL = 0xFFFFFFFFL;
    static final int DEVICE_TYPE = 0x1000;
    static final int DEVICE_MAX_COMPUTE_UNITS = 0x1002;
    static final int DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
    static final int DEVICE_MAX_CLOCK_FREQUENCY = 0x100C;
    static final int DEVICE_HOST_UNIFIED_MEMORY = 0x1035;
    static final int DEVICE_GLOBAL_MEM_SIZE = 0x101F;
    static final int DEVICE_NAME = 0x102B;
    static final int DEVICE_EXTENSIONS = 0x1030;
    static final int PLATFORM_NAME = 0x0902;
    static final int PROGRAM_BUILD_LOG = 0x1183;
    static final long MEM_READ_WRITE = 1L;
    static final long MEM_READ_ONLY = 1L << 2;
    static final long MEM_COPY_HOST_PTR = 1L << 5;
    static final int OUT_OF_RESOURCES = -5;
    static final int MEM_OBJECT_ALLOCATION_FAILURE = -4;

    private static final Map<Integer, String> ERRORS = Map.ofEntries(
            Map.entry(-1, "DEVICE_NOT_FOUND"), Map.entry(-2, "DEVICE_NOT_AVAILABLE"),
            Map.entry(-3, "COMPILER_NOT_AVAILABLE"), Map.entry(-4, "MEM_OBJECT_ALLOCATION_FAILURE"),
            Map.entry(-5, "OUT_OF_RESOURCES"), Map.entry(-6, "OUT_OF_HOST_MEMORY"),
            Map.entry(-11, "BUILD_PROGRAM_FAILURE"), Map.entry(-30, "INVALID_VALUE"),
            Map.entry(-32, "INVALID_PLATFORM"), Map.entry(-33, "INVALID_DEVICE"), Map.entry(-34, "INVALID_CONTEXT"),
            Map.entry(-36, "INVALID_COMMAND_QUEUE"), Map.entry(-38, "INVALID_MEM_OBJECT"),
            Map.entry(-44, "INVALID_PROGRAM"), Map.entry(-46, "INVALID_KERNEL_NAME"), Map.entry(-48, "INVALID_KERNEL"),
            Map.entry(-49, "INVALID_ARG_INDEX"), Map.entry(-50, "INVALID_ARG_VALUE"),
            Map.entry(-51, "INVALID_ARG_SIZE"), Map.entry(-52, "INVALID_KERNEL_ARGS"),
            Map.entry(-54, "INVALID_WORK_GROUP_SIZE"), Map.entry(-55, "INVALID_WORK_ITEM_SIZE"),
            Map.entry(-61, "INVALID_BUFFER_SIZE"), Map.entry(-63, "INVALID_GLOBAL_WORK_SIZE"),
            Map.entry(-1001, "PLATFORM_NOT_FOUND_KHR"));

    private final MethodHandle getPlatformIDs;
    private final MethodHandle getPlatformInfo;
    private final MethodHandle getDeviceIDs;
    private final MethodHandle getDeviceInfo;
    private final MethodHandle createContext;
    private final MethodHandle createCommandQueue;
    private final MethodHandle createBuffer;
    private final MethodHandle releaseMemObject;
    private final MethodHandle enqueueWriteBuffer;
    private final MethodHandle enqueueReadBuffer;
    private final MethodHandle enqueueCopyBuffer;
    private final MethodHandle createProgramWithSource;
    private final MethodHandle buildProgram;
    private final MethodHandle getProgramBuildInfo;
    private final MethodHandle createKernel;
    private final MethodHandle setKernelArg;
    private final MethodHandle enqueueNDRangeKernel;
    private final MethodHandle finish;

    private CL(SymbolLookup lib) {
        Linker linker = Linker.nativeLinker();
        getPlatformIDs = handle(linker, lib, "clGetPlatformIDs", JAVA_INT, JAVA_INT, ADDRESS, ADDRESS);
        getPlatformInfo = handle(linker, lib, "clGetPlatformInfo", JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS,
                ADDRESS);
        getDeviceIDs = handle(linker, lib, "clGetDeviceIDs", JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS);
        getDeviceInfo = handle(linker, lib, "clGetDeviceInfo", JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS,
                ADDRESS);
        createContext = handle(linker, lib, "clCreateContext", ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS,
                ADDRESS);
        createCommandQueue = handle(linker, lib, "clCreateCommandQueue", ADDRESS, ADDRESS, ADDRESS, JAVA_LONG,
                ADDRESS);
        createBuffer = handle(linker, lib, "clCreateBuffer", ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS);
        releaseMemObject = handle(linker, lib, "clReleaseMemObject", JAVA_INT, ADDRESS);
        enqueueWriteBuffer = handle(linker, lib, "clEnqueueWriteBuffer", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        enqueueReadBuffer = handle(linker, lib, "clEnqueueReadBuffer", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        enqueueCopyBuffer = handle(linker, lib, "clEnqueueCopyBuffer", JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG,
                JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS);
        createProgramWithSource = handle(linker, lib, "clCreateProgramWithSource", ADDRESS, ADDRESS, JAVA_INT,
                ADDRESS, ADDRESS, ADDRESS);
        buildProgram = handle(linker, lib, "clBuildProgram", JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS,
                ADDRESS);
        getProgramBuildInfo = handle(linker, lib, "clGetProgramBuildInfo", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                JAVA_LONG, ADDRESS, ADDRESS);
        createKernel = handle(linker, lib, "clCreateKernel", ADDRESS, ADDRESS, ADDRESS, ADDRESS);
        setKernelArg = handle(linker, lib, "clSetKernelArg", JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS);
        enqueueNDRangeKernel = handle(linker, lib, "clEnqueueNDRangeKernel", JAVA_INT, ADDRESS, ADDRESS, JAVA_INT,
                ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        finish = handle(linker, lib, "clFinish", JAVA_INT, ADDRESS);
    }

    /**
     * Loads the system's OpenCL library (the ICD loader).
     *
     * @return the bindings
     * @throws IllegalStateException if no OpenCL library is installed
     */
    static CL load() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        List<String> names = new ArrayList<>();
        String custom = System.getProperty("brainiac.opencl.library");
        if (custom != null) {
            names.add(custom);
        }
        if (os.contains("win")) {
            names.add("OpenCL.dll");
        } else if (os.contains("mac")) {
            names.add("/System/Library/Frameworks/OpenCL.framework/OpenCL");
        } else {
            names.add("libOpenCL.so.1");
            names.add("libOpenCL.so");
        }
        RuntimeException last = null;
        for (String name : names) {
            try {
                return new CL(SymbolLookup.libraryLookup(name, Arena.global()));
            } catch (IllegalArgumentException e) {
                last = e;
            }
        }
        throw new IllegalStateException("no OpenCL library found (tried " + names + ")", last);
    }

    private static MethodHandle handle(Linker linker, SymbolLookup lib, String name,
                                       java.lang.foreign.MemoryLayout result,
                                       java.lang.foreign.MemoryLayout... args) {
        MemorySegment symbol = lib.find(name)
                .orElseThrow(() -> new IllegalStateException("OpenCL library has no " + name));
        return linker.downcallHandle(symbol, FunctionDescriptor.of(result, args));
    }

    static void check(int code, String what) {
        if (code != SUCCESS) {
            throw new IllegalStateException(what + " failed: " + name(code));
        }
    }

    static String name(int code) {
        return "CL_" + ERRORS.getOrDefault(code, "ERROR") + " (" + code + ")";
    }

    // ---------------------------------------------------------------- calls

    List<MemorySegment> platforms() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment count = arena.allocate(JAVA_INT);
            int code = (int) getPlatformIDs.invokeExact(0, MemorySegment.NULL, count);
            if (code == -1001) {
                return List.of();
            }
            check(code, "clGetPlatformIDs");
            int n = count.get(JAVA_INT, 0);
            if (n == 0) {
                return List.of();
            }
            MemorySegment ids = arena.allocate(ADDRESS, n);
            check((int) getPlatformIDs.invokeExact(n, ids, MemorySegment.NULL), "clGetPlatformIDs");
            return addresses(ids, n);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    String platformName(MemorySegment platform) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment size = arena.allocate(JAVA_LONG);
            check((int) getPlatformInfo.invokeExact(platform, PLATFORM_NAME, 0L, MemorySegment.NULL, size),
                    "clGetPlatformInfo");
            MemorySegment value = arena.allocate(size.get(JAVA_LONG, 0));
            check((int) getPlatformInfo.invokeExact(platform, PLATFORM_NAME, value.byteSize(), value,
                    MemorySegment.NULL), "clGetPlatformInfo");
            return value.getString(0).trim();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    List<MemorySegment> devices(MemorySegment platform, long type) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment count = arena.allocate(JAVA_INT);
            int code = (int) getDeviceIDs.invokeExact(platform, type, 0, MemorySegment.NULL, count);
            if (code == -1) {
                return List.of();
            }
            check(code, "clGetDeviceIDs");
            int n = count.get(JAVA_INT, 0);
            MemorySegment ids = arena.allocate(ADDRESS, Math.max(n, 1));
            check((int) getDeviceIDs.invokeExact(platform, type, n, ids, MemorySegment.NULL), "clGetDeviceIDs");
            return addresses(ids, n);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    String deviceString(MemorySegment device, int param) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment size = arena.allocate(JAVA_LONG);
            check((int) getDeviceInfo.invokeExact(device, param, 0L, MemorySegment.NULL, size), "clGetDeviceInfo");
            MemorySegment value = arena.allocate(Math.max(1, size.get(JAVA_LONG, 0)));
            check((int) getDeviceInfo.invokeExact(device, param, value.byteSize(), value, MemorySegment.NULL),
                    "clGetDeviceInfo");
            return value.getString(0).trim();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    long deviceLong(MemorySegment device, int param) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment value = arena.allocate(JAVA_LONG);
            check((int) getDeviceInfo.invokeExact(device, param, 8L, value, MemorySegment.NULL), "clGetDeviceInfo");
            return value.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    MemorySegment createContext(MemorySegment device) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment devices = arena.allocate(ADDRESS);
            devices.set(ADDRESS, 0, device);
            MemorySegment err = arena.allocate(JAVA_INT);
            MemorySegment ctx = (MemorySegment) createContext.invokeExact(MemorySegment.NULL, 1, devices,
                    MemorySegment.NULL, MemorySegment.NULL, err);
            check(err.get(JAVA_INT, 0), "clCreateContext");
            return ctx;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    MemorySegment createQueue(MemorySegment context, MemorySegment device) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = arena.allocate(JAVA_INT);
            MemorySegment q = (MemorySegment) createCommandQueue.invokeExact(context, device, 0L, err);
            check(err.get(JAVA_INT, 0), "clCreateCommandQueue");
            return q;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Creates a buffer.
     *
     * @return the buffer, or {@code null} if the device is out of memory
     */
    MemorySegment createBuffer(MemorySegment context, long flags, long bytes, MemorySegment host) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = arena.allocate(JAVA_INT);
            MemorySegment mem = (MemorySegment) createBuffer.invokeExact(context, flags, bytes, host, err);
            int code = err.get(JAVA_INT, 0);
            if (code == MEM_OBJECT_ALLOCATION_FAILURE || code == OUT_OF_RESOURCES || code == -6) {
                return null;
            }
            check(code, "clCreateBuffer of " + bytes + " bytes");
            return mem;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void release(MemorySegment mem) {
        try {
            int code = (int) releaseMemObject.invokeExact(mem);
            check(code, "clReleaseMemObject");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void write(MemorySegment queue, MemorySegment buffer, long offset, MemorySegment data) {
        write(queue, buffer, offset, data, true);
    }

    /** Enqueues a copy to the device; without blocking, {@code data} must stay alive until the queue is done. */
    void write(MemorySegment queue, MemorySegment buffer, long offset, MemorySegment data, boolean blocking) {
        try {
            check((int) enqueueWriteBuffer.invokeExact(queue, buffer, blocking ? 1 : 0, offset, data.byteSize(), data,
                    0,
                    MemorySegment.NULL, MemorySegment.NULL), "clEnqueueWriteBuffer");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void read(MemorySegment queue, MemorySegment buffer, long offset, MemorySegment data) {
        try {
            check((int) enqueueReadBuffer.invokeExact(queue, buffer, 1, offset, data.byteSize(), data, 0,
                    MemorySegment.NULL, MemorySegment.NULL), "clEnqueueReadBuffer");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void copy(MemorySegment queue, MemorySegment src, MemorySegment dst, long bytes) {
        try {
            check((int) enqueueCopyBuffer.invokeExact(queue, src, dst, 0L, 0L, bytes, 0, MemorySegment.NULL,
                    MemorySegment.NULL), "clEnqueueCopyBuffer");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    MemorySegment buildProgram(MemorySegment context, MemorySegment device, String source, String options) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment text = arena.allocateFrom(source);
            MemorySegment strings = arena.allocate(ADDRESS);
            strings.set(ADDRESS, 0, text);
            MemorySegment err = arena.allocate(JAVA_INT);
            MemorySegment program = (MemorySegment) createProgramWithSource.invokeExact(context, 1, strings,
                    MemorySegment.NULL, err);
            check(err.get(JAVA_INT, 0), "clCreateProgramWithSource");
            MemorySegment devices = arena.allocate(ADDRESS);
            devices.set(ADDRESS, 0, device);
            int code = (int) buildProgram.invokeExact(program, 1, devices, arena.allocateFrom(options),
                    MemorySegment.NULL, MemorySegment.NULL);
            if (code != SUCCESS) {
                throw new IllegalStateException("building the OpenCL kernels failed: " + name(code) + "\n"
                        + buildLog(program, device));
            }
            return program;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private String buildLog(MemorySegment program, MemorySegment device) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment size = arena.allocate(JAVA_LONG);
            int code = (int) getProgramBuildInfo.invokeExact(program, device, PROGRAM_BUILD_LOG, 0L,
                    MemorySegment.NULL, size);
            if (code != SUCCESS) {
                return "(no build log)";
            }
            MemorySegment log = arena.allocate(Math.max(1, size.get(JAVA_LONG, 0)));
            code = (int) getProgramBuildInfo.invokeExact(program, device, PROGRAM_BUILD_LOG, log.byteSize(), log,
                    MemorySegment.NULL);
            return code == SUCCESS ? log.getString(0) : "(no build log)";
        }
    }

    MemorySegment createKernel(MemorySegment program, String name) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = arena.allocate(JAVA_INT);
            MemorySegment kernel = (MemorySegment) createKernel.invokeExact(program, arena.allocateFrom(name), err);
            check(err.get(JAVA_INT, 0), "clCreateKernel " + name);
            return kernel;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void setArg(MemorySegment kernel, int index, MemorySegment value) {
        try {
            check((int) setKernelArg.invokeExact(kernel, index, value.byteSize(), value), "clSetKernelArg " + index);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void run(MemorySegment queue, MemorySegment kernel, long[] global, long[] local) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment g = arena.allocate(JAVA_LONG, global.length);
            for (int i = 0; i < global.length; i++) {
                g.setAtIndex(JAVA_LONG, i, global[i]);
            }
            MemorySegment l = MemorySegment.NULL;
            if (local != null) {
                l = arena.allocate(JAVA_LONG, local.length);
                for (int i = 0; i < local.length; i++) {
                    l.setAtIndex(JAVA_LONG, i, local[i]);
                }
            }
            check((int) enqueueNDRangeKernel.invokeExact(queue, kernel, global.length, MemorySegment.NULL, g, l, 0,
                    MemorySegment.NULL, MemorySegment.NULL), "clEnqueueNDRangeKernel");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    void finish(MemorySegment queue) {
        try {
            check((int) finish.invokeExact(queue), "clFinish");
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static List<MemorySegment> addresses(MemorySegment ids, int n) {
        List<MemorySegment> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(ids.getAtIndex(ADDRESS, i));
        }
        return out;
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) {
            return r;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new IllegalStateException(t);
    }
}
