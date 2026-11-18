package io.github.zzuegg.brainiac.backend.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Bindings to NVRTC, CUDA's runtime compiler, which turns the backend's CUDA C++ kernels into
 * PTX for the device at hand. It comes with the CUDA toolkit (and NVIDIA's
 * {@code cuda-nvrtc} packages); {@code -Dbrainiac.cuda.nvrtc} names the file.
 */
@SuppressWarnings("restricted")
final class Nvrtc {

    private static final int NVRTC_SUCCESS = 0;

    final String library;

    private final MethodHandle nvrtcVersion;
    private final MethodHandle nvrtcGetNumSupportedArchs;
    private final MethodHandle nvrtcGetSupportedArchs;
    private final MethodHandle nvrtcCreateProgram;
    private final MethodHandle nvrtcCompileProgram;
    private final MethodHandle nvrtcGetPTXSize;
    private final MethodHandle nvrtcGetPTX;
    private final MethodHandle nvrtcGetProgramLogSize;
    private final MethodHandle nvrtcGetProgramLog;
    private final MethodHandle nvrtcDestroyProgram;
    private final MethodHandle nvrtcGetErrorString;

    /** The result of a compilation: PTX, ending in a zero byte, and the compiler's messages. */
    record Ptx(byte[] code, String log) {
    }

    Nvrtc() {
        Binder b = Binder.load("NVRTC (the CUDA runtime compiler)", "brainiac.cuda.nvrtc",
                Binder.WINDOWS
                        ? List.of("nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll")
                        : List.of("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"),
                Binder.WINDOWS ? Pattern.compile("nvrtc64_(\\d+)_0\\.dll") : Pattern.compile("libnvrtc\\.so\\.([0-9.]+)"),
                null);
        library = b.library();
        FunctionDescriptor ptrPtr = FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS);
        nvrtcVersion = b.bind("nvrtcVersion", ptrPtr, false);
        nvrtcGetNumSupportedArchs = b.optional("nvrtcGetNumSupportedArchs", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        nvrtcGetSupportedArchs = b.optional("nvrtcGetSupportedArchs", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        nvrtcCreateProgram = b.bind("nvrtcCreateProgram",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS), false);
        nvrtcCompileProgram = b.bind("nvrtcCompileProgram",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS), false);
        nvrtcGetPTXSize = b.bind("nvrtcGetPTXSize", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS), false);
        nvrtcGetPTX = b.bind("nvrtcGetPTX", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS), false);
        nvrtcGetProgramLogSize = b.bind("nvrtcGetProgramLogSize", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS),
                false);
        nvrtcGetProgramLog = b.bind("nvrtcGetProgramLog", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS), false);
        nvrtcDestroyProgram = b.bind("nvrtcDestroyProgram", FunctionDescriptor.of(JAVA_INT, ADDRESS), false);
        nvrtcGetErrorString = b.bind("nvrtcGetErrorString", FunctionDescriptor.of(ADDRESS, JAVA_INT), false);
    }

    private void check(int result, String call) {
        if (result != NVRTC_SUCCESS) {
            throw new CudaException(call + ": " + errorString(result) + " (" + result + ")", result);
        }
    }

    private String errorString(int result) {
        try {
            MemorySegment s = (MemorySegment) nvrtcGetErrorString.invokeExact(result);
            return s.equals(MemorySegment.NULL) ? "NVRTC error" : s.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            return "NVRTC error";
        }
    }

    /** For example "12.4". */
    String version() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment major = a.allocate(JAVA_INT);
            MemorySegment minor = a.allocate(JAVA_INT);
            check((int) nvrtcVersion.invokeExact(major, minor), "nvrtcVersion");
            return major.get(JAVA_INT, 0) + "." + minor.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("nvrtcVersion", t);
        }
    }

    /**
     * The newest virtual architecture this NVRTC can compile for that the device runs, such as
     * 89 for compute_89. PTX for an older architecture runs on newer devices.
     *
     * @param device the device's compute capability as major * 10 + minor
     */
    int architecture(int device) {
        if (nvrtcGetNumSupportedArchs == null || nvrtcGetSupportedArchs == null) {
            return Math.min(device, 80);   // NVRTC before 11.2: sm_80 is the newest it may know
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment count = a.allocate(JAVA_INT);
            check((int) nvrtcGetNumSupportedArchs.invokeExact(count), "nvrtcGetNumSupportedArchs");
            int n = count.get(JAVA_INT, 0);
            MemorySegment archs = a.allocate(JAVA_INT, Math.max(1, n));
            check((int) nvrtcGetSupportedArchs.invokeExact(archs), "nvrtcGetSupportedArchs");
            int best = -1;
            int lowest = Integer.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                int arch = archs.getAtIndex(JAVA_INT, i);
                lowest = Math.min(lowest, arch);
                if (arch <= device) {
                    best = Math.max(best, arch);
                }
            }
            if (best < 0) {
                throw new CudaException("NVRTC " + version() + " cannot compile for compute capability "
                        + device / 10 + "." + device % 10 + " (oldest it supports: " + lowest / 10 + "." + lowest % 10
                        + ")", 0);
            }
            return best;
        } catch (Throwable t) {
            throw Binder.fail("nvrtcGetSupportedArchs", t);
        }
    }

    /**
     * Compiles CUDA C++ to PTX.
     *
     * @param source the program
     * @param name its name in messages
     * @param options compiler options, such as {@code --gpu-architecture=compute_89}
     * @return the PTX and the compiler's messages
     * @throws CudaException with the compiler's messages if compiling failed
     */
    Ptx compile(String source, String name, List<String> options) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment program = a.allocate(JAVA_LONG);
            check((int) nvrtcCreateProgram.invokeExact(program, a.allocateFrom(source), a.allocateFrom(name), 0,
                    MemorySegment.NULL, MemorySegment.NULL), "nvrtcCreateProgram");
            long prog = program.get(JAVA_LONG, 0);
            try {
                MemorySegment opts = a.allocate(ADDRESS, Math.max(1, options.size()));
                for (int i = 0; i < options.size(); i++) {
                    opts.setAtIndex(ADDRESS, i, a.allocateFrom(options.get(i)));
                }
                int result = (int) nvrtcCompileProgram.invokeExact(prog, options.size(), opts);
                String log = log(a, prog);
                if (result != NVRTC_SUCCESS) {
                    throw new CudaException("compiling " + name + " with NVRTC: " + errorString(result) + "\n" + log,
                            result);
                }
                MemorySegment size = a.allocate(JAVA_LONG);
                check((int) nvrtcGetPTXSize.invokeExact(prog, size), "nvrtcGetPTXSize");
                MemorySegment ptx = a.allocate(size.get(JAVA_LONG, 0));
                check((int) nvrtcGetPTX.invokeExact(prog, ptx), "nvrtcGetPTX");
                return new Ptx(ptx.toArray(JAVA_BYTE), log);
            } finally {
                int destroyed = (int) nvrtcDestroyProgram.invokeExact(program);
                if (destroyed != NVRTC_SUCCESS) {
                    System.err.println("brainiac cuda: nvrtcDestroyProgram: " + errorString(destroyed));
                }
            }
        } catch (Throwable t) {
            throw Binder.fail("NVRTC", t);
        }
    }

    private String log(Arena a, long prog) throws Throwable {
        MemorySegment size = a.allocate(JAVA_LONG);
        check((int) nvrtcGetProgramLogSize.invokeExact(prog, size), "nvrtcGetProgramLogSize");
        long n = size.get(JAVA_LONG, 0);
        if (n <= 1) {
            return "";
        }
        MemorySegment log = a.allocate(n);
        check((int) nvrtcGetProgramLog.invokeExact(prog, log), "nvrtcGetProgramLog");
        return log.getString(0).strip();
    }
}
