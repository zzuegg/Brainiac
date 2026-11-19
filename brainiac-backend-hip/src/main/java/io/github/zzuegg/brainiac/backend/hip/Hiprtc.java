package io.github.zzuegg.brainiac.backend.hip;

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
 * Bindings to hipRTC, HIP's runtime compiler, which turns the backend's HIP C++ kernels into a
 * code object for the GPU at hand. It comes with AMD's HIP SDK (ROCm on Linux);
 * {@code -Dbrainiac.hip.hiprtc} names the file.
 */
@SuppressWarnings("restricted")
final class Hiprtc {

    private static final int HIPRTC_SUCCESS = 0;

    final String library;

    private final MethodHandle hiprtcVersion;
    private final MethodHandle hiprtcCreateProgram;
    private final MethodHandle hiprtcCompileProgram;
    private final MethodHandle hiprtcGetCodeSize;
    private final MethodHandle hiprtcGetCode;
    private final MethodHandle hiprtcGetProgramLogSize;
    private final MethodHandle hiprtcGetProgramLog;
    private final MethodHandle hiprtcDestroyProgram;
    private final MethodHandle hiprtcGetErrorString;

    /** The result of a compilation: the code object and the compiler's messages. */
    record Code(byte[] code, String log) {
    }

    Hiprtc() {
        Binder b = Binder.load("hipRTC (the HIP runtime compiler)", "brainiac.hip.hiprtc",
                Binder.WINDOWS
                        ? List.of("hiprtc0702.dll", "hiprtc0701.dll", "hiprtc0700.dll", "hiprtc0604.dll",
                                "hiprtc0603.dll", "hiprtc0602.dll", "hiprtc0601.dll", "hiprtc.dll")
                        : List.of("libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so"),
                Binder.WINDOWS ? Pattern.compile("hiprtc(\\d+)\\.dll") : Pattern.compile("libhiprtc\\.so\\.([0-9.]+)"),
                name -> name.replaceFirst("hiprtc", "hiprtc-builtins"));
        library = b.library();
        hiprtcVersion = b.bind("hiprtcVersion", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS), false);
        hiprtcCreateProgram = b.bind("hiprtcCreateProgram",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS), false);
        hiprtcCompileProgram = b.bind("hiprtcCompileProgram",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS), false);
        hiprtcGetCodeSize = b.bind("hiprtcGetCodeSize", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS), false);
        hiprtcGetCode = b.bind("hiprtcGetCode", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS), false);
        hiprtcGetProgramLogSize = b.bind("hiprtcGetProgramLogSize",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS), false);
        hiprtcGetProgramLog = b.bind("hiprtcGetProgramLog", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS),
                false);
        hiprtcDestroyProgram = b.bind("hiprtcDestroyProgram", FunctionDescriptor.of(JAVA_INT, ADDRESS), false);
        hiprtcGetErrorString = b.bind("hiprtcGetErrorString", FunctionDescriptor.of(ADDRESS, JAVA_INT), false);
    }

    private void check(int result, String call) {
        if (result != HIPRTC_SUCCESS) {
            throw new HipException(call + ": " + errorString(result) + " (" + result + ")", result);
        }
    }

    private String errorString(int result) {
        try {
            MemorySegment s = (MemorySegment) hiprtcGetErrorString.invokeExact(result);
            return s.equals(MemorySegment.NULL) ? "hipRTC error" : s.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            return "hipRTC error";
        }
    }

    /** For example "7.2". */
    String version() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment major = a.allocate(JAVA_INT);
            MemorySegment minor = a.allocate(JAVA_INT);
            check((int) hiprtcVersion.invokeExact(major, minor), "hiprtcVersion");
            return major.get(JAVA_INT, 0) + "." + minor.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("hiprtcVersion", t);
        }
    }

    /**
     * Compiles HIP C++ to a code object.
     *
     * @param source the program
     * @param name its name in messages
     * @param options compiler options, such as {@code --gpu-architecture=gfx1201}
     * @return the code object and the compiler's messages
     * @throws HipException with the compiler's messages if compiling failed
     */
    Code compile(String source, String name, List<String> options) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment program = a.allocate(JAVA_LONG);
            check((int) hiprtcCreateProgram.invokeExact(program, a.allocateFrom(source), a.allocateFrom(name), 0,
                    MemorySegment.NULL, MemorySegment.NULL), "hiprtcCreateProgram");
            long prog = program.get(JAVA_LONG, 0);
            try {
                MemorySegment opts = a.allocate(ADDRESS, Math.max(1, options.size()));
                for (int i = 0; i < options.size(); i++) {
                    opts.setAtIndex(ADDRESS, i, a.allocateFrom(options.get(i)));
                }
                int result = (int) hiprtcCompileProgram.invokeExact(prog, options.size(), opts);
                String log = log(a, prog);
                if (result != HIPRTC_SUCCESS) {
                    throw new HipException("compiling " + name + " with hipRTC: " + errorString(result) + "\n" + log,
                            result);
                }
                MemorySegment size = a.allocate(JAVA_LONG);
                check((int) hiprtcGetCodeSize.invokeExact(prog, size), "hiprtcGetCodeSize");
                MemorySegment code = a.allocate(Math.max(1, size.get(JAVA_LONG, 0)));
                check((int) hiprtcGetCode.invokeExact(prog, code), "hiprtcGetCode");
                return new Code(code.asSlice(0, size.get(JAVA_LONG, 0)).toArray(JAVA_BYTE), log);
            } finally {
                int destroyed = (int) hiprtcDestroyProgram.invokeExact(program);
                if (destroyed != HIPRTC_SUCCESS) {
                    System.err.println("brainiac hip: hiprtcDestroyProgram: " + errorString(destroyed));
                }
            }
        } catch (Throwable t) {
            throw Binder.fail("hipRTC", t);
        }
    }

    private String log(Arena a, long prog) throws Throwable {
        MemorySegment size = a.allocate(JAVA_LONG);
        check((int) hiprtcGetProgramLogSize.invokeExact(prog, size), "hiprtcGetProgramLogSize");
        long n = size.get(JAVA_LONG, 0);
        if (n <= 1) {
            return "";
        }
        MemorySegment log = a.allocate(n);
        check((int) hiprtcGetProgramLog.invokeExact(prog, log), "hiprtcGetProgramLog");
        return log.getString(0).strip();
    }
}
