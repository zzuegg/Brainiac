package io.github.zzuegg.brainiac.backend.hip;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.zzuegg.brainiac.tensor.DType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Bindings to rocBLAS, AMD's matrix library, for the matrix products: it picks kernels tuned
 * for each GPU. Optional: without it the backend multiplies with its own kernel.
 * {@code -Dbrainiac.hip.rocblas} names the file, {@code =none} turns it off.
 *
 * <p>rocBLAS is column-major. A row-major {@code C = A B} is computed as the column-major
 * {@code C^T = B^T A^T}, which reads the same memory with no transposes.
 */
@SuppressWarnings("restricted")
final class Rocblas {

    static final int OP_N = 111;
    static final int OP_T = 112;

    private static final int R_16F = 150;
    private static final int R_32F = 151;
    private static final int R_64F = 152;
    private static final int R_16BF = 168;
    private static final int GEMM_ALGO_STANDARD = 0;

    final String library;
    private final long handle;

    private final MethodHandle sgemm;
    private final MethodHandle dgemm;
    private final MethodHandle gemmEx;
    private final MethodHandle sgemmStridedBatched;
    private final MethodHandle dgemmStridedBatched;
    private final MethodHandle gemmStridedBatchedEx;
    private final String version;
    private final MethodHandle destroyHandle;

    // alpha and beta, read by rocBLAS from host memory
    private static final MemorySegment ONE_F;
    private static final MemorySegment ZERO_F;
    private static final MemorySegment ONE_D;
    private static final MemorySegment ZERO_D;

    static {
        Arena a = Arena.global();
        ONE_F = a.allocateFrom(JAVA_FLOAT, 1f);
        ZERO_F = a.allocateFrom(JAVA_FLOAT, 0f);
        ONE_D = a.allocateFrom(JAVA_DOUBLE, 1d);
        ZERO_D = a.allocateFrom(JAVA_DOUBLE, 0d);
    }

    /**
     * Loads rocBLAS and creates a handle that queues its work on {@code stream}. The device must
     * be current.
     *
     * @param stream the stream every product runs on
     * @param architecture the GPU's target, such as {@code gfx1201}: rocBLAS stops the process
     *        when it has no kernels for it, so that is checked first
     */
    Rocblas(long stream, String architecture) {
        Binder b = Binder.load("rocBLAS", "brainiac.hip.rocblas",
                Binder.WINDOWS ? List.of("rocblas.dll") : List.of("librocblas.so.5", "librocblas.so.4", "librocblas.so"),
                Binder.WINDOWS ? Pattern.compile("rocblas(\\d*)\\.dll") : Pattern.compile("librocblas\\.so\\.([0-9.]+)"),
                null);
        library = b.library();
        checkKernelsFor(Path.of(library), architecture);
        MethodHandle create = b.bind("rocblas_create_handle", FunctionDescriptor.of(JAVA_INT, ADDRESS), false);
        destroyHandle = b.bind("rocblas_destroy_handle", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        MethodHandle setStream = b.bind("rocblas_set_stream", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG),
                false);
        MethodHandle versionSize = b.bind("rocblas_get_version_string_size", FunctionDescriptor.of(JAVA_INT, ADDRESS),
                false);
        MethodHandle versionString = b.bind("rocblas_get_version_string",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG), false);
        // handle, transa, transb, m, n, k
        MemoryLayout[] head = {JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT};
        // alpha, A, lda, B, ldb, beta, C, ldc
        sgemm = b.bind("rocblas_sgemm", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT)), true);
        dgemm = b.bind("rocblas_dgemm", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT)), true);
        // alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, compute, algo, solution, flags
        gemmEx = b.bind("rocblas_gemm_ex", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT)), true);
        // alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch
        sgemmStridedBatched = b.bind("rocblas_sgemm_strided_batched", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_LONG, JAVA_INT)), true);
        dgemmStridedBatched = b.bind("rocblas_dgemm_strided_batched", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_LONG, JAVA_INT)), true);
        // alpha, a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c,
        // d, d_type, ldd, stride_d, batch, compute, algo, solution, flags
        gemmStridedBatchedEx = b.bind("rocblas_gemm_strided_batched_ex", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, ADDRESS,
                JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT,
                JAVA_INT, JAVA_INT, JAVA_INT)), true);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(JAVA_LONG);
            check((int) create.invokeExact(h), "rocblas_create_handle");
            handle = h.get(JAVA_LONG, 0);
            check((int) setStream.invokeExact(handle, stream), "rocblas_set_stream");
            MemorySegment n = a.allocate(JAVA_LONG);
            check((int) versionSize.invokeExact(n), "rocblas_get_version_string_size");
            long len = Math.max(1, n.get(JAVA_LONG, 0));
            MemorySegment v = a.allocate(len);
            check((int) versionString.invokeExact(v, len), "rocblas_get_version_string");
            version = v.getString(0);
        } catch (Throwable t) {
            throw Binder.fail("rocblas_create_handle", t);
        }
    }

    /**
     * rocBLAS reads its kernels from {@code rocblas/library} beside its own file and aborts the
     * process when none are for the GPU; refuses to use it then. Passes when the folder cannot
     * be found, since rocBLAS may then know better ({@code ROCBLAS_TENSILE_LIBPATH}).
     */
    static void checkKernelsFor(Path library, String architecture) {
        String arch = architecture.split(":")[0].toLowerCase(Locale.ROOT);
        String env = System.getenv("ROCBLAS_TENSILE_LIBPATH");
        Path dir = env != null && !env.isBlank() ? Path.of(env.strip())
                : library.getParent() == null ? null : library.getParent().resolve("rocblas").resolve("library");
        if (dir == null || !Files.isDirectory(dir)) {
            return;
        }
        try (Stream<Path> files = Files.list(dir)) {
            if (files.noneMatch(f -> f.getFileName() != null
                    && f.getFileName().toString().toLowerCase(Locale.ROOT).contains(arch))) {
                throw new HipException("rocBLAS in " + library.getParent() + " has no kernels for " + arch
                        + " (nothing for it in " + dir + ")", 0);
            }
        } catch (java.io.IOException | SecurityException e) {
            // let rocBLAS decide
        }
    }

    private static MemoryLayout[] concat(MemoryLayout[] head, MemoryLayout... tail) {
        MemoryLayout[] all = java.util.Arrays.copyOf(head, head.length + tail.length);
        System.arraycopy(tail, 0, all, head.length, tail.length);
        return all;
    }

    /** Releases the handle, at exit; returns the status instead of throwing. */
    int destroy() {
        try {
            return (int) destroyHandle.invokeExact(handle);
        } catch (Throwable t) {
            throw Binder.fail("rocblas_destroy_handle", t);
        }
    }

    /** For example "4.4.0.f21b8e3e". */
    String version() {
        return version;
    }

    static String statusName(int status) {
        return switch (status) {
            case 0 -> "rocblas_status_success";
            case 1 -> "rocblas_status_invalid_handle";
            case 2 -> "rocblas_status_not_implemented";
            case 3 -> "rocblas_status_invalid_pointer";
            case 4 -> "rocblas_status_invalid_size";
            case 5 -> "rocblas_status_memory_error";
            case 6 -> "rocblas_status_internal_error";
            case 7 -> "rocblas_status_perf_degraded";
            case 11 -> "rocblas_status_invalid_value";
            case 13 -> "rocblas_status_arch_mismatch";
            default -> "rocBLAS status " + status;
        };
    }

    private static void check(int status, String call) {
        if (status != 0) {
            throw new HipException(call + ": " + statusName(status) + " (" + status + ")", status);
        }
    }

    private static int type(DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> R_32F;
            case FLOAT64 -> R_64F;
            case FLOAT16 -> R_16F;
            case BFLOAT16 -> R_16BF;
            default -> throw new IllegalArgumentException("rocBLAS has no " + dtype + " products");
        };
    }

    /**
     * Row-major products {@code c[i] = op(a[i]) op(b[i])} for {@code i < batch}, with
     * {@code op(a)} {@code [m, k]} and {@code op(b)} {@code [k, n]}, all of one type. A
     * transposed operand is stored the other way round: {@code a} as {@code [k, m]} when
     * {@code transA}.
     *
     * @param dtype the element type
     * @param a the left matrices
     * @param lda the left matrices' row length in memory
     * @param strideA elements from one left matrix to the next (0 repeats it)
     * @param b the right matrices
     * @param ldb the right matrices' row length in memory
     * @param strideB elements from one right matrix to the next
     * @param c the outputs, contiguous {@code [m, n]} each
     * @param strideC elements from one output to the next
     */
    void gemm(DType dtype, boolean transA, boolean transB, int m, int n, int k, long a, int lda, long strideA, long b,
            int ldb, long strideB, long c, long strideC, int batch) {
        // column-major C^T [n, m] = op(B)^T op(A)^T: B first
        int opB = transB ? OP_T : OP_N;
        int opA = transA ? OP_T : OP_N;
        int status;
        String call;
        try {
            if (batch == 1) {
                switch (dtype) {
                    case FLOAT32 -> {
                        call = "rocblas_sgemm";
                        status = (int) sgemm.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, ldb, a, lda, ZERO_F,
                                c, n);
                    }
                    case FLOAT64 -> {
                        call = "rocblas_dgemm";
                        status = (int) dgemm.invokeExact(handle, opB, opA, n, m, k, ONE_D, b, ldb, a, lda, ZERO_D,
                                c, n);
                    }
                    default -> {
                        call = "rocblas_gemm_ex";
                        int t = type(dtype);
                        status = (int) gemmEx.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, t, ldb, a, t, lda,
                                ZERO_F, c, t, n, c, t, n, R_32F, GEMM_ALGO_STANDARD, 0, 0);
                    }
                }
            } else {
                switch (dtype) {
                    case FLOAT32 -> {
                        call = "rocblas_sgemm_strided_batched";
                        status = (int) sgemmStridedBatched.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, ldb,
                                strideB, a, lda, strideA, ZERO_F, c, n, strideC, batch);
                    }
                    case FLOAT64 -> {
                        call = "rocblas_dgemm_strided_batched";
                        status = (int) dgemmStridedBatched.invokeExact(handle, opB, opA, n, m, k, ONE_D, b, ldb,
                                strideB, a, lda, strideA, ZERO_D, c, n, strideC, batch);
                    }
                    default -> {
                        call = "rocblas_gemm_strided_batched_ex";
                        int t = type(dtype);
                        status = (int) gemmStridedBatchedEx.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, t, ldb,
                                strideB, a, t, lda, strideA, ZERO_F, c, t, n, strideC, c, t, n, strideC, batch, R_32F,
                                GEMM_ALGO_STANDARD, 0, 0);
                    }
                }
            }
        } catch (Throwable t) {
            throw Binder.fail("rocBLAS gemm", t);
        }
        if (status != 0) {
            throw new HipException(call + " " + dtype + " (" + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n
                    + "]): " + statusName(status) + " (" + status + ")", status);
        }
    }
}
