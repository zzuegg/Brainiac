package io.github.zzuegg.brainiac.backend.cuda;

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
import java.util.List;
import java.util.regex.Pattern;

/**
 * Bindings to cuBLAS, NVIDIA's matrix library, for the matrix products: it picks tensor-core
 * kernels tuned for each GPU. Optional: without it the backend multiplies with its own
 * kernel. {@code -Dbrainiac.cuda.cublas} names the file, {@code =none} turns it off.
 *
 * <p>cuBLAS is column-major. A row-major {@code C = A B} is computed as the column-major
 * {@code C^T = B^T A^T}, which reads the same memory with no transposes.
 */
@SuppressWarnings("restricted")
final class Cublas {

    static final int OP_N = 0;
    static final int OP_T = 1;

    private static final int CUDA_R_32F = 0;
    private static final int CUDA_R_64F = 1;
    private static final int CUDA_R_16F = 2;
    private static final int CUDA_R_16BF = 14;
    private static final int CUBLAS_COMPUTE_32F = 68;
    private static final int CUBLAS_GEMM_DEFAULT = -1;
    private static final int CUBLAS_DEFAULT_MATH = 0;
    private static final int CUBLAS_TF32_TENSOR_OP_MATH = 3;

    final String library;
    private final long handle;

    private final MethodHandle cublasSgemm;
    private final MethodHandle cublasDgemm;
    private final MethodHandle cublasGemmEx;
    private final MethodHandle cublasSgemmStridedBatched;
    private final MethodHandle cublasDgemmStridedBatched;
    private final MethodHandle cublasGemmStridedBatchedEx;
    private final int version;

    // alpha and beta, read by cuBLAS from host memory
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
     * Loads cuBLAS and creates a handle that queues its work on {@code stream}. The device's
     * context must be current.
     *
     * @param stream the stream every product runs on
     * @param tf32 whether float32 products may use TensorFloat-32 (faster, 10-bit mantissa inputs)
     */
    Cublas(long stream, boolean tf32) {
        Binder b = Binder.load("cuBLAS", "brainiac.cuda.cublas",
                Binder.WINDOWS
                        ? List.of("cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll")
                        : List.of("libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"),
                Binder.WINDOWS ? Pattern.compile("cublas64_(\\d+)\\.dll") : Pattern.compile("libcublas\\.so\\.([0-9.]+)"),
                name -> name.replaceFirst("cublas", "cublasLt"));
        library = b.library();
        MethodHandle create = b.bind("cublasCreate_v2", FunctionDescriptor.of(JAVA_INT, ADDRESS), false);
        MethodHandle setStream = b.bind("cublasSetStream_v2", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG),
                false);
        MethodHandle setMathMode = b.bind("cublasSetMathMode", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT),
                false);
        MethodHandle getVersion = b.bind("cublasGetVersion_v2", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS),
                false);
        // handle, transa, transb, m, n, k
        MemoryLayout[] head = {JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT};
        cublasSgemm = b.bind("cublasSgemm_v2", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT)), true);
        cublasDgemm = b.bind("cublasDgemm_v2", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT)), true);
        cublasGemmEx = b.bind("cublasGemmEx", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_INT, JAVA_INT, JAVA_INT)), true);
        cublasSgemmStridedBatched = b.bind("cublasSgemmStridedBatched", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_LONG, JAVA_INT)), true);
        cublasDgemmStridedBatched = b.bind("cublasDgemmStridedBatched", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_LONG, JAVA_INT)), true);
        cublasGemmStridedBatchedEx = b.bind("cublasGemmStridedBatchedEx", FunctionDescriptor.of(JAVA_INT, concat(head,
                ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, ADDRESS,
                JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT)), true);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(JAVA_LONG);
            check((int) create.invokeExact(h), "cublasCreate");
            handle = h.get(JAVA_LONG, 0);
            check((int) setStream.invokeExact(handle, stream), "cublasSetStream");
            check((int) setMathMode.invokeExact(handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH),
                    "cublasSetMathMode");
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) getVersion.invokeExact(handle, v), "cublasGetVersion");
            version = v.get(JAVA_INT, 0);
        } catch (Throwable t) {
            throw Binder.fail("cublasCreate", t);
        }
    }

    private static MemoryLayout[] concat(MemoryLayout[] head, MemoryLayout... tail) {
        MemoryLayout[] all = java.util.Arrays.copyOf(head, head.length + tail.length);
        System.arraycopy(tail, 0, all, head.length, tail.length);
        return all;
    }

    /** For example "12.4.5". */
    String version() {
        return version / 10000 + "." + version / 100 % 100 + "." + version % 100;
    }

    static String statusName(int status) {
        return switch (status) {
            case 0 -> "CUBLAS_STATUS_SUCCESS";
            case 1 -> "CUBLAS_STATUS_NOT_INITIALIZED";
            case 3 -> "CUBLAS_STATUS_ALLOC_FAILED";
            case 7 -> "CUBLAS_STATUS_INVALID_VALUE";
            case 8 -> "CUBLAS_STATUS_ARCH_MISMATCH";
            case 11 -> "CUBLAS_STATUS_MAPPING_ERROR";
            case 13 -> "CUBLAS_STATUS_EXECUTION_FAILED";
            case 14 -> "CUBLAS_STATUS_INTERNAL_ERROR";
            case 15 -> "CUBLAS_STATUS_NOT_SUPPORTED";
            case 16 -> "CUBLAS_STATUS_LICENSE_ERROR";
            default -> "cuBLAS status " + status;
        };
    }

    private static void check(int status, String call) {
        if (status != 0) {
            throw new CudaException(call + ": " + statusName(status) + " (" + status + ")", status);
        }
    }

    private static int type(DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> CUDA_R_32F;
            case FLOAT64 -> CUDA_R_64F;
            case FLOAT16 -> CUDA_R_16F;
            case BFLOAT16 -> CUDA_R_16BF;
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
                        call = "cublasSgemm";
                        status = (int) cublasSgemm.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, ldb, a, lda, ZERO_F,
                                c, n);
                    }
                    case FLOAT64 -> {
                        call = "cublasDgemm";
                        status = (int) cublasDgemm.invokeExact(handle, opB, opA, n, m, k, ONE_D, b, ldb, a, lda, ZERO_D,
                                c, n);
                    }
                    default -> {
                        call = "cublasGemmEx";
                        int t = type(dtype);
                        status = (int) cublasGemmEx.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, t, ldb, a, t,
                                lda, ZERO_F, c, t, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                    }
                }
            } else {
                switch (dtype) {
                    case FLOAT32 -> {
                        call = "cublasSgemmStridedBatched";
                        status = (int) cublasSgemmStridedBatched.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, ldb,
                                strideB, a, lda, strideA, ZERO_F, c, n, strideC, batch);
                    }
                    case FLOAT64 -> {
                        call = "cublasDgemmStridedBatched";
                        status = (int) cublasDgemmStridedBatched.invokeExact(handle, opB, opA, n, m, k, ONE_D, b, ldb,
                                strideB, a, lda, strideA, ZERO_D, c, n, strideC, batch);
                    }
                    default -> {
                        call = "cublasGemmStridedBatchedEx";
                        int t = type(dtype);
                        status = (int) cublasGemmStridedBatchedEx.invokeExact(handle, opB, opA, n, m, k, ONE_F, b, t,
                                ldb, strideB, a, t, lda, strideA, ZERO_F, c, t, n, strideC, batch, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT);
                    }
                }
            }
        } catch (Throwable t) {
            throw Binder.fail("cuBLAS gemm", t);
        }
        if (status != 0) {
            throw new CudaException(call + " " + dtype + " (" + batch + " x [" + m + ", " + k + "] x [" + k + ", " + n
                    + "]): " + statusName(status) + " (" + status + ")", status);
        }
    }
}
