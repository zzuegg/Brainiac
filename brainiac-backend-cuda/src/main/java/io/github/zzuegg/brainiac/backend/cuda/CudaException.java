package io.github.zzuegg.brainiac.backend.cuda;

/**
 * A failed call into the CUDA driver, NVRTC or cuBLAS. The message names the call, the error
 * (such as {@code CUDA_ERROR_OUT_OF_MEMORY}) and what the backend was doing.
 */
public final class CudaException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    private final int code;

    /**
     * Creates the exception.
     *
     * @param message what failed
     * @param code the library's error code, 0 when there is none
     */
    public CudaException(String message, int code) {
        super(message);
        this.code = code;
    }

    /**
     * Creates the exception for a failure without an error code.
     *
     * @param message what failed
     * @param cause the underlying failure
     */
    public CudaException(String message, Throwable cause) {
        super(message, cause);
        this.code = 0;
    }

    /**
     * The library's error code, such as 2 for {@code CUDA_ERROR_OUT_OF_MEMORY}.
     *
     * @return the code, 0 when there is none
     */
    public int code() {
        return code;
    }
}
