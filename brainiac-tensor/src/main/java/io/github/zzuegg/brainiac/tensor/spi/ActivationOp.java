package io.github.zzuegg.brainiac.tensor.spi;

/**
 * Activation functions a backend computes in one pass, forward and backward, instead of
 * as a chain of simpler element-wise operations. Each takes one optional parameter
 * {@code p}; operations without a parameter ignore it.
 */
public enum ActivationOp {
    /** {@code max(0, x)} */
    RELU(false),
    /** {@code 1 / (1 + e^-x)} */
    SIGMOID(true),
    /** hyperbolic tangent */
    TANH(true),
    /** {@code log(1 + e^x)} */
    SOFTPLUS(true),
    /** {@code x * sigmoid(x)} */
    SILU(true),
    /** {@code x * Phi(x)} with the exact normal CDF */
    GELU(true),
    /** {@code x * Phi(x)} with the tanh approximation of the CDF */
    GELU_TANH(true),
    /** {@code x * tanh(softplus(x))} */
    MISH(true),
    /** {@code x} if positive, else {@code p * x} */
    LEAKY_RELU(false),
    /** {@code x} if positive, else {@code p * (e^x - 1)} */
    ELU(true),
    /** {@code 1.0507 * elu(x)} with {@code alpha = 1.6733} */
    SELU(true),
    /** {@code min(max(0, x), 6)} */
    RELU6(false),
    /** {@code min(max(-1, x), 1)} */
    HARD_TANH(false),
    /** {@code min(max(0, x / 6 + 1/2), 1)} */
    HARD_SIGMOID(false),
    /** {@code x * hardSigmoid(x)} */
    HARD_SWISH(false),
    /** {@code x / (1 + |x|)} */
    SOFTSIGN(false);

    private final boolean transcendental;

    ActivationOp(boolean transcendental) {
        this.transcendental = transcendental;
    }

    /**
     * @return whether the function needs exp, log, tanh or erf; the others use only
     *     exactly rounded arithmetic, so SIMD versions give bit-identical results
     */
    public boolean transcendental() {
        return transcendental;
    }
}
