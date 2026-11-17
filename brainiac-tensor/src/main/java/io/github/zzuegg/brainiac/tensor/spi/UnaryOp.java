package io.github.zzuegg.brainiac.tensor.spi;

/** Element-wise functions of one value. */
public enum UnaryOp {
    /** {@code -x} */
    NEG,
    /** {@code |x|} */
    ABS,
    /** {@code e^x} */
    EXP,
    /** natural logarithm */
    LOG,
    /** square root */
    SQRT,
    /** {@code x * x} */
    SQUARE,
    /** hyperbolic tangent */
    TANH,
    /** {@code 1 / (1 + e^-x)} */
    SIGMOID,
    /** {@code max(0, x)} */
    RELU,
    /** {@code 1} if {@code x > 0}, else {@code 0}; the derivative of ReLU */
    STEP,
    /** {@code -1}, {@code 0} or {@code 1} */
    SIGN,
    /** the Gauss error function */
    ERF,
    /** {@code log(1 + e^x)}, computed without overflow */
    SOFTPLUS,
    /** the nearest whole number, ties to even */
    ROUND
}
