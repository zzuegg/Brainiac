package io.github.zzuegg.brainiac.tensor.spi;

/** Element-wise functions of two values. */
public enum BinaryOp {
    /** {@code a + b} */
    ADD,
    /** {@code a - b} */
    SUB,
    /** {@code a * b} */
    MUL,
    /** {@code a / b} */
    DIV,
    /** {@code a} raised to {@code b} */
    POW,
    /** the larger value */
    MAX,
    /** the smaller value */
    MIN,
    /** {@code 1} if {@code a > b}, else {@code 0} */
    GREATER,
    /** {@code 1} if {@code a == b}, else {@code 0} */
    EQUAL,
    /** {@code a} where {@code b} is non-zero, else {@code 0} (even if {@code a} is infinite or NaN) */
    MASK
}
