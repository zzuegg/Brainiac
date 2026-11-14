package io.github.zzuegg.brainiac.tensor.spi;

/** Reductions along one axis. */
public enum ReduceOp {
    /** the sum */
    SUM,
    /** the arithmetic mean */
    MEAN,
    /** the largest value */
    MAX,
    /** the smallest value */
    MIN,
    /** the index of the largest value (first one on ties), as a float */
    ARGMAX
}
