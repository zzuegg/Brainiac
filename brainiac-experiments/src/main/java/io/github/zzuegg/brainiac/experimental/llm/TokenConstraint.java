package io.github.zzuegg.brainiac.experimental.llm;

/**
 * Structured decoding: which tokens may come next, as the state of a grammar that the
 * output so far has reached. A decoder masks every other token before it samples
 * ({@link #mask}) and moves the state with each token it picks.
 */
public interface TokenConstraint {

    /** @return the state before any token */
    int start();

    /**
     * @param state the current state
     * @param token a candidate token
     * @return the state after it, or a negative number if it is not allowed
     */
    int advance(int state, int token);

    /**
     * @param state the current state
     * @return whether the output may end here
     */
    boolean complete(int state);

    /**
     * Masks the logits of every token not allowed in {@code state}.
     *
     * @param state the current state
     * @param logits a row of logits; changed
     */
    default void mask(int state, float[] logits) {
        for (int t = 0; t < logits.length; t++) {
            if (advance(state, t) < 0) {
                logits[t] = LogitMath.MASKED;
            }
        }
    }
}
