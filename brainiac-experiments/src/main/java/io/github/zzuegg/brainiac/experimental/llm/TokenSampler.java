package io.github.zzuegg.brainiac.experimental.llm;

import java.util.random.RandomGenerator;

/**
 * Picks the next token from processed logits. Samplers may keep state across steps (as
 * {@link Mirostat} does); {@link #reset()} clears it for a new sequence.
 */
@FunctionalInterface
public interface TokenSampler {

    /**
     * Picks a token.
     *
     * @param context every token so far
     * @param logits the processed logits; masked tokens are never picked
     * @param random the source of randomness
     * @return the token
     */
    int sample(int[] context, float[] logits, RandomGenerator random);

    /** Clears any state, for a new sequence. */
    default void reset() {
    }

    /** @return the most likely token */
    static TokenSampler greedy() {
        return (c, l, r) -> LogitMath.argmax(l);
    }

    /** @return a draw from the softmax of the logits (temperature belongs in a processor) */
    static TokenSampler ancestral() {
        return (c, l, r) -> LogitMath.draw(LogitMath.softmax(l), r);
    }
}
