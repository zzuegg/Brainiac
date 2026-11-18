package io.github.zzuegg.brainiac.studio.chat;

/**
 * How a model picks its next word beyond the temperature. Each part is optional, and a provider
 * whose API has no such setting leaves it out.
 *
 * @param topP the model picks among the likeliest words that together reach this probability,
 *        above 0 up to 1, or null for the model's default
 * @param topK the model picks among this many likeliest words, from 1 to 1000, or null for the
 *        model's default
 * @param repeatPenalty how strongly words already written are held back, from 1 (not at all)
 *        to 2, or null for the model's default; only models on this PC and on worker hosts take it
 */
public record Sampler(Float topP, Integer topK, Float repeatPenalty) {

    /** Everything left to the model. */
    public static final Sampler NONE = new Sampler(null, null, null);

    /** Checks the ranges. */
    public Sampler {
        if (topP != null && !(topP > 0 && topP <= 1)) {
            throw new IllegalArgumentException("topP must be above 0 and at most 1");
        }
        if (topK != null && (topK < 1 || topK > 1000)) {
            throw new IllegalArgumentException("topK must be between 1 and 1000");
        }
        if (repeatPenalty != null && !(repeatPenalty >= 1 && repeatPenalty <= 2)) {
            throw new IllegalArgumentException("repeatPenalty must be between 1 and 2");
        }
    }

    /**
     * Returns whether nothing is set.
     *
     * @return whether everything is left to the model
     */
    public boolean none() {
        return topP == null && topK == null && repeatPenalty == null;
    }
}
