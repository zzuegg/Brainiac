package io.github.zzuegg.brainiac.experimental.llm;

/**
 * The green-list watermark of Kirchenbauer et al. 2023, "A Watermark for Large Language
 * Models" (<a href="https://arxiv.org/abs/2301.10226">arXiv:2301.10226</a>). At each step a
 * secret key and the previous token pick a "green" share {@code gamma} of the vocabulary,
 * and {@code delta} is added to the green logits: a soft bias that barely changes the text
 * where the model is sure, but makes green tokens over-represented. Anyone with the key
 * detects it from the tokens alone with a one-proportion z-test,
 * {@code z = (green - gamma T) / sqrt(T gamma (1 - gamma))}, without the model.
 *
 * <p>Green membership is a keyed hash of {@code (previous, token)}, so detection needs no
 * vocabulary permutation and costs one hash per token.
 */
public final class GreenListWatermark implements LogitsProcessor {

    private final long key;
    private final double gamma;
    private final float delta;

    /**
     * @param key the secret key
     * @param gamma the green share of the vocabulary, for example 0.25
     * @param delta the bias added to green logits, for example 2
     */
    public GreenListWatermark(long key, double gamma, float delta) {
        if (!(gamma > 0 && gamma < 1)) {
            throw new IllegalArgumentException("gamma must be in (0, 1)");
        }
        this.key = key;
        this.gamma = gamma;
        this.delta = delta;
    }

    /**
     * @param previous the token before
     * @param token a candidate
     * @return whether the candidate is green after {@code previous}
     */
    public boolean green(int previous, int token) {
        long h = mix(key ^ mix(previous * 0x9E3779B97F4A7C15L + token));
        return (h >>> 11) * 0x1.0p-53 < gamma;
    }

    @Override
    public void process(int[] context, float[] logits) {
        if (context.length == 0) {
            return;
        }
        int prev = context[context.length - 1];
        for (int t = 0; t < logits.length; t++) {
            if (green(prev, t)) {
                logits[t] += delta;
            }
        }
    }

    /**
     * The detection statistic over the tokens of a text.
     *
     * @param tokens the text, the first token used only as context
     * @return the z-score; above 4 is strong evidence of the watermark
     */
    public double zScore(int[] tokens) {
        int n = tokens.length - 1;
        if (n < 1) {
            return 0;
        }
        int g = 0;
        for (int i = 1; i < tokens.length; i++) {
            if (green(tokens[i - 1], tokens[i])) {
                g++;
            }
        }
        return (g - gamma * n) / Math.sqrt(n * gamma * (1 - gamma));
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
