package io.github.zzuegg.brainiac.experimental.llm;

import java.util.HashSet;
import java.util.Set;
import java.util.random.RandomGenerator;

/**
 * The SynthID-Text tournament watermark (Dathathri et al., Nature 2024,
 * <a href="https://doi.org/10.1038/s41586-024-08025-4">doi:10.1038/s41586-024-08025-4</a>).
 * Each step draws {@code 2^m} candidate tokens from the unchanged LM distribution and runs an
 * {@code m}-layer knockout tournament: in layer {@code l} each pair keeps the candidate with
 * the higher pseudorandom bit {@code g_l(context, token)} (ties at random), keyed by a hash of
 * the secret key, the previous {@code H} tokens and the layer. The winner is emitted, so tokens
 * with {@code g = 1} are over-represented; detection averages the g-values of a text, which is
 * about 0.5 without the watermark.
 *
 * <p>With binary g-values one tournament layer turns the distribution {@code p} into
 * {@code p(x) (2 - G)} for {@code g(x) = 1} and {@code p(x) (1 - G)} otherwise, where
 * {@code G} is the probability mass with {@code g = 1}; {@link #process} applies the {@code m}
 * layers in closed form, so the watermark is a {@link LogitsProcessor} after which any sampler
 * draws exactly the tournament winner's distribution ({@link #sampleTournament} runs the
 * explicit tournament). Averaged over keys the output distribution equals {@code p}: the
 * watermark is non-distortionary per token in expectation, unlike the green list
 * ({@link GreenListWatermark}), which adds a bias.
 *
 * <p>Detection: {@link #meanScore} (mean g over tokens and layers), {@link #weightedMeanScore}
 * (layer weights decreasing linearly from 10 to 1, as in the paper), both skipping tokens whose
 * context window already occurred in the text (repeated-context masking), and {@link #zScore}.
 */
public final class TournamentWatermark implements LogitsProcessor {

    private final long key;
    private final int layers;
    private final int context;

    /**
     * Creates the watermark.
     *
     * @param key the secret key
     * @param layers the tournament depth {@code m}, such as 30 in the paper; small values suit small vocabularies
     * @param context the number of previous tokens {@code H} hashed into the g-values, such as 4
     */
    public TournamentWatermark(long key, int layers, int context) {
        if (layers < 1 || context < 1) {
            throw new IllegalArgumentException("layers and context must be positive");
        }
        this.key = key;
        this.layers = layers;
        this.context = context;
    }

    /** @return the tournament depth */
    public int layers() {
        return layers;
    }

    /**
     * The g-value of a candidate.
     *
     * @param tokens the text so far
     * @param end the index after the last context token (the candidate's position)
     * @param token the candidate
     * @param layer the tournament layer, from 0
     * @return 0 or 1
     */
    public int g(int[] tokens, int end, int token, int layer) {
        long h = mix(key + 0x632BE59BD9B4E019L * (layer + 1));
        for (int i = Math.max(0, end - context); i < end; i++) {
            h = mix(h ^ (tokens[i] + 0x9E3779B97F4A7C15L));
        }
        h = mix(h ^ (token * 0xD6E8FEB86659FD93L + 1));
        return (int) (h >>> 63);
    }

    @Override
    public void process(int[] tokens, float[] logits) {
        double[] p = LogitMath.softmax(logits);
        int end = tokens.length;
        for (int l = 0; l < layers; l++) {
            double big = 0;
            int[] gv = new int[p.length];
            for (int t = 0; t < p.length; t++) {
                if (p[t] > 0) {
                    gv[t] = g(tokens, end, t, l);
                    big += gv[t] * p[t];
                }
            }
            double total = 0;
            for (int t = 0; t < p.length; t++) {
                p[t] *= gv[t] == 1 ? 2 - big : 1 - big;
                total += p[t];
            }
            for (int t = 0; t < p.length; t++) {
                p[t] /= total;
            }
        }
        for (int t = 0; t < logits.length; t++) {
            logits[t] = p[t] > 0 ? (float) Math.log(p[t]) : LogitMath.MASKED;
        }
    }

    /**
     * Runs the explicit tournament: {@code 2^m} candidates drawn from {@code p}, knocked out
     * pairwise layer by layer.
     *
     * @param tokens the text so far
     * @param p the LM's next-token probabilities
     * @param random the sampling source
     * @return the winner
     */
    public int sampleTournament(int[] tokens, double[] p, RandomGenerator random) {
        int[] cand = new int[1 << layers];
        for (int i = 0; i < cand.length; i++) {
            cand[i] = draw(p, random);
        }
        int end = tokens.length;
        for (int l = 0; l < layers; l++) {
            int[] next = new int[cand.length / 2];
            for (int i = 0; i < next.length; i++) {
                int a = cand[2 * i];
                int b = cand[2 * i + 1];
                int ga = g(tokens, end, a, l);
                int gb = g(tokens, end, b, l);
                next[i] = ga > gb ? a : gb > ga ? b : random.nextBoolean() ? a : b;
            }
            cand = next;
        }
        return cand[0];
    }

    /**
     * The mean g-value of a text, over tokens after the prompt and all layers.
     *
     * @param tokens prompt and continuation
     * @param promptLength tokens to skip
     * @return about 0.5 without the watermark, higher with it
     */
    public double meanScore(int[] tokens, int promptLength) {
        return score(tokens, promptLength, false)[0];
    }

    /**
     * As {@link #meanScore} with layer weights {@code 10 .. 1}.
     *
     * @param tokens prompt and continuation
     * @param promptLength tokens to skip
     * @return the weighted mean g-value
     */
    public double weightedMeanScore(int[] tokens, int promptLength) {
        return score(tokens, promptLength, true)[0];
    }

    /**
     * The z-score of the mean g-value against the unwatermarked mean 0.5.
     *
     * @param tokens prompt and continuation
     * @param promptLength tokens to skip
     * @return {@code (mean - 0.5) / sqrt(0.25 / (tokens * layers))}
     */
    public double zScore(int[] tokens, int promptLength) {
        double[] s = score(tokens, promptLength, false);
        return s[1] == 0 ? 0 : (s[0] - 0.5) / Math.sqrt(0.25 / (s[1] * layers));
    }

    /** {mean, scored token count}. */
    private double[] score(int[] tokens, int promptLength, boolean weighted) {
        Set<String> seen = new HashSet<>();
        double sum = 0;
        double weight = 0;
        int count = 0;
        for (int t = Math.max(promptLength, 1); t < tokens.length; t++) {
            StringBuilder sb = new StringBuilder();
            for (int i = Math.max(0, t - context); i <= t; i++) {
                sb.append(tokens[i]).append(',');
            }
            if (!seen.add(sb.toString())) {
                continue;   // repeated context and token: scored once
            }
            count++;
            for (int l = 0; l < layers; l++) {
                double w = weighted ? 10 - 9.0 * l / Math.max(1, layers - 1) : 1;
                sum += w * g(tokens, t, tokens[t], l);
                weight += w;
            }
        }
        return new double[] {weight == 0 ? 0.5 : sum / weight, count};
    }

    private static int draw(double[] p, RandomGenerator random) {
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
