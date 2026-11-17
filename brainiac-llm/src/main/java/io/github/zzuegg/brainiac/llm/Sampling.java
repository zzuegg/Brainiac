package io.github.zzuegg.brainiac.llm;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * How the next token is picked from the logits.
 *
 * <p>The steps run in the order transformers runs them: the repetition penalty, then the
 * temperature, then top-k, top-p and min-p, then a draw from what is left. A temperature of
 * 0 picks the most likely token.
 *
 * @param temperature divides the logits; 0 for greedy decoding
 * @param topK keeps the {@code k} most likely tokens; 0 for all
 * @param topP keeps the fewest tokens whose probabilities add up to {@code p}; 1 for all
 * @param minP keeps the tokens at least {@code minP} times as likely as the most likely; 0
 *     for all
 * @param repetitionPenalty divides the positive logits (and multiplies the negative ones) of
 *     tokens already in the sequence; 1 for none
 */
public record Sampling(float temperature, int topK, float topP, float minP, float repetitionPenalty) {

    /** Always the most likely token. */
    public static final Sampling GREEDY = new Sampling(0, 0, 1, 0, 1);

    /**
     * Checks the settings.
     *
     * @param temperature divides the logits
     * @param topK keeps the k most likely
     * @param topP keeps the smallest nucleus
     * @param minP keeps tokens likely enough
     * @param repetitionPenalty penalizes repeats
     */
    public Sampling {
        if (temperature < 0 || topK < 0 || topP <= 0 || topP > 1 || minP < 0 || minP > 1 || repetitionPenalty <= 0) {
            throw new IllegalArgumentException("bad sampling settings: " + this);
        }
    }

    /**
     * @param temperature divides the logits
     * @return sampling at that temperature with nothing cut off
     */
    public static Sampling temperature(float temperature) {
        return new Sampling(temperature, 0, 1, 0, 1);
    }

    /**
     * @param k the tokens to keep
     * @return a copy with top-k
     */
    public Sampling withTopK(int k) {
        return new Sampling(temperature, k, topP, minP, repetitionPenalty);
    }

    /**
     * @param p the probability mass to keep
     * @return a copy with top-p
     */
    public Sampling withTopP(float p) {
        return new Sampling(temperature, topK, p, minP, repetitionPenalty);
    }

    /**
     * @param p the share of the top probability a token needs
     * @return a copy with min-p
     */
    public Sampling withMinP(float p) {
        return new Sampling(temperature, topK, topP, p, repetitionPenalty);
    }

    /**
     * @param penalty the repetition penalty
     * @return a copy with it
     */
    public Sampling withRepetitionPenalty(float penalty) {
        return new Sampling(temperature, topK, topP, minP, penalty);
    }

    /**
     * Applies the repetition penalty in place.
     *
     * @param logits one row of logits; changed
     * @param seen the tokens already in the sequence
     */
    public void penalize(float[] logits, int[] seen) {
        if (repetitionPenalty == 1) {
            return;
        }
        boolean[] done = new boolean[logits.length];
        for (int t : seen) {
            if (t >= 0 && t < logits.length && !done[t]) {
                done[t] = true;
                logits[t] = logits[t] > 0 ? logits[t] / repetitionPenalty : logits[t] * repetitionPenalty;
            }
        }
    }

    /**
     * The probabilities left after temperature, top-k, top-p and min-p.
     *
     * @param logits one row of logits (after {@link #penalize}); not changed
     * @return probabilities adding up to 1, zero for the tokens cut off
     */
    public double[] probabilities(float[] logits) {
        int n = logits.length;
        double[] p = new double[n];
        if (temperature == 0) {
            p[argmax(logits)] = 1;
            return p;
        }
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp((logits[i] - max) / temperature);
            sum += p[i];
        }
        for (int i = 0; i < n; i++) {
            p[i] /= sum;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(p[b], p[a]));
        int keep = topK > 0 ? Math.min(topK, n) : n;
        if (topP < 1) {
            double top = 0;
            for (int i = 0; i < keep; i++) {
                top += p[order[i]];
            }
            double cum = 0;
            for (int i = 0; i < keep; i++) {
                cum += p[order[i]];
                if (cum >= topP * top) {
                    keep = i + 1;
                    break;
                }
            }
        }
        double floor = minP * p[order[0]];
        double kept = 0;
        for (int i = 0; i < n; i++) {
            int t = order[i];
            if (i >= keep || (i > 0 && p[t] < floor)) {
                p[t] = 0;
            } else {
                kept += p[t];
            }
        }
        for (int i = 0; i < n; i++) {
            p[i] /= kept;
        }
        return p;
    }

    /**
     * Picks a token.
     *
     * @param logits one row of logits; not changed
     * @param seen the tokens already in the sequence, for the repetition penalty
     * @param random the source of randomness
     * @return the token
     */
    public int sample(float[] logits, int[] seen, RandomGenerator random) {
        float[] l = repetitionPenalty == 1 ? logits : logits.clone();
        penalize(l, seen);
        if (temperature == 0) {
            return argmax(l);
        }
        double[] p = probabilities(l);
        double u = random.nextDouble();
        int last = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                last = i;
                u -= p[i];
                if (u < 0) {
                    return i;
                }
            }
        }
        return last;
    }

    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }
        return best;
    }
}
