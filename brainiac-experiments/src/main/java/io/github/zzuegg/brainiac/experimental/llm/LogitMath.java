package io.github.zzuegg.brainiac.experimental.llm;

import java.util.random.RandomGenerator;

/**
 * Small helpers on one row of logits as decoders need them: softmax, log-softmax, entropy,
 * argmax and drawing from weights. Tokens cut off by a processor carry a logit of
 * {@link #MASKED} (negative infinity), which every helper treats as probability 0.
 */
public final class LogitMath {

    /** The logit of a token that may not be picked. */
    public static final float MASKED = Float.NEGATIVE_INFINITY;

    private LogitMath() {
    }

    /**
     * @param logits a row
     * @return the index of the largest value (the first on ties)
     */
    public static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * @param logits a row
     * @return the softmax, in double precision
     */
    public static double[] softmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = logits[i] == MASKED ? 0 : Math.exp(logits[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= sum;
        }
        return p;
    }

    /**
     * @param logits a row
     * @return the log-softmax (negative infinity for masked tokens)
     */
    public static double[] logSoftmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double sum = 0;
        for (float l : logits) {
            if (l != MASKED) {
                sum += Math.exp(l - max);
            }
        }
        double lse = max + Math.log(sum);
        double[] out = new double[logits.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = logits[i] == MASKED ? Double.NEGATIVE_INFINITY : logits[i] - lse;
        }
        return out;
    }

    /**
     * @param p probabilities
     * @return the entropy in nats
     */
    public static double entropy(double[] p) {
        double h = 0;
        for (double q : p) {
            if (q > 0) {
                h -= q * Math.log(q);
            }
        }
        return h;
    }

    /**
     * Draws an index with probability proportional to its weight.
     *
     * @param weights non-negative weights, not all zero
     * @param random the source of randomness
     * @return the index
     */
    public static int draw(double[] weights, RandomGenerator random) {
        double total = 0;
        for (double w : weights) {
            total += w;
        }
        double u = random.nextDouble() * total;
        int last = 0;
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] > 0) {
                last = i;
                u -= weights[i];
                if (u < 0) {
                    return i;
                }
            }
        }
        return last;
    }

    /**
     * Masks (sets to {@link #MASKED}) every token whose probability is zero in {@code keep}.
     *
     * @param logits the row; changed
     * @param keep which tokens survive
     */
    public static void keepOnly(float[] logits, boolean[] keep) {
        for (int i = 0; i < logits.length; i++) {
            if (!keep[i]) {
                logits[i] = MASKED;
            }
        }
    }
}
