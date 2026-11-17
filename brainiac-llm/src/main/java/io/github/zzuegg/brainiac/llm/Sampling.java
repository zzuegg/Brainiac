package io.github.zzuegg.brainiac.llm;

import java.util.Arrays;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * How the next token is picked from the logits.
 *
 * <p>The steps run in the order transformers runs them: the logit bias, the repetition
 * penalty and OpenAI's presence and frequency penalties, then the temperature, then top-k,
 * top-p and min-p, then a draw from what is left. A temperature of 0 picks the most likely
 * token.
 *
 * @param temperature divides the logits; 0 for greedy decoding
 * @param topK keeps the {@code k} most likely tokens; 0 for all
 * @param topP keeps the fewest tokens whose probabilities add up to {@code p}; 1 for all
 * @param minP keeps the tokens at least {@code minP} times as likely as the most likely; 0
 *     for all
 * @param repetitionPenalty divides the positive logits (and multiplies the negative ones) of
 *     tokens already in the sequence; 1 for none
 * @param presencePenalty subtracted from the logit of each token already written (not the
 *     prompt's), as OpenAI's {@code presence_penalty}; 0 for none
 * @param frequencyPenalty subtracted from the logit of each token written, once per time it
 *     was written, as OpenAI's {@code frequency_penalty}; 0 for none
 * @param logitBias added to the logits of the tokens it names, as OpenAI's
 *     {@code logit_bias}; empty for none
 */
public record Sampling(float temperature, int topK, float topP, float minP, float repetitionPenalty,
        float presencePenalty, float frequencyPenalty, Map<Integer, Float> logitBias) {

    /** Always the most likely token. */
    public static final Sampling GREEDY = new Sampling(0, 0, 1, 0, 1);

    /**
     * Settings without presence or frequency penalties or a logit bias.
     *
     * @param temperature divides the logits
     * @param topK keeps the k most likely
     * @param topP keeps the smallest nucleus
     * @param minP keeps tokens likely enough
     * @param repetitionPenalty penalizes repeats
     */
    public Sampling(float temperature, int topK, float topP, float minP, float repetitionPenalty) {
        this(temperature, topK, topP, minP, repetitionPenalty, 0, 0, Map.of());
    }

    /**
     * Checks the settings.
     *
     * @param temperature divides the logits
     * @param topK keeps the k most likely
     * @param topP keeps the smallest nucleus
     * @param minP keeps tokens likely enough
     * @param repetitionPenalty penalizes repeats
     * @param presencePenalty penalizes tokens written
     * @param frequencyPenalty penalizes tokens by how often they were written
     * @param logitBias biases tokens
     */
    public Sampling {
        logitBias = logitBias == null ? Map.of() : Map.copyOf(logitBias);
        if (temperature < 0 || topK < 0 || topP <= 0 || topP > 1 || minP < 0 || minP > 1 || repetitionPenalty <= 0
                || !Float.isFinite(presencePenalty) || !Float.isFinite(frequencyPenalty)) {
            throw new IllegalArgumentException("bad sampling settings: " + this);
        }
    }

    /** @return whether picking a token depends on the tokens before it */
    public boolean usesHistory() {
        return repetitionPenalty != 1 || presencePenalty != 0 || frequencyPenalty != 0;
    }

    /** @return whether the logits are taken as they come (no bias or penalty) */
    public boolean plain() {
        return !usesHistory() && logitBias.isEmpty();
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
        return new Sampling(temperature, k, topP, minP, repetitionPenalty, presencePenalty, frequencyPenalty, logitBias);
    }

    /**
     * @param p the probability mass to keep
     * @return a copy with top-p
     */
    public Sampling withTopP(float p) {
        return new Sampling(temperature, topK, p, minP, repetitionPenalty, presencePenalty, frequencyPenalty, logitBias);
    }

    /**
     * @param p the share of the top probability a token needs
     * @return a copy with min-p
     */
    public Sampling withMinP(float p) {
        return new Sampling(temperature, topK, topP, p, repetitionPenalty, presencePenalty, frequencyPenalty, logitBias);
    }

    /**
     * @param penalty the repetition penalty
     * @return a copy with it
     */
    public Sampling withRepetitionPenalty(float penalty) {
        return new Sampling(temperature, topK, topP, minP, penalty, presencePenalty, frequencyPenalty, logitBias);
    }

    /**
     * @param presence the presence penalty
     * @param frequency the frequency penalty
     * @return a copy with them
     */
    public Sampling withPenalties(float presence, float frequency) {
        return new Sampling(temperature, topK, topP, minP, repetitionPenalty, presence, frequency, logitBias);
    }

    /**
     * @param bias added to the logits of the tokens it names
     * @return a copy with it
     */
    public Sampling withLogitBias(Map<Integer, Float> bias) {
        return new Sampling(temperature, topK, topP, minP, repetitionPenalty, presencePenalty, frequencyPenalty, bias);
    }

    /**
     * Applies the repetition penalty in place, counting every token of {@code seen} as
     * written for the presence and frequency penalties.
     *
     * @param logits one row of logits; changed
     * @param seen the tokens already in the sequence
     */
    public void penalize(float[] logits, int[] seen) {
        penalize(logits, seen, seen.length);
    }

    /**
     * Applies the logit bias and the penalties in place.
     *
     * @param logits one row of logits; changed
     * @param seen the tokens already in the sequence, the prompt's then the written ones
     * @param written how many of the last tokens of {@code seen} were written, for the
     *     presence and frequency penalties
     */
    public void penalize(float[] logits, int[] seen, int written) {
        for (Map.Entry<Integer, Float> e : logitBias.entrySet()) {
            int t = e.getKey();
            if (t >= 0 && t < logits.length) {
                logits[t] += e.getValue();
            }
        }
        if (repetitionPenalty != 1) {
            boolean[] done = new boolean[logits.length];
            for (int t : seen) {
                if (t >= 0 && t < logits.length && !done[t]) {
                    done[t] = true;
                    logits[t] = logits[t] > 0 ? logits[t] / repetitionPenalty : logits[t] * repetitionPenalty;
                }
            }
        }
        if (presencePenalty != 0 || frequencyPenalty != 0) {
            int[] counts = new int[logits.length];
            for (int i = Math.max(0, seen.length - written); i < seen.length; i++) {
                int t = seen[i];
                if (t >= 0 && t < logits.length) {
                    counts[t]++;
                }
            }
            for (int t = 0; t < logits.length; t++) {
                if (counts[t] > 0) {
                    logits[t] -= presencePenalty + frequencyPenalty * counts[t];
                }
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
     * @param seen the tokens already in the sequence, all counted as written
     * @param random the source of randomness
     * @return the token
     */
    public int sample(float[] logits, int[] seen, RandomGenerator random) {
        return sample(logits, seen, seen.length, random);
    }

    /**
     * Picks a token.
     *
     * @param logits one row of logits; not changed
     * @param seen the tokens already in the sequence, for the penalties
     * @param written how many of the last of {@code seen} were written
     * @param random the source of randomness
     * @return the token
     */
    public int sample(float[] logits, int[] seen, int written, RandomGenerator random) {
        float[] l = plain() ? logits : logits.clone();
        penalize(l, seen, written);
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
