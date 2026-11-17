package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

/**
 * The common {@link LogitsProcessor}s of modern decoders, as static factories.
 *
 * <p><b>Penalties</b> push down tokens the text already has:
 * <ul>
 * <li>{@link #repetition}: the CTRL penalty (Keskar et al. 2019,
 *     <a href="https://arxiv.org/abs/1909.05858">arXiv:1909.05858</a>), dividing positive
 *     and multiplying negative logits of seen tokens;</li>
 * <li>{@link #frequencyPresence}: OpenAI's additive penalties, {@code count * frequency +
 *     presence} for each seen token;</li>
 * <li>{@link #noRepeatNgram}: bans any token that would repeat an n-gram;</li>
 * <li>{@link #dry}: "Don't Repeat Yourself" (p-e-w, 2024, text-generation-webui / llama.cpp),
 *     which penalizes the token that would extend a repeat of an earlier passage, more the
 *     longer the repeat: {@code multiplier * base^(length - allowedLength)}, with sequence
 *     breakers that reset a match.</li>
 * </ul>
 *
 * <p><b>Truncation</b> masks unlikely tokens before a draw:
 * <ul>
 * <li>{@link #topK}, {@link #topP} (Holtzman et al. 2019,
 *     <a href="https://arxiv.org/abs/1904.09751">arXiv:1904.09751</a>) and {@link #minP}
 *     (Nguyen et al. 2024, <a href="https://arxiv.org/abs/2407.01082">arXiv:2407.01082</a>);</li>
 * <li>{@link #topA}: keeps tokens with {@code p >= a * pmax^2} (BlinkDL, 2021);</li>
 * <li>{@link #typical}: locally typical sampling (Meister et al. 2022,
 *     <a href="https://arxiv.org/abs/2202.00666">arXiv:2202.00666</a>), keeping the tokens
 *     whose surprise is closest to the entropy until they hold {@code mass};</li>
 * <li>{@link #epsilon} and {@link #eta}: truncation sampling (Hewitt et al. 2022,
 *     <a href="https://arxiv.org/abs/2210.15191">arXiv:2210.15191</a>), an absolute floor
 *     and an entropy-dependent floor {@code min(eta, sqrt(eta) * exp(-H))}.</li>
 * </ul>
 *
 * <p><b>Temperature</b>: {@link #temperature}, and {@link #entropyTemperature}, the
 * entropy-based dynamic temperature (EDT, Zhang et al. 2024,
 * <a href="https://arxiv.org/abs/2403.14541">arXiv:2403.14541</a>), {@code T = T0 *
 * N^(theta / H)}: sharp when the model is sure, flatter when it is not.
 *
 * <p>Plus {@link #logitBias} and {@link #minLength}.
 */
public final class LogitsProcessors {

    private LogitsProcessors() {
    }

    /**
     * @param t the temperature, positive
     * @return a processor dividing the logits by {@code t}
     */
    public static LogitsProcessor temperature(float t) {
        if (!(t > 0)) {
            throw new IllegalArgumentException("temperature must be positive; use greedy sampling for 0");
        }
        return (c, l) -> {
            for (int i = 0; i < l.length; i++) {
                l[i] /= t;
            }
        };
    }

    /**
     * Entropy-based dynamic temperature: {@code T = t0 * base^(theta / H)} with {@code H} the
     * entropy of the logits in nats.
     *
     * @param t0 the temperature at very high entropy
     * @param base below 1, for example 0.8
     * @param theta the strength, for example 1
     * @return the processor
     */
    public static LogitsProcessor entropyTemperature(float t0, float base, float theta) {
        return (c, l) -> {
            double h = Math.max(1e-3, LogitMath.entropy(LogitMath.softmax(l)));
            double t = Math.max(1e-3, t0 * Math.pow(base, theta / h));
            for (int i = 0; i < l.length; i++) {
                l[i] = (float) (l[i] / t);
            }
        };
    }

    /**
     * @param penalty above 1 to discourage repeats
     * @return the CTRL repetition penalty
     */
    public static LogitsProcessor repetition(float penalty) {
        return (c, l) -> {
            boolean[] done = new boolean[l.length];
            for (int t : c) {
                if (t >= 0 && t < l.length && !done[t]) {
                    done[t] = true;
                    l[t] = l[t] > 0 ? l[t] / penalty : l[t] * penalty;
                }
            }
        };
    }

    /**
     * @param frequency subtracted once per earlier occurrence
     * @param presence subtracted once if the token occurred at all
     * @return the additive penalties
     */
    public static LogitsProcessor frequencyPresence(float frequency, float presence) {
        return (c, l) -> {
            int[] count = new int[l.length];
            for (int t : c) {
                if (t >= 0 && t < l.length) {
                    count[t]++;
                }
            }
            for (int i = 0; i < l.length; i++) {
                if (count[i] > 0) {
                    l[i] -= count[i] * frequency + presence;
                }
            }
        };
    }

    /**
     * @param n the n-gram length, at least 1
     * @return a processor banning any token that completes an n-gram already in the context
     */
    public static LogitsProcessor noRepeatNgram(int n) {
        return (c, l) -> {
            if (c.length < n - 1) {
                return;
            }
            int start = c.length - (n - 1);
            for (int i = 0; i + n - 1 < c.length; i++) {
                boolean match = true;
                for (int k = 0; k < n - 1 && match; k++) {
                    match = c[i + k] == c[start + k];
                }
                if (match) {
                    int banned = c[i + n - 1];
                    if (banned >= 0 && banned < l.length) {
                        l[banned] = LogitMath.MASKED;
                    }
                }
            }
        };
    }

    /**
     * The DRY penalty.
     *
     * @param multiplier the penalty at a repeat of exactly {@code allowedLength}, for example 0.8
     * @param base the growth per extra repeated token, for example 1.75
     * @param allowedLength repeats up to this long are free, for example 2
     * @param breakers tokens that end a match (newlines, quotes)
     * @return the processor
     */
    public static LogitsProcessor dry(float multiplier, float base, int allowedLength, Set<Integer> breakers) {
        return (c, l) -> {
            int n = c.length;
            if (n < 1) {
                return;
            }
            int[] longest = new int[l.length];
            Arrays.fill(longest, -1);
            int last = c[n - 1];
            if (breakers.contains(last)) {
                return;
            }
            for (int i = 0; i < n - 1; i++) {
                if (c[i] != last) {
                    continue;
                }
                int next = c[i + 1];
                if (breakers.contains(next) || next < 0 || next >= l.length) {
                    continue;
                }
                int len = 0;
                while (len <= i && n - 1 - len > i && c[i - len] == c[n - 1 - len]
                        && !breakers.contains(c[i - len])) {
                    len++;
                }
                longest[next] = Math.max(longest[next], len);
            }
            for (int t = 0; t < l.length; t++) {
                if (longest[t] >= allowedLength && l[t] != LogitMath.MASKED) {
                    l[t] -= (float) (multiplier * Math.pow(base, longest[t] - allowedLength));
                }
            }
        };
    }

    /**
     * @param bias added to the logit of each token
     * @return the processor
     */
    public static LogitsProcessor logitBias(Map<Integer, Float> bias) {
        return (c, l) -> bias.forEach((t, b) -> l[t] += b);
    }

    /**
     * @param eos the end-of-sequence token
     * @param length the context length before which {@code eos} is banned
     * @return the processor
     */
    public static LogitsProcessor minLength(int eos, int length) {
        return (c, l) -> {
            if (c.length < length) {
                l[eos] = LogitMath.MASKED;
            }
        };
    }

    // ---------------------------------------------------------------- truncation

    /**
     * @param k the tokens to keep
     * @return top-k truncation
     */
    public static LogitsProcessor topK(int k) {
        return (c, l) -> {
            Integer[] order = order(l);
            for (int i = k; i < order.length; i++) {
                l[order[i]] = LogitMath.MASKED;
            }
        };
    }

    /**
     * @param p the mass to keep
     * @return nucleus truncation
     */
    public static LogitsProcessor topP(double p) {
        return (c, l) -> {
            double[] prob = LogitMath.softmax(l);
            Integer[] order = order(l);
            double cum = 0;
            for (int i = 0; i < order.length; i++) {
                if (cum >= p) {
                    l[order[i]] = LogitMath.MASKED;
                }
                cum += prob[order[i]];
            }
        };
    }

    /**
     * @param p the share of the top probability a token needs
     * @return min-p truncation
     */
    public static LogitsProcessor minP(double p) {
        return floor((prob, max, h) -> p * max);
    }

    /**
     * @param a the strength, for example 0.2
     * @return top-a truncation, threshold {@code a * pmax^2}
     */
    public static LogitsProcessor topA(double a) {
        return floor((prob, max, h) -> a * max * max);
    }

    /**
     * @param eps the probability floor, for example 3e-4
     * @return epsilon truncation
     */
    public static LogitsProcessor epsilon(double eps) {
        return floor((prob, max, h) -> eps);
    }

    /**
     * @param eta the strength, for example 3e-4
     * @return eta truncation, floor {@code min(eta, sqrt(eta) * exp(-H))}
     */
    public static LogitsProcessor eta(double eta) {
        return floor((prob, max, h) -> Math.min(eta, Math.sqrt(eta) * Math.exp(-h)));
    }

    /**
     * @param mass the probability mass to keep, for example 0.9
     * @return locally typical truncation
     */
    public static LogitsProcessor typical(double mass) {
        return (c, l) -> {
            double[] logp = LogitMath.logSoftmax(l);
            double[] p = LogitMath.softmax(l);
            double h = LogitMath.entropy(p);
            Integer[] order = new Integer[l.length];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(dev(logp[a], h), dev(logp[b], h)));
            double cum = 0;
            for (Integer t : order) {
                if (cum >= mass) {
                    l[t] = LogitMath.MASKED;
                }
                cum += p[t];
            }
        };
    }

    private static double dev(double logp, double h) {
        return logp == Double.NEGATIVE_INFINITY ? Double.POSITIVE_INFINITY : Math.abs(-logp - h);
    }

    private interface Floor {
        double of(double[] prob, double max, double entropy);
    }

    /** Masks tokens below a probability floor, always keeping the most likely. */
    private static LogitsProcessor floor(Floor floor) {
        return (c, l) -> {
            double[] p = LogitMath.softmax(l);
            int best = LogitMath.argmax(l);
            double f = floor.of(p, p[best], LogitMath.entropy(p));
            for (int i = 0; i < l.length; i++) {
                if (i != best && p[i] < f) {
                    l[i] = LogitMath.MASKED;
                }
            }
        };
    }

    private static Integer[] order(float[] l) {
        Integer[] order = new Integer[l.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Float.compare(l[b], l[a]));
        return order;
    }
}
