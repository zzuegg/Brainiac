package io.github.zzuegg.brainiac.experimental.llm;

import java.util.EnumMap;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * Entropy-adaptive sampling in the style of entropix (xjdr, 2024): each step reads how sure
 * the model is from two numbers of its next-token distribution, the entropy {@code H} (how
 * spread the mass is) and the varentropy {@code V} (the variance of the surprisal
 * {@code -log p}, how uneven the spread is), and picks a strategy:
 * <ul>
 * <li>low {@code H}, low {@code V}: the model is sure, take the argmax ({@link Decision#ARGMAX});</li>
 * <li>high {@code H}, low {@code V}: evenly unsure between several options. With a
 *     {@linkplain #thinkToken think token} the sampler emits it to buy a reasoning step
 *     (once in a row); otherwise it samples among the plausible options
 *     ({@link Decision#THINK} / {@link Decision#EXPLORE});</li>
 * <li>low {@code H}, high {@code V}: one favourite with a few distinct alternatives: branch,
 *     sampling at the higher temperature among the tokens min-p keeps
 *     ({@link Decision#EXPLORE});</li>
 * <li>high {@code H}, high {@code V}: lost, sample at the base temperature from the min-p
 *     nucleus ({@link Decision#RESAMPLE}).</li>
 * </ul>
 * The effect is greedy accuracy where the model knows the answer and diversity only where it
 * genuinely does not. {@link #decisions()} counts what was chosen.
 */
public final class EntropyAdaptiveSampler implements TokenSampler {

    /** What a step did. */
    public enum Decision {
        /** took the argmax */
        ARGMAX,
        /** emitted the think token */
        THINK,
        /** sampled among the plausible options at the explore temperature */
        EXPLORE,
        /** sampled at the base temperature */
        RESAMPLE
    }

    private final double lowEntropy;
    private final double highVarentropy;
    private final double highEntropy;
    private final float temperature;
    private final float exploreTemperature;
    private final double minP;
    private int thinkToken = -1;
    private boolean thought;
    private final Map<Decision, Integer> counts = new EnumMap<>(Decision.class);

    /**
     * @param lowEntropy below this entropy (nats) a step counts as sure
     * @param highEntropy above this entropy a step counts as lost
     * @param highVarentropy above this varentropy the spread counts as uneven
     * @param temperature the base sampling temperature
     * @param exploreTemperature the temperature when exploring
     * @param minP the min-p truncation used whenever the sampler draws
     */
    public EntropyAdaptiveSampler(double lowEntropy, double highEntropy, double highVarentropy, float temperature,
            float exploreTemperature, double minP) {
        if (lowEntropy < 0 || highEntropy < lowEntropy || temperature <= 0 || exploreTemperature <= 0) {
            throw new IllegalArgumentException("bad thresholds");
        }
        this.lowEntropy = lowEntropy;
        this.highEntropy = highEntropy;
        this.highVarentropy = highVarentropy;
        this.temperature = temperature;
        this.exploreTemperature = exploreTemperature;
        this.minP = minP;
    }

    /**
     * @return entropix-like defaults for small vocabularies: sure below 0.1 nats, lost above
     *     2.5, uneven above varentropy 1.5, temperatures 0.7 and 1.2, min-p 0.1
     */
    public static EntropyAdaptiveSampler defaults() {
        return new EntropyAdaptiveSampler(0.1, 2.5, 1.5, 0.7f, 1.2f, 0.1);
    }

    /**
     * @param token a token the model reads as "think more" (a pause or "Wait"), or -1 for none
     * @return this sampler
     */
    public EntropyAdaptiveSampler thinkToken(int token) {
        this.thinkToken = token;
        return this;
    }

    /** @return how often each decision was taken since the sampler was made */
    public Map<Decision, Integer> decisions() {
        return Map.copyOf(counts);
    }

    /**
     * The entropy and varentropy of a row of logits.
     *
     * @param logits the logits
     * @return {@code {entropy, varentropy}} in nats
     */
    public static double[] entropyAndVarentropy(float[] logits) {
        double[] p = LogitMath.softmax(logits);
        double[] logp = LogitMath.logSoftmax(logits);
        double h = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                h -= p[i] * logp[i];
            }
        }
        double v = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                double d = -logp[i] - h;
                v += p[i] * d * d;
            }
        }
        return new double[] {h, v};
    }

    @Override
    public int sample(int[] context, float[] logits, RandomGenerator random) {
        double[] hv = entropyAndVarentropy(logits);
        double h = hv[0];
        double v = hv[1];
        Decision d;
        int token;
        if (h < lowEntropy && v < highVarentropy) {
            d = Decision.ARGMAX;
            token = LogitMath.argmax(logits);
        } else if (h >= lowEntropy && h <= highEntropy && v < highVarentropy && thinkToken >= 0 && !thought) {
            d = Decision.THINK;
            token = thinkToken;
        } else if (h <= highEntropy) {
            d = Decision.EXPLORE;
            token = draw(logits, exploreTemperature, random);
        } else {
            d = Decision.RESAMPLE;
            token = draw(logits, temperature, random);
        }
        thought = d == Decision.THINK;
        counts.merge(d, 1, Integer::sum);
        return token;
    }

    private int draw(float[] logits, float t, RandomGenerator random) {
        float[] l = logits.clone();
        LogitsProcessors.minP(minP).process(new int[0], l);
        for (int i = 0; i < l.length; i++) {
            l[i] = l[i] == LogitMath.MASKED ? l[i] : l[i] / t;
        }
        return LogitMath.draw(LogitMath.softmax(l), random);
    }

    @Override
    public void reset() {
        thought = false;
    }
}
