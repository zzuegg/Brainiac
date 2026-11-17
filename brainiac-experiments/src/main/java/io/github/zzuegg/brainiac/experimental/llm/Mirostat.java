package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Mirostat (Basu et al. 2021, <a href="https://arxiv.org/abs/2007.14966">arXiv:2007.14966</a>):
 * a sampler that steers the <em>surprise</em> {@code -log2 p} of the picked tokens towards a
 * target {@code tau} with a feedback loop, so the text stays neither boring (repetition
 * traps of low surprise) nor incoherent (high surprise), whatever the model's confidence.
 *
 * <ul>
 * <li>{@link #v2}: keep the tokens whose surprise is below {@code mu}, draw, then
 *     {@code mu -= eta * (surprise - tau)}. Starts at {@code mu = 2 tau}.</li>
 * <li>{@link #v1}: estimate the Zipf exponent {@code s} from the top {@code m} tokens and
 *     choose top-k from it: {@code k = ((eps 2^mu) / (1 - N^-eps))^(1/s)},
 *     {@code eps = s - 1}.</li>
 * </ul>
 * Use it after penalties and temperature but without other truncation.
 */
public final class Mirostat implements TokenSampler {

    private final boolean v1;
    private final double tau;
    private final double eta;
    private final int m;
    private double mu;
    private double lastSurprise;

    private Mirostat(boolean v1, double tau, double eta, int m) {
        if (!(tau > 0) || !(eta > 0)) {
            throw new IllegalArgumentException("tau and eta must be positive");
        }
        this.v1 = v1;
        this.tau = tau;
        this.eta = eta;
        this.m = m;
        this.mu = 2 * tau;
    }

    /**
     * @param tau the target surprise in bits, for example 5
     * @param eta the learning rate, for example 0.1
     * @return Mirostat 2.0
     */
    public static Mirostat v2(double tau, double eta) {
        return new Mirostat(false, tau, eta, 0);
    }

    /**
     * @param tau the target surprise in bits
     * @param eta the learning rate
     * @param m the tokens used to estimate the Zipf exponent, for example 100
     * @return Mirostat 1.0
     */
    public static Mirostat v1(double tau, double eta, int m) {
        return new Mirostat(true, tau, eta, Math.max(2, m));
    }

    /** @return the current surprise threshold {@code mu} */
    public double mu() {
        return mu;
    }

    /** @return the surprise of the last token picked, in bits */
    public double lastSurprise() {
        return lastSurprise;
    }

    @Override
    public void reset() {
        mu = 2 * tau;
    }

    @Override
    public int sample(int[] context, float[] logits, RandomGenerator random) {
        double[] p = LogitMath.softmax(logits);
        Integer[] order = new Integer[p.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(p[b], p[a]));
        double[] w = new double[p.length];
        if (v1) {
            int k = topK(p, order);
            for (int i = 0; i < k; i++) {
                w[order[i]] = p[order[i]];
            }
        } else {
            for (int i = 0; i < order.length; i++) {
                int t = order[i];
                if (i == 0 || -log2(p[t]) <= mu) {
                    w[t] = p[t];
                }
            }
        }
        int token = LogitMath.draw(w, random);
        lastSurprise = -log2(p[token]);
        mu -= eta * (lastSurprise - tau);
        return token;
    }

    private int topK(double[] p, Integer[] order) {
        int n = Math.min(m, p.length);
        double num = 0;
        double den = 0;
        for (int i = 0; i < n - 1; i++) {
            double pi = p[order[i]];
            double pj = p[order[i + 1]];
            if (pi <= 0 || pj <= 0) {
                break;
            }
            double ti = Math.log((i + 2.0) / (i + 1.0));
            double bi = Math.log(pi / pj);
            num += ti * bi;
            den += ti * ti;
        }
        double s = den > 0 ? num / den : 1.0;
        double eps = Math.max(1e-6, s - 1);
        double k = Math.pow(eps * Math.pow(2, mu) / (1 - Math.pow(p.length, -eps)), 1 / s);
        return (int) Math.max(1, Math.min(p.length, Math.round(k)));
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    @Override
    public String toString() {
        return "Mirostat(" + (v1 ? "v1" : "v2") + ", tau " + tau + ", eta " + eta + ")";
    }
}
