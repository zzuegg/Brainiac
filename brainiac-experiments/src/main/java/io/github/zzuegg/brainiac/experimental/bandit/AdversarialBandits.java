package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.experimental.data.BanditMixer;

import java.util.ArrayDeque;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Randomised bandit policies that stay safe when an adversary picks the rewards, and the
 * best-of-both-worlds Tsallis-INF:
 *
 * <ul>
 *   <li>{@link Method#EXP3} (Auer et al., SIAM J. Computing 2002): exponential weights on
 *       importance-weighted reward estimates, mixed with {@code gamma} uniform exploration;</li>
 *   <li>{@link Method#EXP3_IX} (Neu, NeurIPS 2015): losses estimated as
 *       {@code loss / (p + gamma)}, an implicit-exploration bias that gives high-probability
 *       regret bounds without explicit mixing;</li>
 *   <li>{@link Method#TSALLIS_INF} (Zimmert &amp; Seldin, JMLR 2021,
 *       <a href="https://arxiv.org/abs/1807.07623">arXiv:1807.07623</a>): online mirror descent
 *       with the 1/2-Tsallis entropy, {@code p_i = 4 (eta_t (L_i - x))^-2} with
 *       {@code eta_t = 2 / sqrt(t)} and the normaliser {@code x} found by Newton's method, using
 *       the reduced-variance loss estimator with baseline 1/2; optimal in both the adversarial
 *       and the stochastic regime;</li>
 *   <li>{@link Method#SLIDING_WINDOW_UCB} (Garivier &amp; Moulines, 2011): UCB1 on the last
 *       {@code window} pulls, the non-stationary baseline.</li>
 * </ul>
 *
 * <p>Rewards are assumed in [0, 1]; losses are {@code 1 - reward}. {@link #banditMixer} wraps
 * the existing EXP3 of {@link BanditMixer} (as used for data mixing) as a policy.
 */
public final class AdversarialBandits implements BanditPolicy {

    /** The policy. */
    public enum Method {
        /** exponential weights for exploration and exploitation */
        EXP3,
        /** EXP3 with implicit exploration */
        EXP3_IX,
        /** Tsallis-INF with the reduced-variance estimator */
        TSALLIS_INF,
        /** UCB1 on a sliding window */
        SLIDING_WINDOW_UCB
    }

    private final Method method;
    private final int arms;
    private final int horizon;
    private final SplittableRandom random;
    private final double[] cumulative;
    private double[] p;
    private final ArrayDeque<double[]> window = new ArrayDeque<>();
    private int windowSize = 1_000;
    private double x = Double.NaN;
    private long t;

    private AdversarialBandits(Method method, int arms, int horizon, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.arms = arms;
        this.horizon = horizon;
        this.random = new SplittableRandom(seed);
        this.cumulative = new double[arms];
        this.p = uniform(arms);
    }

    private static double[] uniform(int k) {
        double[] u = new double[k];
        java.util.Arrays.fill(u, 1.0 / k);
        return u;
    }

    /**
     * @param method the policy
     * @param arms the arms
     * @param horizon the horizon, which sets EXP3's and EXP3-IX's rates
     * @param seed the seed
     * @return the policy
     */
    public static AdversarialBandits of(Method method, int arms, int horizon, long seed) {
        return new AdversarialBandits(method, arms, horizon, seed);
    }

    /**
     * @param pulls the sliding window length of {@link Method#SLIDING_WINDOW_UCB}
     * @return this policy
     */
    public AdversarialBandits window(int pulls) {
        this.windowSize = pulls;
        return this;
    }

    /**
     * The EXP3 of {@link BanditMixer} as a policy: it samples arms from its weights and is
     * updated with the reward of the pulled arm in place of a loss.
     *
     * @param arms the arms
     * @param seed the seed
     * @return the policy
     */
    public static BanditPolicy banditMixer(int arms, long seed) {
        BanditMixer mixer = BanditMixer.of(arms, 0);
        SplittableRandom random = new SplittableRandom(seed);
        return new BanditPolicy() {
            @Override
            public int arms() {
                return arms;
            }

            @Override
            public int select() {
                return mixer.sample(random);
            }

            @Override
            public void update(int arm, double reward) {
                mixer.update(arm, reward);
            }

            @Override
            public String toString() {
                return "BanditMixer EXP3";
            }
        };
    }

    @Override
    public int arms() {
        return arms;
    }

    /** @return the current sampling distribution (a copy) */
    public double[] probabilities() {
        return p.clone();
    }

    @Override
    public int select() {
        t++;
        switch (method) {
            case EXP3 -> {
                double gamma = gamma();
                double eta = gamma / arms;
                double max = Double.NEGATIVE_INFINITY;
                for (double c : cumulative) {
                    max = Math.max(max, eta * c);
                }
                double[] w = new double[arms];
                double s = 0;
                for (int i = 0; i < arms; i++) {
                    w[i] = Math.exp(eta * cumulative[i] - max);
                    s += w[i];
                }
                for (int i = 0; i < arms; i++) {
                    p[i] = (1 - gamma) * w[i] / s + gamma / arms;
                }
            }
            case EXP3_IX -> {
                double eta = Math.sqrt(2 * Math.log(arms) / (arms * (double) horizon));
                // cumulative holds estimated losses here
                double min = Double.POSITIVE_INFINITY;
                for (double c : cumulative) {
                    min = Math.min(min, c);
                }
                double s = 0;
                for (int i = 0; i < arms; i++) {
                    p[i] = Math.exp(-eta * (cumulative[i] - min));
                    s += p[i];
                }
                for (int i = 0; i < arms; i++) {
                    p[i] /= s;
                }
            }
            case TSALLIS_INF -> tsallis();
            case SLIDING_WINDOW_UCB -> {
                return slidingUcb();
            }
            default -> throw new IllegalStateException();
        }
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < arms; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return arms - 1;
    }

    private double gamma() {
        return Math.min(1, Math.sqrt(arms * Math.log(arms) / ((Math.E - 1) * horizon)));
    }

    private void tsallis() {
        double eta = 2 / Math.sqrt(t);
        double min = Double.POSITIVE_INFINITY;
        for (double c : cumulative) {
            min = Math.min(min, c);
        }
        // x must stay below min L; start from the previous normaliser when valid
        double guess = Double.isNaN(x) || x >= min ? min - 2 / eta : x;
        for (int it = 0; it < 50; it++) {
            double s = 0;
            double ds = 0;
            for (double c : cumulative) {
                double w = 4 / Math.pow(eta * (c - guess), 2);
                s += w;
                ds += eta * Math.pow(w, 1.5);
            }
            double nextGuess = guess - (s - 1) / ds;
            if (nextGuess >= min) {
                nextGuess = (guess + min) / 2;
            }
            if (Math.abs(nextGuess - guess) < 1e-12) {
                guess = nextGuess;
                break;
            }
            guess = nextGuess;
        }
        x = guess;
        double s = 0;
        for (int i = 0; i < arms; i++) {
            p[i] = 4 / Math.pow(eta * (cumulative[i] - x), 2);
            s += p[i];
        }
        for (int i = 0; i < arms; i++) {
            p[i] /= s;
        }
    }

    private int slidingUcb() {
        int[] n = new int[arms];
        double[] sum = new double[arms];
        for (double[] e : window) {
            n[(int) e[0]]++;
            sum[(int) e[0]] += e[1];
        }
        for (int a = 0; a < arms; a++) {
            if (n[a] == 0) {
                return a;
            }
        }
        double log = Math.log(Math.min(t, windowSize));
        double[] index = new double[arms];
        for (int a = 0; a < arms; a++) {
            index[a] = sum[a] / n[a] + Math.sqrt(2 * log / n[a]);
        }
        return BanditMath.argmax(index, random);
    }

    @Override
    public void update(int arm, double reward) {
        double r = Math.min(1, Math.max(0, reward));
        switch (method) {
            case EXP3 -> cumulative[arm] += r / p[arm];
            case EXP3_IX -> cumulative[arm] += (1 - r) / (p[arm] + Math.sqrt(2 * Math.log(arms)
                    / (arms * (double) horizon)) / 2);
            case TSALLIS_INF -> {
                double eta = 2 / Math.sqrt(Math.max(t, 1));
                double loss = 1 - r;
                for (int i = 0; i < arms; i++) {
                    double b = p[i] >= eta * eta ? 0.5 : 0;
                    cumulative[i] += (i == arm ? (loss - b) / p[i] : 0) + b;
                }
            }
            case SLIDING_WINDOW_UCB -> {
                window.addLast(new double[] {arm, r});
                if (window.size() > windowSize) {
                    window.removeFirst();
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    @Override
    public String toString() {
        return method.toString();
    }
}
