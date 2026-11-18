package io.github.zzuegg.brainiac.experimental.bandit;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Index and sampling policies for stochastic multi-armed bandits, the asymptotically or
 * minimax optimal successors of UCB1:
 *
 * <ul>
 *   <li>{@link Method#UCB1} (Auer et al. 2002): mean plus {@code sqrt(2 log t / n)}, the
 *       Hoeffding bonus that is loose for Bernoulli means near 0 or 1;</li>
 *   <li>{@link Method#EPSILON_GREEDY}: the greedy arm, a uniformly random one with probability
 *       {@code min(1, c K / t)};</li>
 *   <li>{@link Method#KL_UCB} (Garivier &amp; Cappe, COLT 2011,
 *       <a href="https://arxiv.org/abs/1102.2490">arXiv:1102.2490</a>): the largest plausible
 *       mean {@code q} with {@code n KL(mean, q) <= log t}, which matches the Lai-Robbins lower
 *       bound for Bernoulli rewards;</li>
 *   <li>{@link Method#THOMPSON_BETA} (Agrawal &amp; Goyal, COLT 2012,
 *       <a href="https://arxiv.org/abs/1111.1797">arXiv:1111.1797</a>): a draw from each arm's
 *       Beta posterior, with rewards in [0, 1] binarised by a Bernoulli trial;
 *       {@link Method#THOMPSON_GAUSSIAN}: a draw from {@code N(mean, sigma^2 / (n + 1))};</li>
 *   <li>{@link Method#MOSS} (Audibert &amp; Bubeck, COLT 2009): the horizon-dependent bonus
 *       {@code sqrt(max(0, log(T / (K n))) / n)}, minimax optimal;</li>
 *   <li>{@link Method#IMED} (Honda &amp; Takemura, JMLR 2015): the deterministic arm minimising
 *       {@code n KL(mean, best mean) + log n}, asymptotically optimal.</li>
 * </ul>
 *
 * <p>KL-UCB and IMED use the Bernoulli divergence by default; {@link #gaussian(double)} switches
 * them (and the Thompson variance) to Gaussian arms of known standard deviation.
 */
public final class StochasticBandits implements BanditPolicy {

    /** The policy. */
    public enum Method {
        /** upper confidence bound with the Hoeffding bonus */
        UCB1,
        /** epsilon-greedy with decaying epsilon */
        EPSILON_GREEDY,
        /** KL upper confidence bound */
        KL_UCB,
        /** Beta-Bernoulli Thompson sampling */
        THOMPSON_BETA,
        /** Gaussian Thompson sampling */
        THOMPSON_GAUSSIAN,
        /** minimax optimal strategy in the stochastic case */
        MOSS,
        /** indexed minimum empirical divergence */
        IMED
    }

    private final Method method;
    private final int arms;
    private final int horizon;
    private final SplittableRandom random;
    private final int[] pulls;
    private final double[] sums;
    private final double[] successes;
    private BanditMath.Family family = BanditMath.Family.BERNOULLI;
    private double sigma = 0.5;
    private double epsilonScale = 5;
    private long t;

    private StochasticBandits(Method method, int arms, int horizon, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.arms = arms;
        this.horizon = horizon;
        this.random = new SplittableRandom(seed);
        this.pulls = new int[arms];
        this.sums = new double[arms];
        this.successes = new double[arms];
    }

    /**
     * @param method the policy
     * @param arms the arms
     * @param horizon the horizon (used by MOSS only)
     * @param seed the seed of the random choices
     * @return the policy
     */
    public static StochasticBandits of(Method method, int arms, int horizon, long seed) {
        return new StochasticBandits(method, arms, horizon, seed);
    }

    /**
     * Treats rewards as Gaussian with a known standard deviation in KL-UCB, IMED and Gaussian
     * Thompson sampling.
     *
     * @param standardDeviation the noise standard deviation
     * @return this policy
     */
    public StochasticBandits gaussian(double standardDeviation) {
        this.family = BanditMath.Family.GAUSSIAN;
        this.sigma = standardDeviation;
        return this;
    }

    /**
     * @param c epsilon-greedy's scale: epsilon is {@code min(1, c K / t)}
     * @return this policy
     */
    public StochasticBandits epsilonScale(double c) {
        this.epsilonScale = c;
        return this;
    }

    @Override
    public int arms() {
        return arms;
    }

    private double mean(int a) {
        return sums[a] / pulls[a];
    }

    @Override
    public int select() {
        for (int a = 0; a < arms; a++) {
            if (pulls[a] == 0 && method != Method.THOMPSON_BETA && method != Method.THOMPSON_GAUSSIAN) {
                return a;
            }
        }
        double[] index = new double[arms];
        double logT = Math.log(Math.max(t, 1));
        switch (method) {
            case UCB1 -> {
                for (int a = 0; a < arms; a++) {
                    index[a] = mean(a) + Math.sqrt(2 * logT / pulls[a]);
                }
            }
            case EPSILON_GREEDY -> {
                if (random.nextDouble() < Math.min(1, epsilonScale * arms / (double) Math.max(t, 1))) {
                    return random.nextInt(arms);
                }
                for (int a = 0; a < arms; a++) {
                    index[a] = mean(a);
                }
            }
            case KL_UCB -> {
                for (int a = 0; a < arms; a++) {
                    index[a] = BanditMath.klUpper(family, sigma, mean(a), pulls[a], logT);
                }
            }
            case THOMPSON_BETA -> {
                for (int a = 0; a < arms; a++) {
                    index[a] = BanditMath.beta(1 + successes[a], 1 + pulls[a] - successes[a], random);
                }
            }
            case THOMPSON_GAUSSIAN -> {
                for (int a = 0; a < arms; a++) {
                    double m = pulls[a] == 0 ? 0 : mean(a);
                    index[a] = m + sigma / Math.sqrt(pulls[a] + 1) * random.nextGaussian();
                }
            }
            case MOSS -> {
                for (int a = 0; a < arms; a++) {
                    index[a] = mean(a) + Math.sqrt(Math.max(0, Math.log(horizon / (arms * (double) pulls[a])))
                            / pulls[a]);
                }
            }
            case IMED -> {
                double best = Double.NEGATIVE_INFINITY;
                for (int a = 0; a < arms; a++) {
                    best = Math.max(best, mean(a));
                }
                for (int a = 0; a < arms; a++) {
                    double m = mean(a);
                    double d = m >= best ? 0 : BanditMath.kl(family, sigma, m, best);
                    index[a] = -(pulls[a] * d + Math.log(pulls[a]));
                }
            }
            default -> throw new IllegalStateException();
        }
        return BanditMath.argmax(index, random);
    }

    @Override
    public void update(int arm, double reward) {
        t++;
        pulls[arm]++;
        sums[arm] += reward;
        double r = Math.min(1, Math.max(0, reward));
        successes[arm] += random.nextDouble() < r ? 1 : 0;
    }

    @Override
    public String toString() {
        return method.toString();
    }
}
