package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.experimental.data.BanditSuites;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.recsys.ClickDebiasing;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Online learning to rank with partial click feedback: each round the policy shows a list of
 * {@code K} of {@code L} items and sees clicks per position.
 *
 * <ul>
 *   <li>{@link Method#CUCB} (Chen et al., ICML 2013): combinatorial UCB that shows the top
 *       {@code K} items by UCB1 index and treats <em>every</em> shown item as observed (semi-bandit
 *       feedback), which under a cascade biases the estimates of items below the click;</li>
 *   <li>{@link Method#CASCADE_UCB1} and {@link Method#CASCADE_KL_UCB} (Kveton et al., ICML 2015,
 *       <a href="https://arxiv.org/abs/1502.02763">arXiv:1502.02763</a>): the cascade model's
 *       exact feedback, every item up to and including the first click is observed (the
 *       clicked one attractive, the others not), nothing below it;</li>
 *   <li>{@link Method#PBM_UCB} (Lagree et al., NeurIPS 2016): under the position-based model,
 *       attraction is estimated as clicks over the summed examination probabilities of the
 *       positions an item was shown at (the examination curve is assumed known);</li>
 *   <li>{@link Method#CASCADE_LIN_UCB} (Zong et al., UAI 2016): a linear contextual cascade
 *       version, one shared ridge model of attraction on item features with the cascade's
 *       feedback;</li>
 *   <li>{@link Method#ITEMWISE_LIN_UCB} and {@link Method#ITEMWISE_LIN_TS}: the same ridge model
 *       scored item by item as in {@code ContextualBandit}'s LinUCB and linear Thompson sampling,
 *       updated with every shown item (unclicked = 0), ignoring the cascade;</li>
 *   <li>{@link Method#RANDOM}: random lists.</li>
 * </ul>
 *
 * <p>{@link #offlineUltr} is the offline baseline: a linear ranker trained with inverse
 * propensity weighting ({@link ClickDebiasing#trainIpw}) on clicks logged under random lists,
 * then deployed as a fixed ranking.
 */
public final class RankingBandits {

    /** The policy. */
    public enum Method {
        /** combinatorial UCB with semi-bandit feedback */
        CUCB,
        /** CascadeUCB1 */
        CASCADE_UCB1,
        /** CascadeKL-UCB */
        CASCADE_KL_UCB,
        /** UCB for the position-based model */
        PBM_UCB,
        /** CascadeLinUCB on item features */
        CASCADE_LIN_UCB,
        /** item-wise LinUCB ignoring the cascade */
        ITEMWISE_LIN_UCB,
        /** item-wise linear Thompson sampling ignoring the cascade */
        ITEMWISE_LIN_TS,
        /** random lists */
        RANDOM
    }

    private final Method method;
    private final BanditSuites.ClickItems items;
    private final int l;
    private final int k;
    private final SplittableRandom random;
    private final double[] observed;
    private final double[] clicks;
    private final double[][] a;
    private final double[] b;
    private double alpha = 1;
    private long t;

    private RankingBandits(Method method, BanditSuites.ClickItems items, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.items = items;
        this.l = items.items();
        this.k = items.listLength();
        this.random = new SplittableRandom(seed);
        this.observed = new double[l];
        this.clicks = new double[l];
        int d = items.features()[0].length + 1;
        this.a = MlMatrices.identity(d);
        this.b = new double[d];
    }

    /**
     * @param method the policy
     * @param items the items (the policy reads their features and examination curve only)
     * @param seed the seed
     * @return the policy
     */
    public static RankingBandits of(Method method, BanditSuites.ClickItems items, long seed) {
        return new RankingBandits(method, items, seed);
    }

    /**
     * @param scale the width of the linear confidence bonus or posterior
     * @return this policy
     */
    public RankingBandits alpha(double scale) {
        this.alpha = scale;
        return this;
    }

    private double[] feature(int i) {
        double[] x = items.features()[i];
        double[] f = new double[x.length + 1];
        System.arraycopy(x, 0, f, 0, x.length);
        f[x.length] = 1;
        return f;
    }

    /** @return the list to show, top first */
    public int[] select() {
        t++;
        double[] score = new double[l];
        double log = Math.log(Math.max(t, 2));
        switch (method) {
            case RANDOM -> {
                for (int i = 0; i < l; i++) {
                    score[i] = random.nextDouble();
                }
            }
            case CUCB, CASCADE_UCB1, PBM_UCB -> {
                for (int i = 0; i < l; i++) {
                    score[i] = observed[i] == 0 ? 1e9 + random.nextDouble()
                            : clicks[i] / observed[i] + Math.sqrt(1.5 * log / observed[i]);
                }
            }
            case CASCADE_KL_UCB -> {
                for (int i = 0; i < l; i++) {
                    score[i] = observed[i] == 0 ? 1e9 + random.nextDouble()
                            : BanditMath.klUpper(BanditMath.Family.BERNOULLI, 0, clicks[i] / observed[i],
                                    (int) Math.max(1, Math.round(observed[i])), log);
                }
            }
            case CASCADE_LIN_UCB, ITEMWISE_LIN_UCB, ITEMWISE_LIN_TS -> {
                double[][] lower = MlMatrices.cholesky(a);
                double[] theta = MlMatrices.choleskySolve(lower, b);
                if (method == Method.ITEMWISE_LIN_TS) {
                    double[][] cov = MlMatrices.choleskyInverse(lower);
                    double[][] c = MlMatrices.cholesky(cov);
                    double[] z = new double[theta.length];
                    for (int i = 0; i < z.length; i++) {
                        z[i] = random.nextGaussian();
                    }
                    theta = theta.clone();
                    for (int i = 0; i < z.length; i++) {
                        for (int j = 0; j <= i; j++) {
                            theta[i] += alpha * c[i][j] * z[j];
                        }
                    }
                }
                for (int i = 0; i < l; i++) {
                    double[] f = feature(i);
                    score[i] = MlMatrices.dot(theta, f);
                    if (method != Method.ITEMWISE_LIN_TS) {
                        score[i] += alpha * Math.sqrt(MlMatrices.dot(f, MlMatrices.choleskySolve(lower, f)));
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
        final double[] s = score;
        return java.util.stream.IntStream.range(0, l).boxed().sorted((x, y) -> Double.compare(s[y], s[x]))
                .limit(k).mapToInt(Integer::intValue).toArray();
    }

    /**
     * Records the clicks on a shown list.
     *
     * @param list the list shown
     * @param click a 0/1 click per position
     */
    public void update(int[] list, int[] click) {
        int first = list.length;
        for (int p = 0; p < list.length; p++) {
            if (click[p] == 1) {
                first = p;
                break;
            }
        }
        for (int p = 0; p < list.length; p++) {
            int i = list[p];
            switch (method) {
                case RANDOM -> {
                    // nothing to learn
                }
                case CUCB -> observe(i, click[p]);
                case PBM_UCB -> {
                    observed[i] += items.examination()[p];
                    clicks[i] += click[p];
                }
                case CASCADE_UCB1, CASCADE_KL_UCB -> {
                    if (p <= first) {
                        observe(i, p == first ? 1 : 0);
                    }
                }
                case CASCADE_LIN_UCB -> {
                    if (p <= first) {
                        ridge(i, p == first ? 1 : 0);
                    }
                }
                case ITEMWISE_LIN_UCB, ITEMWISE_LIN_TS -> ridge(i, click[p]);
                default -> throw new IllegalStateException();
            }
        }
    }

    private void observe(int i, double c) {
        observed[i]++;
        clicks[i] += c;
    }

    private void ridge(int i, double y) {
        double[] f = feature(i);
        for (int r = 0; r < f.length; r++) {
            for (int c = 0; c < f.length; c++) {
                a[r][c] += f[r] * f[c];
            }
            b[r] += y * f[r];
        }
    }

    /**
     * A ranking run's scores.
     *
     * @param regret the cumulative expected regret against the best list
     * @param clickThrough the share of rounds with at least one click
     */
    public record Run(double regret, double clickThrough) {
    }

    /**
     * Plays a ranking policy.
     *
     * @param items the items and click model
     * @param policy the policy (fresh)
     * @param rounds the rounds
     * @param seed the seed of the clicks
     * @return the scores
     */
    public static Run run(BanditSuites.ClickItems items, RankingBandits policy, int rounds, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double best = items.value(items.bestList());
        double regret = 0;
        int clicked = 0;
        for (int t = 0; t < rounds; t++) {
            int[] list = policy.select();
            int[] c = items.clicks(list, r);
            policy.update(list, c);
            regret += best - items.value(list);
            for (int x : c) {
                if (x == 1) {
                    clicked++;
                    break;
                }
            }
        }
        return new Run(regret, clicked / (double) rounds);
    }

    /**
     * The offline unbiased-learning-to-rank baseline: clicks are logged on random lists, a
     * linear ranker is trained on them with inverse propensity weighting by the examination
     * curve, and its fixed ranking is then shown for {@code rounds} rounds.
     *
     * @param items the items and click model
     * @param loggedSessions logged sessions
     * @param steps training steps
     * @param rounds deployment rounds
     * @param seed the seed
     * @return the deployed ranking's scores
     */
    public static Run offlineUltr(BanditSuites.ClickItems items, int loggedSessions, int steps, int rounds, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int k = items.listLength();
        int d = items.features()[0].length;
        double[][][] feats = new double[loggedSessions][k][];
        int[][] logged = new int[loggedSessions][];
        RankingBandits logger = of(Method.RANDOM, items, seed + 1);
        for (int s = 0; s < loggedSessions; s++) {
            int[] list = logger.select();
            logged[s] = items.clicks(list, r);
            for (int p = 0; p < k; p++) {
                feats[s][p] = items.features()[list[p]];
            }
        }
        Model ranker = Nets.mlp(d, 1, seed, 16);
        ClickDebiasing.trainIpw(ranker, new ClickDebiasing.Impressions(feats, logged), items.examination(), steps, 32,
                3e-3f, seed);
        double[] score = new double[items.items()];
        for (int i = 0; i < score.length; i++) {
            float[] x = new float[d];
            for (int j = 0; j < d; j++) {
                x[j] = (float) items.features()[i][j];
            }
            score[i] = io.github.zzuegg.brainiac.experimental.rl.RlTensors.row(ranker, x)[0];
        }
        int[] list = java.util.stream.IntStream.range(0, score.length).boxed()
                .sorted((x, y) -> Double.compare(score[y], score[x])).limit(k).mapToInt(Integer::intValue).toArray();
        double best = items.value(items.bestList());
        double v = items.value(list);
        int clicked = 0;
        for (int t = 0; t < rounds; t++) {
            for (int c : items.clicks(list, r)) {
                if (c == 1) {
                    clicked++;
                    break;
                }
            }
        }
        return new Run(rounds * (best - v), clicked / (double) rounds);
    }

    @Override
    public String toString() {
        return method.toString();
    }
}
