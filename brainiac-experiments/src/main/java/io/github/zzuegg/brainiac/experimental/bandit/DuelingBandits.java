package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.BanditSuites;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Dueling bandits: each round the learner picks two arms and sees only which one won, as in
 * interleaved ranking or RLHF comparisons; regret is measured against the Condorcet winner
 * (or the Copeland winners when there is none, see {@link BanditSuites.Preferences#regret}).
 *
 * <ul>
 *   <li>{@link Method#RUCB} (Zoghi et al., ICML 2014,
 *       <a href="https://arxiv.org/abs/1312.3393">arXiv:1312.3393</a>): optimistic pairwise win
 *       probabilities {@code U = W / (W + W^T) + sqrt(alpha log t / (W + W^T))}; a candidate
 *       that could beat everyone ({@code U_cj >= 1/2} for all {@code j}) duels its strongest
 *       optimistic challenger {@code argmax_j U_jc};</li>
 *   <li>{@link Method#DOUBLE_TS} (Wu &amp; Liu, NeurIPS 2016,
 *       <a href="https://arxiv.org/abs/1604.07101">arXiv:1604.07101</a>): among the arms with the
 *       highest optimistic Copeland score, the first arm is the Copeland winner of one Beta
 *       sample of the preference matrix; the second is the arm that a fresh sample says beats
 *       it most, among arms whose lower bound does not rule that out. It handles instances
 *       without a Condorcet winner;</li>
 *   <li>{@link Method#INDEPENDENT_TS}: the reduction that treats each duel as two independent
 *       Bernoulli pulls (win = 1) with Thompson sampling on each arm;</li>
 *   <li>{@link Method#BT_GREEDY}: a Bradley-Terry model refitted every {@code refit} duels with
 *       {@link PreferenceLosses#bradleyTerry}, dueling its top two arms (epsilon-greedy
 *       exploration);</li>
 *   <li>{@link Method#UNIFORM}: random duels.</li>
 * </ul>
 */
public final class DuelingBandits {

    /** The policy. */
    public enum Method {
        /** relative upper confidence bound */
        RUCB,
        /** double Thompson sampling */
        DOUBLE_TS,
        /** independent Thompson sampling on each arm */
        INDEPENDENT_TS,
        /** refitted Bradley-Terry with greedy duels */
        BT_GREEDY,
        /** uniformly random duels */
        UNIFORM
    }

    private final Method method;
    private final int k;
    private final SplittableRandom random;
    private final double[][] wins;
    private final double alpha = 0.51;
    private long t;
    private int refit = 100;
    private double[] utilities;
    private final List<int[]> history = new ArrayList<>();

    private DuelingBandits(Method method, int arms, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.k = arms;
        this.random = new SplittableRandom(seed);
        this.wins = new double[arms][arms];
        this.utilities = new double[arms];
    }

    /**
     * @param method the policy
     * @param arms the arms
     * @param seed the seed
     * @return the policy
     */
    public static DuelingBandits of(Method method, int arms, long seed) {
        return new DuelingBandits(method, arms, seed);
    }

    /**
     * @param duels duels between Bradley-Terry refits
     * @return this policy
     */
    public DuelingBandits refitEvery(int duels) {
        this.refit = duels;
        return this;
    }

    private double u(int i, int j, double sign) {
        if (i == j) {
            return 0.5;
        }
        double n = wins[i][j] + wins[j][i];
        if (n == 0) {
            return sign > 0 ? 1 : 0;
        }
        return wins[i][j] / n + sign * Math.sqrt(alpha * Math.log(Math.max(t, 2)) / n);
    }

    /** @return the two arms to duel next */
    public int[] select() {
        t++;
        return switch (method) {
            case UNIFORM -> new int[] {random.nextInt(k), random.nextInt(k)};
            case INDEPENDENT_TS -> {
                double[] d1 = new double[k];
                double[] d2 = new double[k];
                for (int a = 0; a < k; a++) {
                    double w = 0;
                    double l = 0;
                    for (int b = 0; b < k; b++) {
                        w += wins[a][b];
                        l += wins[b][a];
                    }
                    d1[a] = BanditMath.beta(1 + w, 1 + l, random);
                    d2[a] = BanditMath.beta(1 + w, 1 + l, random);
                }
                yield new int[] {BanditMath.argmax(d1, random), BanditMath.argmax(d2, random)};
            }
            case RUCB -> rucb();
            case DOUBLE_TS -> doubleTs();
            case BT_GREEDY -> {
                int[] order = java.util.stream.IntStream.range(0, k).boxed()
                        .sorted((a, b) -> Double.compare(utilities[b], utilities[a])).mapToInt(Integer::intValue)
                        .toArray();
                if (random.nextDouble() < 0.1) {
                    yield new int[] {order[0], random.nextInt(k)};
                }
                yield new int[] {order[0], order[Math.min(1, k - 1)]};
            }
        };
    }

    private int[] rucb() {
        List<Integer> candidates = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            boolean ok = true;
            for (int j = 0; j < k; j++) {
                if (u(c, j, 1) < 0.5) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                candidates.add(c);
            }
        }
        int c = candidates.isEmpty() ? random.nextInt(k) : candidates.get(random.nextInt(candidates.size()));
        double[] challenge = new double[k];
        for (int j = 0; j < k; j++) {
            challenge[j] = j == c ? -1 : u(j, c, 1);
        }
        int d = BanditMath.argmax(challenge, random);
        return new int[] {c, d};
    }

    private int[] doubleTs() {
        double[] upperScore = new double[k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (j != i && u(i, j, 1) > 0.5) {
                    upperScore[i]++;
                }
            }
        }
        double top = java.util.Arrays.stream(upperScore).max().orElse(0);
        double[][] theta = new double[k][k];
        for (int i = 0; i < k; i++) {
            theta[i][i] = 0.5;
            for (int j = i + 1; j < k; j++) {
                theta[i][j] = BanditMath.beta(wins[i][j] + 1, wins[j][i] + 1, random);
                theta[j][i] = 1 - theta[i][j];
            }
        }
        double[] score = new double[k];
        for (int i = 0; i < k; i++) {
            if (upperScore[i] < top) {
                score[i] = -1;
                continue;
            }
            for (int j = 0; j < k; j++) {
                if (j != i && theta[i][j] > 0.5) {
                    score[i]++;
                }
            }
        }
        int first = BanditMath.argmax(score, random);
        double[] second = new double[k];
        for (int i = 0; i < k; i++) {
            if (i == first) {
                second[i] = 0.5;
            } else if (u(i, first, -1) > 0.5) {
                second[i] = -1;
            } else {
                second[i] = BanditMath.beta(wins[i][first] + 1, wins[first][i] + 1, random);
            }
        }
        boolean any = false;
        for (int i = 0; i < k; i++) {
            if (i != first && u(i, first, -1) <= 0.5) {
                any = true;
                break;
            }
        }
        if (!any) {
            return new int[] {first, first};
        }
        return new int[] {first, BanditMath.argmax(second, random)};
    }

    /**
     * Records a duel's outcome.
     *
     * @param i the first arm
     * @param j the second arm
     * @param firstWon whether {@code i} won
     */
    public void update(int i, int j, boolean firstWon) {
        if (i == j) {
            return;
        }
        if (firstWon) {
            wins[i][j]++;
        } else {
            wins[j][i]++;
        }
        if (method == Method.BT_GREEDY) {
            history.add(firstWon ? new int[] {i, j} : new int[] {j, i});
            if (history.size() % refit == 0) {
                fitBradleyTerry();
            }
        }
    }

    private void fitBradleyTerry() {
        int n = history.size();
        float[] w = new float[n * k];
        float[] l = new float[n * k];
        for (int r = 0; r < n; r++) {
            w[r * k + history.get(r)[0]] = 1;
            l[r * k + history.get(r)[1]] = 1;
        }
        Tensor winners = Tensor.of(Shape.of(n, k), w);
        Tensor losers = Tensor.of(Shape.of(n, k), l);
        float[] init = new float[k];
        for (int a = 0; a < k; a++) {
            init[a] = (float) utilities[a];
        }
        Var u = Var.param(Tensor.of(Shape.of(k, 1), init));
        Optimizer opt = Optimizer.adam(0.1f);
        for (int it = 0; it < 60; it++) {
            Var loss = PreferenceLosses.bradleyTerry(Var.of(winners).matmul(u), Var.of(losers).matmul(u))
                    .add(u.square().mean().mul(1e-3f));
            loss.backward();
            opt.step(List.of(u));
        }
        float[] out = u.value().toArray();
        utilities = new double[k];
        for (int a = 0; a < k; a++) {
            utilities[a] = out[a];
        }
    }

    /**
     * A dueling run's scores.
     *
     * @param regret the cumulative regret
     * @param identified the arm the policy played most often as first arm
     * @param winnerShare the share of duels in the last tenth whose first arm was a Condorcet or
     *     Copeland winner
     */
    public record Run(double regret, int identified, double winnerShare) {
    }

    /**
     * Plays a dueling policy against a preference matrix.
     *
     * @param prefs the preferences
     * @param policy the policy (fresh)
     * @param horizon the duels
     * @param seed the seed of the outcomes
     * @return the scores
     */
    public static Run run(BanditSuites.Preferences prefs, DuelingBandits policy, int horizon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double regret = 0;
        int[] first = new int[prefs.arms()];
        int c = prefs.condorcetWinner();
        java.util.Set<Integer> winners = new java.util.HashSet<>();
        if (c >= 0) {
            winners.add(c);
        } else {
            for (int w : prefs.copelandWinners()) {
                winners.add(w);
            }
        }
        int late = 0;
        int lateHits = 0;
        for (int t = 0; t < horizon; t++) {
            int[] d = policy.select();
            boolean won = prefs.duel(d[0], d[1], r);
            policy.update(d[0], d[1], won);
            regret += prefs.regret(d[0], d[1]);
            first[d[0]]++;
            if (t >= horizon - horizon / 10) {
                late++;
                lateHits += winners.contains(d[0]) ? 1 : 0;
            }
        }
        int id = 0;
        for (int a = 1; a < first.length; a++) {
            if (first[a] > first[id]) {
                id = a;
            }
        }
        return new Run(regret, id, lateHits / (double) Math.max(1, late));
    }

    @Override
    public String toString() {
        return method.toString();
    }
}
