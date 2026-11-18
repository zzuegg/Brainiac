package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A learned improvement heuristic: a policy that, given the current tour, picks the next
 * 2-opt move. The tour is encoded by a {@link TransformerBlock} over the nodes in tour order,
 * each with its coordinates and a sinusoidal encoding of its tour position (the order is in
 * the input, as in Wu et al.); every move {@code (i, j)} (reverse positions {@code i+1..j})
 * gets a compatibility score {@code q_i . k_j / sqrt(d)} plus a learned multiple of the move's
 * normalised length gain, the scores of all legal moves are softmaxed, and a critic reads the
 * mean embedding.
 *
 * <p>Training is n-step advantage actor-critic on the best-so-far reward
 * {@code r_t = best_{t-1} - best_t} (Wu et al.), so a move that lengthens the tour costs
 * nothing as long as it leads to a better tour later; this is what lets a learned selector
 * escape the local optima where best-improvement 2-opt stops.
 *
 * <p>Papers: d'O Costa, Rhuggenaath, Zhang and Akcay, "Learning 2-opt Heuristics for the
 * Traveling Salesman Problem via Deep Reinforcement Learning" (ACML 2020),
 * <a href="https://arxiv.org/abs/2004.01608">arXiv:2004.01608</a>; Wu, Song, Cao, Zhang and Lim,
 * "Learning Improvement Heuristics for Solving Routing Problems" (IEEE TNNLS 2021),
 * <a href="https://arxiv.org/abs/1912.05784">arXiv:1912.05784</a>.
 */
public final class LearnedTwoOpt {

    private final int dim;
    private final Var embed;
    private final Var embedBias;
    private final TransformerBlock block;
    private final Var wq;
    private final Var wk;
    private final Var gainWeight;
    private final Var value;
    private final float gamma;

    private LearnedTwoOpt(int dim, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.dim = dim;
        this.embed = AttentionModel.matrix(r, 4, dim);
        this.embedBias = Var.param(Tensor.zeros(Shape.of(dim)));
        this.block = (TransformerBlock) TransformerBlock.of(4).ffnSize(2 * dim).create(Shape.of(1, dim), r);
        this.wq = AttentionModel.matrix(r, dim, dim);
        this.wk = AttentionModel.matrix(r, dim, dim);
        this.gainWeight = Var.param(Tensor.of(Shape.of(1), new float[] {5f}));
        this.value = AttentionModel.matrix(r, dim, 1);
        this.gamma = 0.9f;
    }

    /**
     * Creates a selector.
     *
     * @param dim the embedding width
     * @param seed the seed
     * @return the policy
     */
    public static LearnedTwoOpt of(int dim, long seed) {
        return new LearnedTwoOpt(dim, seed);
    }

    private record Step(Var logp, Var value, int[] moves) {
    }

    /** Scores every move of every tour; samples one per tour. */
    private Step act(List<RoutingEnv> envs, int[][] tours, SplittableRandom random, boolean greedy) {
        int b = envs.size();
        int n = tours[0].length;
        float[] x = new float[b * n * 4];
        float[] gain = new float[b * n * n];
        float[] mask = new float[b * n * n];
        for (int s = 0; s < b; s++) {
            double[][] c = envs.get(s).coords();
            double[][] d = envs.get(s).distances();
            double scale = ClassicHeuristics.tourLength(d, tours[s]) / n;
            for (int p = 0; p < n; p++) {
                int v = tours[s][p];
                int o = (s * n + p) * 4;
                x[o] = (float) c[v][0];
                x[o + 1] = (float) c[v][1];
                x[o + 2] = (float) Math.cos(2 * Math.PI * p / n);
                x[o + 3] = (float) Math.sin(2 * Math.PI * p / n);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int o = (s * n + i) * n + j;
                    boolean legal = j >= i + 2 && !(i == 0 && j == n - 1);
                    mask[o] = legal ? 0 : -1e9f;
                    gain[o] = legal ? (float) (ClassicHeuristics.twoOptGain(d, tours[s], i, j) / scale) : 0;
                }
            }
        }
        Var h = Var.of(Tensor.of(Shape.of(b, n, 4), x)).matmul(embed).add(embedBias);
        h = block.forward(List.of(h), false);
        Var scores = h.matmul(wq).matmul(h.matmul(wk).permute(0, 2, 1)).mul((float) (1 / Math.sqrt(dim))).tanh()
                .mul(5f).add(Var.of(Tensor.of(Shape.of(b, n, n), gain)).mul(gainWeight))
                .add(Tensor.of(Shape.of(b, n, n), mask)).reshape(b, n * n);
        Var lp = scores.logSoftmax(1);
        float[] p = lp.value().toArray();
        int[] pick = new int[b];
        float[] idx = new float[b];
        for (int s = 0; s < b; s++) {
            int best = -1;
            if (greedy) {
                for (int k = 0; k < n * n; k++) {
                    if (best < 0 || p[s * n * n + k] > p[s * n * n + best]) {
                        best = k;
                    }
                }
            } else {
                double u = random.nextDouble();
                for (int k = 0; k < n * n; k++) {
                    if (mask[s * n * n + k] == 0) {
                        best = k;
                        u -= Math.exp(p[s * n * n + k]);
                        if (u <= 0) {
                            break;
                        }
                    }
                }
            }
            pick[s] = best;
            idx[s] = best;
        }
        Var v = h.mean(1, false).matmul(value);
        return new Step(lp.takeAlong(Tensor.of(Shape.of(b, 1), idx)), v, pick);
    }

    /**
     * Trains on episodes from random tours.
     *
     * @param n nodes per instance
     * @param episodes episodes
     * @param batch instances per episode
     * @param horizon moves per episode
     * @param nStep moves per update
     * @param optimizer the optimizer
     * @param random the source
     * @return the mean final best-so-far length of the last episode
     */
    public double train(int n, int episodes, int batch, int horizon, int nStep, Optimizer optimizer,
            SplittableRandom random) {
        double last = 0;
        for (int e = 0; e < episodes; e++) {
            List<RoutingEnv> envs = new ArrayList<>();
            int[][] tours = new int[batch][];
            double[] best = new double[batch];
            for (int s = 0; s < batch; s++) {
                envs.add(RoutingEnv.tsp(io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.points(n, false,
                        random)));
                tours[s] = ClassicHeuristics.randomTour(n, random);
                best[s] = envs.get(s).length(tours[s]);
            }
            for (int t0 = 0; t0 < horizon; t0 += nStep) {
                List<Var> logps = new ArrayList<>();
                List<Var> values = new ArrayList<>();
                List<float[]> rewards = new ArrayList<>();
                for (int k = 0; k < nStep; k++) {
                    Step st = act(envs, tours, random, false);
                    float[] rw = new float[batch];
                    for (int s = 0; s < batch; s++) {
                        int i = st.moves()[s] / n;
                        int j = st.moves()[s] % n;
                        ClassicHeuristics.applyTwoOpt(tours[s], i, j);
                        double len = envs.get(s).length(tours[s]);
                        if (len < best[s]) {
                            rw[s] = (float) ((best[s] - len) * 10 / Math.sqrt(n));
                            best[s] = len;
                        }
                    }
                    logps.add(st.logp());
                    values.add(st.value());
                    rewards.add(rw);
                }
                float[] boot = Var.noGrad(() -> act(envs, tours, random, false)).value().value().toArray();
                float[] ret = boot.clone();
                Var loss = null;
                for (int k = nStep - 1; k >= 0; k--) {
                    float[] rw = rewards.get(k);
                    for (int s = 0; s < batch; s++) {
                        ret[s] = rw[s] + gamma * ret[s];
                    }
                    Tensor target = Tensor.of(Shape.of(batch, 1), ret.clone());
                    Var v = values.get(k);
                    Tensor adv = target.sub(v.value());
                    Var term = logps.get(k).mul(adv).neg().add(v.sub(target).square().mul(0.5f)).mean();
                    loss = loss == null ? term : loss.add(term);
                }
                AttentionModel.step(loss.mul(1f / nStep), parameters(), optimizer, 1.0);
            }
            double m = 0;
            for (double v : best) {
                m += v;
            }
            last = m / batch;
        }
        return last;
    }

    /**
     * Improves a tour by {@code moves} sampled moves, keeping the best tour seen.
     *
     * @param env the instance
     * @param start the starting tour
     * @param moves the move budget
     * @param random the source
     * @return the best tour
     */
    public ClassicHeuristics.Improvement improve(RoutingEnv env, int[] start, int moves, SplittableRandom random) {
        return improveAll(List.of(env), new int[][] {start}, moves, random).get(0);
    }

    /**
     * Improves a batch of same-size tours in step, each keeping its best tour seen.
     *
     * @param envs the instances
     * @param starts the starting tours
     * @param moves the move budget
     * @param random the source
     * @return the best tours
     */
    public List<ClassicHeuristics.Improvement> improveAll(List<RoutingEnv> envs, int[][] starts, int moves,
            SplittableRandom random) {
        int b = envs.size();
        int n = starts[0].length;
        int[][] t = new int[b][];
        int[][] best = new int[b][];
        double[] bestLen = new double[b];
        for (int s = 0; s < b; s++) {
            t[s] = starts[s].clone();
            best[s] = t[s].clone();
            bestLen[s] = envs.get(s).length(t[s]);
        }
        for (int m = 0; m < moves; m++) {
            Step st = Var.noGrad(() -> act(envs, t, random, false));
            for (int s = 0; s < b; s++) {
                ClassicHeuristics.applyTwoOpt(t[s], st.moves()[s] / n, st.moves()[s] % n);
                double len = envs.get(s).length(t[s]);
                if (len < bestLen[s] - 1e-12) {
                    bestLen[s] = len;
                    best[s] = t[s].clone();
                }
            }
        }
        List<ClassicHeuristics.Improvement> out = new ArrayList<>();
        for (int s = 0; s < b; s++) {
            out.add(new ClassicHeuristics.Improvement(best[s], moves, bestLen[s]));
        }
        return out;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias));
        p.addAll(block.parameters());
        p.addAll(List.of(wq, wk, gainWeight, value));
        return p;
    }
}
