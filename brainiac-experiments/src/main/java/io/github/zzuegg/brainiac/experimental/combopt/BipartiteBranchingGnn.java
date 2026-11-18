package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Ilp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Gasse et al.'s branching GNN: the LP relaxation at a branch-and-bound node as a bipartite
 * graph of variables and constraints, with one half-convolution from variables to
 * constraints and one back, scoring every variable; trained by imitation (cross-entropy over
 * the fractional candidates) of strong branching's choices, recorded with
 * {@link BranchAndBound#record}. As a {@link BranchAndBound.Scorer} it drops into
 * {@link BranchAndBound.Rule#LEARNED}.
 *
 * <p>Variable features: objective coefficient (scaled), LP value, fractionality, whether
 * fixed, reduced cost {@code c_j - A_j^T y} (scaled), column density and node depth.
 * Constraint features: bound over row sum, dual value (scaled) and slack. Edges carry
 * {@code A_ij} normalised by the row sum.
 *
 * <p>Paper: Gasse, Chételat, Ferroni, Charlin and Lodi, "Exact Combinatorial Optimization with
 * Graph Convolutional Neural Networks" (NeurIPS 2019),
 * <a href="https://arxiv.org/abs/1906.01629">arXiv:1906.01629</a>.
 */
public final class BipartiteBranchingGnn implements BranchAndBound.Scorer {

    private static final int VAR_FEATURES = 7;
    private static final int CON_FEATURES = 3;

    private final Var wv;
    private final Var wc;
    private final Var vToC;
    private final Var cToV;
    private final Var self;
    private final Var out;

    /**
     * Creates the network.
     *
     * @param hidden the embedding width
     * @param seed the seed
     */
    public BipartiteBranchingGnn(int hidden, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.wv = AttentionModel.matrix(r, VAR_FEATURES, hidden);
        this.wc = AttentionModel.matrix(r, CON_FEATURES, hidden);
        this.vToC = AttentionModel.matrix(r, hidden, hidden);
        this.cToV = AttentionModel.matrix(r, hidden, hidden);
        this.self = AttentionModel.matrix(r, hidden, hidden);
        this.out = AttentionModel.matrix(r, hidden, 1);
    }

    private Var forward(BranchAndBound.NodeState s) {
        Ilp ilp = s.ilp();
        int n = ilp.variables();
        int m = ilp.b().length;
        double cmax = 1e-9;
        for (double c : ilp.c()) {
            cmax = Math.max(cmax, Math.abs(c));
        }
        float[] vf = new float[n * VAR_FEATURES];
        float[] cf = new float[m * CON_FEATURES];
        float[] edge = new float[m * n];
        double[] rowSum = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rowSum[i] += ilp.a()[i][j];
            }
            rowSum[i] = Math.max(rowSum[i], 1e-9);
            double used = 0;
            for (int j = 0; j < n; j++) {
                edge[i * n + j] = (float) (ilp.a()[i][j] / rowSum[i]);
                used += ilp.a()[i][j] * s.x()[j];
            }
            cf[i * CON_FEATURES] = (float) (ilp.b()[i] / rowSum[i]);
            cf[i * CON_FEATURES + 1] = (float) (s.dual()[i] / cmax);
            cf[i * CON_FEATURES + 2] = (float) ((ilp.b()[i] - used) / rowSum[i]);
        }
        for (int j = 0; j < n; j++) {
            double reduced = ilp.c()[j];
            double density = 0;
            for (int i = 0; i < m; i++) {
                reduced -= ilp.a()[i][j] * s.dual()[i];
                density += ilp.a()[i][j] != 0 ? 1 : 0;
            }
            double x = s.x()[j];
            int o = j * VAR_FEATURES;
            vf[o] = (float) (ilp.c()[j] / cmax);
            vf[o + 1] = (float) x;
            vf[o + 2] = (float) Math.min(x, 1 - x);
            vf[o + 3] = s.fixed()[j] >= 0 ? 1 : 0;
            vf[o + 4] = (float) (reduced / cmax);
            vf[o + 5] = (float) (density / m);
            vf[o + 6] = (float) (s.depth() / (double) n);
        }
        Var e = Var.of(Tensor.of(Shape.of(m, n), edge));
        Var hv = Var.of(Tensor.of(Shape.of(n, VAR_FEATURES), vf)).matmul(wv).relu();
        Var hc = Var.of(Tensor.of(Shape.of(m, CON_FEATURES), cf)).matmul(wc).add(e.matmul(hv).matmul(vToC)).relu();
        Var hv2 = hv.matmul(self).add(e.transpose().matmul(hc).matmul(cToV)).relu();
        return hv2.matmul(out).reshape(n);
    }

    @Override
    public double[] score(BranchAndBound.NodeState node, int[] candidates) {
        float[] s = Var.noGrad(() -> forward(node)).value().toArray();
        double[] out2 = new double[candidates.length];
        for (int k = 0; k < candidates.length; k++) {
            out2[k] = s[candidates[k]];
        }
        return out2;
    }

    /**
     * Trains by imitation on recorded strong-branching samples.
     *
     * @param samples the samples
     * @param epochs passes
     * @param learningRate the Adam rate
     * @param random shuffling
     * @return the last epoch's top-1 agreement with strong branching
     */
    public double train(List<BranchAndBound.Sample> samples, int epochs, float learningRate, SplittableRandom random) {
        Optimizer opt = Optimizer.adam(learningRate);
        double acc = 0;
        int[] order = new int[samples.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int e = 0; e < epochs; e++) {
            for (int i = order.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            int right = 0;
            for (int idx : order) {
                BranchAndBound.Sample s = samples.get(idx);
                float[] cand = new float[s.candidates().length];
                for (int k = 0; k < cand.length; k++) {
                    cand[k] = s.candidates()[k];
                }
                Var logits = forward(s.node()).gather(0, Tensor.of(Shape.of(cand.length), cand)).reshape(1, cand.length);
                float[] v = logits.value().toArray();
                int arg = 0;
                for (int k = 1; k < v.length; k++) {
                    if (v[k] > v[arg]) {
                        arg = k;
                    }
                }
                right += arg == s.best() ? 1 : 0;
                Var loss = Functional.crossEntropy(logits, Tensor.of(Shape.of(1), new float[] {s.best()}));
                loss.backward();
                opt.step(parameters());
            }
            acc = right / (double) Math.max(1, samples.size());
        }
        return acc;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.of(wv, wc, vToC, cToV, self, out);
    }
}
