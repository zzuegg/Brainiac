package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.experimental.nn.SiluMlpExpert;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * MC-SMoE expert merging (Li et al. 2024, "Merge, then compress",
 * <a href="https://arxiv.org/abs/2310.01334">arXiv:2310.01334</a>): instead of dropping experts,
 * fold them into fewer ones.
 * <ol>
 * <li>The most frequently routed experts on calibration data become the group leaders.</li>
 * <li>Every other expert joins the leader whose router logits over the calibration tokens are
 *     most similar (cosine), so experts the router treats alike merge.</li>
 * <li>Each member's hidden neurons are permutation-aligned to the leader's (the Hungarian
 *     assignment of {@link MinibatchOptimalTransport#assignCost} on the neurons' concatenated
 *     input weights, bias and output weights), because two MLPs computing the same function can
 *     order their neurons differently.</li>
 * <li>The aligned weights are averaged, weighted by usage frequency.</li>
 * </ol>
 * {@link #averageAll} (all experts averaged into one, without alignment: sparse upcycling in
 * reverse) is the crude baseline. Works on {@link FineGrainedMoe} layers of {@link SiluMlpExpert}s
 * without shared experts.
 */
public final class ExpertMerging {

    private ExpertMerging() {
    }

    /**
     * Merges a layer's experts into {@code keep} experts.
     *
     * @param moe the layer
     * @param keep the number of merged experts
     * @param topK the routed experts per token of the result
     * @param calibration calibration inputs {@code [rows, features]}
     * @return the merged layer
     */
    public static FineGrainedMoe merge(FineGrainedMoe moe, int keep, int topK, Tensor calibration) {
        int e = moe.routedExperts().size();
        int[] load = ExpertPruning.usage(moe, calibration);
        int[] leaders = ExpertPruning.mostUsed(moe, keep, calibration);
        float[] logits = Var.noGrad(() -> Var.of(calibration).matmul(moe.router()).value()).toArray();
        int rows = calibration.shape().dim(0);
        int[] group = new int[e];
        for (int i = 0; i < e; i++) {
            int best = 0;
            double bestSim = Double.NEGATIVE_INFINITY;
            for (int g = 0; g < leaders.length; g++) {
                double sim = i == leaders[g] ? Double.POSITIVE_INFINITY : cosine(logits, rows, e, i, leaders[g]);
                if (sim > bestSim) {
                    bestSim = sim;
                    best = g;
                }
            }
            group[i] = best;
        }
        List<Layer> merged = new ArrayList<>();
        for (int g = 0; g < leaders.length; g++) {
            SiluMlpExpert lead = (SiluMlpExpert) moe.routedExperts().get(leaders[g]);
            double total = 0;
            double[][] acc = null;
            for (int i = 0; i < e; i++) {
                if (group[i] != g) {
                    continue;
                }
                SiluMlpExpert x = (SiluMlpExpert) moe.routedExperts().get(i);
                double[][] neurons = neurons(x);
                int[] perm = i == leaders[g] ? identity(neurons.length) : align(neurons(lead), neurons);
                double w = Math.max(1, load[i]);
                if (acc == null) {
                    acc = new double[neurons.length][neurons[0].length];
                }
                for (int h = 0; h < neurons.length; h++) {
                    for (int d = 0; d < neurons[0].length; d++) {
                        acc[h][d] += w * neurons[perm[h]][d];
                    }
                }
                total += w;
            }
            for (double[] row : acc) {
                for (int d = 0; d < row.length; d++) {
                    row[d] /= total;
                }
            }
            merged.add(fromNeurons(acc, lead));
        }
        return ExpertPruning.withRouter(moe.router().value(), leaders, merged, topK);
    }

    /**
     * All experts averaged into one dense expert, without alignment.
     *
     * @param moe the layer
     * @return the averaged expert
     */
    public static SiluMlpExpert averageAll(FineGrainedMoe moe) {
        List<Layer> experts = moe.routedExperts();
        double[][] acc = null;
        for (Layer l : experts) {
            double[][] n = neurons((SiluMlpExpert) l);
            if (acc == null) {
                acc = new double[n.length][n[0].length];
            }
            for (int h = 0; h < n.length; h++) {
                for (int d = 0; d < n[0].length; d++) {
                    acc[h][d] += n[h][d] / experts.size();
                }
            }
        }
        return fromNeurons(acc, (SiluMlpExpert) experts.getFirst());
    }

    /**
     * Matches the neurons of one expert to another's.
     *
     * @param reference the reference neurons {@code [hidden][dims]}
     * @param other the other expert's neurons
     * @return {@code perm}: reference neuron {@code h} is matched with {@code other[perm[h]]}
     */
    public static int[] align(double[][] reference, double[][] other) {
        int n = reference.length;
        double[][] cost = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                double dot = 0;
                for (int d = 0; d < reference[a].length; d++) {
                    dot += reference[a][d] * other[b][d];
                }
                cost[a][b] = -dot;
            }
        }
        return MinibatchOptimalTransport.assignCost(cost);
    }

    /** Each hidden neuron as one vector: its input weights, bias and output weights. */
    static double[][] neurons(SiluMlpExpert x) {
        float[] w1 = x.w1().value().toArray();
        float[] b1 = x.b1().value().toArray();
        float[] w2 = x.w2().value().toArray();
        int in = x.w1().shape().dim(0);
        int hidden = x.w1().shape().dim(1);
        int out = x.w2().shape().dim(1);
        double[][] n = new double[hidden][in + 1 + out];
        for (int h = 0; h < hidden; h++) {
            for (int i = 0; i < in; i++) {
                n[h][i] = w1[i * hidden + h];
            }
            n[h][in] = b1[h];
            for (int o = 0; o < out; o++) {
                n[h][in + 1 + o] = w2[h * out + o];
            }
        }
        return n;
    }

    private static SiluMlpExpert fromNeurons(double[][] n, SiluMlpExpert like) {
        int in = like.w1().shape().dim(0);
        int hidden = n.length;
        int out = like.w2().shape().dim(1);
        float[] w1 = new float[in * hidden];
        float[] b1 = new float[hidden];
        float[] w2 = new float[hidden * out];
        for (int h = 0; h < hidden; h++) {
            for (int i = 0; i < in; i++) {
                w1[i * hidden + h] = (float) n[h][i];
            }
            b1[h] = (float) n[h][in];
            for (int o = 0; o < out; o++) {
                w2[h * out + o] = (float) n[h][in + 1 + o];
            }
        }
        return new SiluMlpExpert(Tensor.of(Shape.of(in, hidden), w1), Tensor.of(like.b1().shape(), b1),
                Tensor.of(Shape.of(hidden, out), w2));
    }

    private static double cosine(float[] logits, int rows, int e, int a, int b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int r = 0; r < rows; r++) {
            double x = logits[r * e + a];
            double y = logits[r * e + b];
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        return dot / Math.sqrt(na * nb + 1e-30);
    }

    private static int[] identity(int n) {
        int[] p = new int[n];
        Arrays.setAll(p, i -> i);
        return p;
    }
}
