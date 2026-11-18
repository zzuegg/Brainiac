package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.graph.JumpingKnowledgeGcn;
import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Community detection (item 12.1.18). Unsupervised: Louvain, Bethe-Hessian (non-backtracking)
 * spectral clustering and SBM belief propagation against normalised spectral clustering of the
 * adjacency and asynchronous label propagation, on sparse two-block SBMs whose signal
 * {@code c_in - c_out} sweeps across the Kesten-Stigum threshold {@code k sqrt(c)}, scored by
 * NMI. Supervised: GCN stacks of depth 2 and 8 with and without residual skips and with
 * jumping-knowledge readouts ({@link JumpingKnowledgeGcn}) on a three-block SBM with weak
 * features and 10% labelled nodes, scored by accuracy on the rest.
 */
public final class CommunityDetection {

    private CommunityDetection() {
    }

    /**
     * NMI per method across a signal sweep.
     *
     * @param size nodes per block (two blocks)
     * @param degree the average degree {@code c}
     * @param gaps the values of {@code c_in - c_out}
     * @param reps graphs per gap
     * @param seed the seed
     * @return per method, the NMI at each gap (Louvain also reports its mean community count
     *     under the key {@code "LOUVAIN communities"})
     */
    public static Map<String, double[]> sweep(int size, double degree, double[] gaps, int reps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"adjacency spectral", "label propagation", "LOUVAIN", "SPECTRAL_NON_BACKTRACKING",
            "SBM_BP", "LOUVAIN communities"};
        for (String nm : names) {
            out.put(nm, new double[gaps.length]);
        }
        int n = 2 * size;
        for (int g = 0; g < gaps.length; g++) {
            double cIn = degree + gaps[g] / 2;
            double cOut = degree - gaps[g] / 2;
            for (int rep = 0; rep < reps; rep++) {
                long s = seed + 100L * g + rep;
                SyntheticGraphs.Graph graph = SyntheticGraphs.communities(2, size, cIn / n, cOut / n, 0, s);
                int[][] nb = io.github.zzuegg.brainiac.experimental.graph.CommunityDetection.neighbours(
                        graph.adjacency(), n);
                int[] truth = graph.nodeLabels();
                double[][] a = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j : nb[i]) {
                        a[i][j] = 1;
                    }
                }
                out.get("adjacency spectral")[g] += ClusteringMetrics.nmi(truth, SpectralClustering.cluster(a, 2, s))
                        / reps;
                out.get("label propagation")[g] += ClusteringMetrics.nmi(truth, labelPropagation(nb, s)) / reps;
                int[] louvain = io.github.zzuegg.brainiac.experimental.graph.CommunityDetection.detect(
                        io.github.zzuegg.brainiac.experimental.graph.CommunityDetection.Method.LOUVAIN, nb, 2, s);
                out.get("LOUVAIN")[g] += ClusteringMetrics.nmi(truth, louvain) / reps;
                out.get("LOUVAIN communities")[g] += (Arrays.stream(louvain).max().orElse(0) + 1.0) / reps;
                int[] bh = io.github.zzuegg.brainiac.experimental.graph.CommunityDetection.detect(
                        io.github.zzuegg.brainiac.experimental.graph.CommunityDetection.Method.SPECTRAL_NON_BACKTRACKING,
                        nb, 2, s);
                out.get("SPECTRAL_NON_BACKTRACKING")[g] += ClusteringMetrics.nmi(truth, bh) / reps;
                // SBM_BP refines the same Bethe-Hessian partition (what detect(SBM_BP, ...) does)
                int[] bp = io.github.zzuegg.brainiac.experimental.graph.CommunityDetection.beliefPropagation(nb, 2, bh,
                        s);
                out.get("SBM_BP")[g] += ClusteringMetrics.nmi(truth, bp) / reps;
            }
        }
        return out;
    }

    /** Asynchronous label propagation (Raghavan, Albert and Kumara 2007). */
    static int[] labelPropagation(int[][] nb, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = nb.length;
        int[] lab = new int[n];
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            lab[i] = i;
            order[i] = i;
        }
        int[] count = new int[n];
        for (int it = 0; it < 100; it++) {
            boolean changed = false;
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int i : order) {
                if (nb[i].length == 0) {
                    continue;
                }
                int best = -1;
                int bestCount = 0;
                int ties = 0;
                for (int j : nb[i]) {
                    count[lab[j]]++;
                }
                for (int j : nb[i]) {
                    int c = count[lab[j]];
                    if (c > bestCount) {
                        bestCount = c;
                        best = lab[j];
                        ties = 1;
                    } else if (c == bestCount && lab[j] != best && r.nextInt(++ties) == 0) {
                        best = lab[j];
                    }
                }
                for (int j : nb[i]) {
                    count[lab[j]] = 0;
                }
                if (count[lab[i]] == 0 && best != lab[i] && bestCount > 0) {
                    lab[i] = best;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
        int[] id = new int[n];
        Arrays.fill(id, -1);
        int next = 0;
        for (int i = 0; i < n; i++) {
            if (id[lab[i]] < 0) {
                id[lab[i]] = next++;
            }
            lab[i] = id[lab[i]];
        }
        return lab;
    }

    /**
     * Semi-supervised node classification with deep GCN stacks.
     *
     * @param depths the depths to try
     * @param steps Adam steps
     * @param seed the seed
     * @return test accuracy per "variant @ depth"
     */
    public static Map<String, Double> depth(int[] depths, int steps, long seed) {
        int k = 3;
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(k, 60, 0.12, 0.02, 1.5, seed);
        int n = g.nodes();
        boolean[] train = new boolean[n];
        float[] t = new float[n * k];
        int labelled = 0;
        for (int i = 0; i < n; i++) {
            train[i] = i % 10 == 0;
            if (train[i]) {
                t[i * k + g.nodeLabels()[i]] = 1;
                labelled++;
            }
        }
        Tensor target = Tensor.of(Shape.of(n, k), t);
        float inv = 1f / labelled;
        Var x = Var.of(g.featureTensor());
        Var adj = Var.of(g.adjacencyTensor());
        Map<String, Double> out = new LinkedHashMap<>();
        Object[][] variants = {{"plain", JumpingKnowledgeGcn.Readout.LAST, false},
            {"skip", JumpingKnowledgeGcn.Readout.LAST, true}, {"JK concat", JumpingKnowledgeGcn.Readout.CONCAT, false},
            {"JK max", JumpingKnowledgeGcn.Readout.MAX, false}};
        for (int depth : depths) {
            for (Object[] v : variants) {
                JumpingKnowledgeGcn net = new JumpingKnowledgeGcn(k, 16, k, depth, (JumpingKnowledgeGcn.Readout) v[1],
                        (Boolean) v[2], seed);
                Optimizer opt = Optimizer.adam(0.01f);
                for (int s = 0; s < steps; s++) {
                    net.forward(x, adj, true).logSoftmax(1).mul(target).sum().mul(-inv).backward();
                    opt.step(net.parameters());
                }
                float[] z = Var.noGrad(() -> net.forward(x, adj, false)).value().toArray();
                int ok = 0;
                int total = 0;
                for (int i = 0; i < n; i++) {
                    if (train[i]) {
                        continue;
                    }
                    int best = 0;
                    for (int c = 1; c < k; c++) {
                        if (z[i * k + c] > z[i * k + best]) {
                            best = c;
                        }
                    }
                    ok += best == g.nodeLabels()[i] ? 1 : 0;
                    total++;
                }
                out.put(v[0] + " @ " + depth, (double) ok / total);
            }
        }
        return out;
    }
}
