package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.nn.GatLayer;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.experimental.nn.GraphReadout;
import io.github.zzuegg.brainiac.experimental.nn.SageLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Graph neural networks on the three classic tasks, on synthetic graphs with known
 * structure.
 * <ul>
 *   <li>{@link #nodeClassification}: semi-supervised community detection on a stochastic
 *       block model with 10% labelled nodes, sweeping how noisy the node features are; a
 *       feature-only MLP collapses as noise grows while message passing (GCN, GAT,
 *       GraphSAGE) recovers labels from neighbours (Kipf and Welling,
 *       <a href="https://arxiv.org/abs/1609.02907">arXiv:1609.02907</a>);</li>
 *   <li>{@link #graphClassification}: trees versus cycles with identical node features,
 *       where GIN with a sum readout separates what mean aggregation cannot see (Xu et al.,
 *       <a href="https://arxiv.org/abs/1810.00826">arXiv:1810.00826</a>);</li>
 *   <li>{@link #linkPrediction}: a graph autoencoder (GCN encoder, inner-product decoder)
 *       predicting held-out edges, against the common-neighbours heuristic (Kipf and
 *       Welling, <a href="https://arxiv.org/abs/1611.07308">arXiv:1611.07308</a>).</li>
 * </ul>
 */
public final class GnnBenchmarks {

    private GnnBenchmarks() {
    }

    private static Var crossEntropy(Var logits, int[] labels, boolean[] mask, int classes) {
        int n = labels.length;
        float[] t = new float[n * classes];
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (mask[i]) {
                t[i * classes + labels[i]] = 1;
                count++;
            }
        }
        Var shifted = logits.sub(logits.value().max(1, true));
        Var logp = shifted.sub(shifted.exp().sum(1, true).log());
        return logp.mul(Tensor.of(Shape.of(n, classes), t)).sum().neg().div(Math.max(1, count));
    }

    private static double accuracy(Tensor logits, int[] labels, boolean[] mask, boolean want) {
        float[] l = logits.argmax(1).toArray();
        int right = 0;
        int total = 0;
        for (int i = 0; i < labels.length; i++) {
            if (mask[i] == want) {
                total++;
                right += Math.round(l[i]) == labels[i] ? 1 : 0;
            }
        }
        return right / (double) Math.max(1, total);
    }

    private static double trainNodes(MultiInputLayer a, MultiInputLayer b, SyntheticGraphs.Graph g, Tensor adj,
            boolean[] train, int classes, int steps) {
        Var x = Var.of(g.featureTensor());
        Var adjacency = Var.of(adj);
        List<Var> params = new ArrayList<>(a.parameters());
        params.addAll(b.parameters());
        Optimizer opt = Optimizer.adam(0.02f);
        for (int s = 0; s < steps; s++) {
            Var out = b.forward(List.of(a.forward(List.of(x, adjacency), true), adjacency), true);
            crossEntropy(out, g.nodeLabels(), train, classes).backward();
            opt.step(params);
        }
        Var out = b.forward(List.of(a.forward(List.of(x, adjacency), false), adjacency), false);
        return accuracy(out.value(), g.nodeLabels(), train, false);
    }

    /**
     * @param noises feature noise levels to sweep
     * @param steps training steps per model
     * @param seed the seed
     * @return "model @ noise" to test accuracy on unlabelled nodes
     */
    public static Map<String, Double> nodeClassification(double[] noises, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        int k = 3;
        for (double noise : noises) {
            SyntheticGraphs.Graph g = SyntheticGraphs.communities(k, 30, 0.25, 0.02, noise, seed);
            boolean[] train = new boolean[g.nodes()];
            for (int i = 0; i < train.length; i++) {
                train[i] = i % 10 == 0;
            }
            Tensor adj = g.adjacencyTensor();
            Tensor none = Tensor.zeros(Shape.of(g.nodes(), g.nodes()));
            SplittableRandom r = new SplittableRandom(seed);
            out.put("MLP @ " + noise, trainNodes(new SageLayer(k, 16, Activation.RELU, false, r),
                    new SageLayer(16, k, Activation.IDENTITY, false, r), g, none, train, k, steps));
            out.put("GCN @ " + noise, trainNodes(new GcnLayer(k, 16, Activation.RELU, r),
                    new GcnLayer(16, k, Activation.IDENTITY, r), g, adj, train, k, steps));
            out.put("GAT @ " + noise, trainNodes(new GatLayer(k, 4, 4, true, Activation.elu(), r),
                    new GatLayer(16, 1, k, false, Activation.IDENTITY, r), g, adj, train, k, steps));
            out.put("SAGE @ " + noise, trainNodes(new SageLayer(k, 16, Activation.RELU, false, r),
                    new SageLayer(16, k, Activation.IDENTITY, false, r), g, adj, train, k, steps));
        }
        return out;
    }

    /**
     * @param steps training steps
     * @param seed the seed
     * @return "layer + readout" to test accuracy
     */
    public static Map<String, Double> graphClassification(int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        List<SyntheticGraphs.Graph> train = SyntheticGraphs.treesAndCycles(60, seed);
        List<SyntheticGraphs.Graph> test = SyntheticGraphs.treesAndCycles(40, seed + 1);
        SyntheticGraphs.Batch tb = SyntheticGraphs.batch(train);
        SyntheticGraphs.Batch eb = SyntheticGraphs.batch(test);
        String[] names = {"GIN + sum", "GIN + mean", "SAGE + mean", "GCN + sum"};
        for (String name : names) {
            SplittableRandom r = new SplittableRandom(seed);
            MultiInputLayer l1;
            MultiInputLayer l2;
            if (name.startsWith("GIN")) {
                l1 = new GinLayer(1, 16, 16, r);
                l2 = new GinLayer(16, 16, 16, r);
            } else if (name.startsWith("SAGE")) {
                l1 = new SageLayer(1, 16, Activation.RELU, false, r);
                l2 = new SageLayer(16, 16, Activation.RELU, false, r);
            } else {
                l1 = new GcnLayer(1, 16, Activation.RELU, r);
                l2 = new GcnLayer(16, 16, Activation.RELU, r);
            }
            GraphReadout readout = new GraphReadout(name.endsWith("sum") ? GraphReadout.Mode.SUM
                    : GraphReadout.Mode.MEAN);
            Var head = Var.param(Tensor.normal(Shape.of(16, 2), 0, 0.3f, r));
            List<Var> params = new ArrayList<>(l1.parameters());
            params.addAll(l2.parameters());
            params.add(head);
            Optimizer opt = Optimizer.adam(0.01f);
            boolean[] all = new boolean[tb.labels().length];
            java.util.Arrays.fill(all, true);
            for (int s = 0; s < steps; s++) {
                Var h = l2.forward(List.of(l1.forward(List.of(Var.of(tb.features()), Var.of(tb.adjacency())), true),
                        Var.of(tb.adjacency())), true);
                crossEntropy(readout.forward(List.of(h, Var.of(tb.membership())), true).matmul(head), tb.labels(), all,
                        2).backward();
                opt.step(params);
            }
            Var h = l2.forward(List.of(l1.forward(List.of(Var.of(eb.features()), Var.of(eb.adjacency())), false),
                    Var.of(eb.adjacency())), false);
            Tensor logits = readout.forward(List.of(h, Var.of(eb.membership())), false).matmul(head).value();
            out.put(name, accuracy(logits, eb.labels(), new boolean[eb.labels().length], false));
        }
        return out;
    }

    /** Area under the ROC curve of scores for positives versus negatives. */
    static double auc(double[] positives, double[] negatives) {
        double wins = 0;
        for (double p : positives) {
            for (double n : negatives) {
                wins += p > n ? 1 : p == n ? 0.5 : 0;
            }
        }
        return wins / ((double) positives.length * negatives.length);
    }

    /**
     * @param steps training steps of the autoencoder
     * @param seed the seed
     * @return "GAE" and "common neighbours" to held-out link AUC
     */
    public static Map<String, Double> linkPrediction(int steps, long seed) {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 30, 0.3, 0.02, 1.0, seed);
        int n = g.nodes();
        SplittableRandom r = new SplittableRandom(seed);
        float[] trainAdj = g.adjacency().clone();
        List<int[]> heldOut = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (g.adjacency()[i * n + j] > 0 && r.nextDouble() < 0.15) {
                    heldOut.add(new int[] {i, j});
                    trainAdj[i * n + j] = 0;
                    trainAdj[j * n + i] = 0;
                }
            }
        }
        List<int[]> nonEdges = new ArrayList<>();
        while (nonEdges.size() < heldOut.size()) {
            int i = r.nextInt(n);
            int j = r.nextInt(n);
            if (i != j && g.adjacency()[i * n + j] == 0) {
                nonEdges.add(new int[] {i, j});
            }
        }
        Tensor adj = Tensor.of(Shape.of(n, n), trainAdj);
        GcnLayer l1 = new GcnLayer(3, 32, Activation.RELU, r);
        GcnLayer l2 = new GcnLayer(32, 16, Activation.IDENTITY, r);
        List<Var> params = new ArrayList<>(l1.parameters());
        params.addAll(l2.parameters());
        Optimizer opt = Optimizer.adam(0.01f);
        Var x = Var.of(g.featureTensor());
        // reconstruct the training adjacency: positives weighted up to balance the sparse matrix
        float density = 0;
        for (float a : trainAdj) {
            density += a;
        }
        density /= (float) n * n;
        float posWeight = (1 - density) / Math.max(density, 1e-6f);
        Tensor target = adj;
        Tensor weight = adj.mul(posWeight - 1).add(1f);
        for (int s = 0; s < steps; s++) {
            Var z = l2.forward(List.of(l1.forward(List.of(x, Var.of(adj)), true), Var.of(adj)), true);
            Var logits = z.matmul(z.transpose());
            Var loss = logits.softplus().sub(logits.mul(target)).mul(weight).mean();
            loss.backward();
            opt.step(params);
        }
        float[] z = l2.forward(List.of(l1.forward(List.of(x, Var.of(adj)), false), Var.of(adj)), false).value()
                .toArray();
        int d = 16;
        java.util.function.ToDoubleBiFunction<Integer, Integer> gae = (i, j) -> {
            double s = 0;
            for (int k = 0; k < d; k++) {
                s += z[i * d + k] * z[j * d + k];
            }
            return s;
        };
        java.util.function.ToDoubleBiFunction<Integer, Integer> common = (i, j) -> {
            double c = 0;
            for (int k = 0; k < n; k++) {
                c += trainAdj[i * n + k] * trainAdj[j * n + k];
            }
            return c;
        };
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("GAE", auc(heldOut.stream().mapToDouble(e -> gae.applyAsDouble(e[0], e[1])).toArray(),
                nonEdges.stream().mapToDouble(e -> gae.applyAsDouble(e[0], e[1])).toArray()));
        out.put("common neighbours", auc(heldOut.stream().mapToDouble(e -> common.applyAsDouble(e[0], e[1])).toArray(),
                nonEdges.stream().mapToDouble(e -> common.applyAsDouble(e[0], e[1])).toArray()));
        return out;
    }
}
