package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.nn.GatLayer;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.experimental.nn.SageLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** The shared training harness of the wave 7 graph experiments (package-private). */
final class GraphTraining {

    private GraphTraining() {
    }

    /** A node classifier. */
    interface NodeModel {
        Var forward(Var x, Tensor adjacency, boolean training);

        List<Var> parameters();
    }

    /** A graph-level model on a block-diagonal batch: {@code [graphs, outputs]}. */
    interface GraphModel {
        Var forward(SyntheticGraphs.Batch batch, boolean training);

        List<Var> parameters();
    }

    static Var crossEntropy(Var logits, int[] labels, boolean[] mask, int classes) {
        int n = labels.length;
        float[] t = new float[n * classes];
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (mask == null || mask[i]) {
                t[i * classes + labels[i]] = 1;
                count++;
            }
        }
        Var shifted = logits.sub(logits.value().max(1, true));
        Var logp = shifted.sub(shifted.exp().sum(1, true).log());
        return logp.mul(Tensor.of(Shape.of(n, classes), t)).sum().neg().div(Math.max(1, count));
    }

    static double accuracy(Tensor logits, int[] labels, boolean[] mask, boolean want) {
        float[] l = logits.argmax(1).toArray();
        int right = 0;
        int total = 0;
        for (int i = 0; i < labels.length; i++) {
            if (mask == null || mask[i] == want) {
                total++;
                right += Math.round(l[i]) == labels[i] ? 1 : 0;
            }
        }
        return right / (double) Math.max(1, total);
    }

    /** A stratified training mask: about {@code rate} of the nodes, at least one per class. */
    static boolean[] labelMask(int[] labels, int classes, double rate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[] m = new boolean[labels.length];
        int[] per = new int[classes];
        for (int i = 0; i < labels.length; i++) {
            if (r.nextDouble() < rate) {
                m[i] = true;
                per[labels[i]]++;
            }
        }
        for (int c = 0; c < classes; c++) {
            for (int i = 0; i < labels.length && per[c] == 0; i++) {
                if (labels[i] == c) {
                    m[i] = true;
                    per[c]++;
                }
            }
        }
        return m;
    }

    /** Trains with Adam on the masked nodes; returns accuracy on the unmasked ones. */
    static double trainNodes(NodeModel model, SyntheticGraphs.Graph g, Tensor adjacency, boolean[] train, int classes,
            int steps, float lr) {
        Var x = Var.of(g.featureTensor());
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            crossEntropy(model.forward(x, adjacency, true), g.nodeLabels(), train, classes).backward();
            opt.step(model.parameters());
        }
        Tensor out = Var.noGrad(() -> model.forward(x, adjacency, false)).value();
        return accuracy(out, g.nodeLabels(), train, false);
    }

    /** Trains a graph classifier; returns test accuracy. */
    static double trainGraphs(GraphModel model, SyntheticGraphs.Batch train, SyntheticGraphs.Batch test, int classes,
            int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            crossEntropy(model.forward(train, true), train.labels(), null, classes).backward();
            opt.step(model.parameters());
        }
        Tensor out = Var.noGrad(() -> model.forward(test, false)).value();
        return accuracy(out, test.labels(), null, true);
    }

    /** A two-layer stack of multi-input graph layers as a node model. */
    static NodeModel stack(MultiInputLayer a, MultiInputLayer b) {
        return new NodeModel() {
            @Override
            public Var forward(Var x, Tensor adjacency, boolean training) {
                Var adj = Var.of(adjacency);
                return b.forward(List.of(a.forward(List.of(x, adj), training), adj), training);
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(a.parameters());
                p.addAll(b.parameters());
                return p;
            }
        };
    }

    /** One multi-input layer as a node model. */
    static NodeModel single(MultiInputLayer a) {
        return new NodeModel() {
            @Override
            public Var forward(Var x, Tensor adjacency, boolean training) {
                return a.forward(List.of(x, Var.of(adjacency)), training);
            }

            @Override
            public List<Var> parameters() {
                return a.parameters();
            }
        };
    }

    static NodeModel gcn(int in, int hidden, int out, SplittableRandom r) {
        return stack(new GcnLayer(in, hidden, Activation.RELU, r), new GcnLayer(hidden, out, Activation.IDENTITY, r));
    }

    static NodeModel gat(int in, int out, SplittableRandom r) {
        return stack(new GatLayer(in, 4, 4, true, Activation.elu(), r),
                new GatLayer(16, 1, out, false, Activation.IDENTITY, r));
    }

    static NodeModel sage(int in, int hidden, int out, SplittableRandom r) {
        return stack(new SageLayer(in, hidden, Activation.RELU, false, r),
                new SageLayer(hidden, out, Activation.IDENTITY, false, r));
    }

    /** A feature-only MLP: GraphSAGE on an empty graph, as in {@link GnnBenchmarks}. */
    static NodeModel mlp(int in, int hidden, int out, SplittableRandom r) {
        NodeModel inner = sage(in, hidden, out, r);
        return new NodeModel() {
            @Override
            public Var forward(Var x, Tensor adjacency, boolean training) {
                int n = adjacency.shape().dim(0);
                return inner.forward(x, Tensor.zeros(Shape.of(n, n)), training);
            }

            @Override
            public List<Var> parameters() {
                return inner.parameters();
            }
        };
    }

    /** Softmax regression on fixed embeddings; returns accuracy on the unlabelled nodes. */
    static double probe(float[][] emb, int[] labels, boolean[] train, int classes, long seed) {
        int n = emb.length;
        int d = emb[0].length;
        float[] flat = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(emb[i], 0, flat, i * d, d);
        }
        Var x = Var.of(Tensor.of(Shape.of(n, d), flat));
        SplittableRandom r = new SplittableRandom(seed);
        Var w = Var.param(Tensor.normal(Shape.of(d, classes), 0, 0.1f, r));
        Var b = Var.param(Tensor.zeros(Shape.of(1, classes)));
        Optimizer opt = Optimizer.adam(0.05f);
        for (int s = 0; s < 150; s++) {
            GraphTraining.crossEntropy(x.matmul(w).add(b), labels, train, classes).backward();
            opt.step(List.of(w, b));
        }
        return GraphTraining.accuracy(x.matmul(w).add(b).value(), labels, train, false);
    }

    static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    /** The mean over seeds of a seeded measurement. */
    static double overSeeds(int seeds, long seed, java.util.function.LongToDoubleFunction run) {
        double[] v = new double[seeds];
        for (int s = 0; s < seeds; s++) {
            v[s] = run.applyAsDouble(seed + 1000L * s);
        }
        return mean(v);
    }

    static double round(double v) {
        return Math.round(v * 1000) / 1000.0;
    }
}
