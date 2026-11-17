package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

class GraphLayersTest {

    /** Masked cross-entropy over the training nodes. */
    private static Var loss(Var logits, int[] labels, boolean[] mask, int classes) {
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
        return logp.mul(Tensor.of(Shape.of(n, classes), t)).sum().neg().div(count);
    }

    private static double accuracy(Var logits, int[] labels, boolean[] mask, boolean want) {
        float[] l = logits.value().argmax(1).toArray();
        int right = 0;
        int total = 0;
        for (int i = 0; i < labels.length; i++) {
            if (mask[i] == want) {
                total++;
                right += Math.round(l[i]) == labels[i] ? 1 : 0;
            }
        }
        return right / (double) total;
    }

    /** Two graph layers, trained on 10% of the nodes, tested on the rest. */
    static double nodeClassification(String kind) {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 30, 0.25, 0.02, 1.2, 7);
        SplittableRandom r = new SplittableRandom(3);
        boolean[] train = new boolean[g.nodes()];
        for (int i = 0; i < train.length; i++) {
            train[i] = i % 10 == 0;
        }
        MultiInputLayer a;
        MultiInputLayer b;
        switch (kind) {
            case "gcn" -> {
                a = new GcnLayer(3, 16, Activation.RELU, r);
                b = new GcnLayer(16, 3, Activation.IDENTITY, r);
            }
            case "gat" -> {
                a = new GatLayer(3, 4, 4, true, Activation.elu(), r);
                b = new GatLayer(16, 1, 3, false, Activation.IDENTITY, r);
            }
            case "sage" -> {
                a = new SageLayer(3, 16, Activation.RELU, false, r);
                b = new SageLayer(16, 3, Activation.IDENTITY, false, r);
            }
            case "mlp" -> {
                // no edges: both layers see only the node itself
                a = new SageLayer(3, 16, Activation.RELU, false, r);
                b = new SageLayer(16, 3, Activation.IDENTITY, false, r);
            }
            default -> throw new IllegalArgumentException(kind);
        }
        Var x = Var.of(g.featureTensor());
        Var adj = Var.of(kind.equals("mlp") ? Tensor.zeros(Shape.of(g.nodes(), g.nodes())) : g.adjacencyTensor());
        List<Var> params = new ArrayList<>(a.parameters());
        params.addAll(b.parameters());
        Optimizer opt = Optimizer.adam(0.02f);
        for (int step = 0; step < 100; step++) {
            Var out = b.forward(List.of(a.forward(List.of(x, adj), true), adj), true);
            loss(out, g.nodeLabels(), train, 3).backward();
            opt.step(params);
        }
        Var out = b.forward(List.of(a.forward(List.of(x, adj), false), adj), false);
        return accuracy(out, g.nodeLabels(), train, false);
    }

    @Test
    void graphLayersBeatAFeatureOnlyBaselineOnCommunities() {
        double mlp = nodeClassification("mlp");
        double gcn = nodeClassification("gcn");
        double gat = nodeClassification("gat");
        double sage = nodeClassification("sage");
        System.out.printf("node classification (test acc): mlp %.3f gcn %.3f gat %.3f sage %.3f%n", mlp, gcn, gat,
                sage);
        assertTrue(gcn > mlp + 0.15, "gcn " + gcn + " mlp " + mlp);
        assertTrue(gat > mlp + 0.15, "gat " + gat);
        assertTrue(sage > mlp + 0.1, "sage " + sage);
    }

    @Test
    void gatAttendsOnlyToNeighbours() {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(2, 5, 0.5, 0.1, 0.5, 1);
        GatLayer gat = new GatLayer(2, 2, 3, true, Activation.IDENTITY, new SplittableRandom(1));
        float[] w = gat.attention(g.featureTensor(), g.adjacencyTensor()).toArray();
        int n = g.nodes();
        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    double v = w[(h * n + i) * n + j];
                    sum += v;
                    if (i != j && g.adjacency()[i * n + j] == 0) {
                        assertEquals(0, v, 1e-6);
                    }
                }
                assertEquals(1, sum, 1e-4);
            }
        }
    }

    @Test
    void graphLayerGradientsMatchFiniteDifferences() {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(2, 3, 0.8, 0.2, 0.5, 2);
        Tensor adj = g.adjacencyTensor();
        List<MultiInputLayer> layers = List.of(new GcnLayer(2, 3, Activation.TANH, new SplittableRandom(1)),
                new GatLayer(2, 2, 2, true, Activation.TANH, new SplittableRandom(2)),
                new SageLayer(2, 3, Activation.TANH, true, new SplittableRandom(3)),
                new GinLayer(2, 4, 3, new SplittableRandom(4)));
        for (MultiInputLayer layer : layers) {
            GradCheck.Result res = GradCheck.check(xs -> layer.forward(List.of(xs.get(0), Var.of(adj)), true)
                    .square().sum(), List.of(g.featureTensor()));
            assertTrue(res.passed(), layer + ": " + res);
        }
    }

    /** GIN with a sum readout versus mean-aggregating GraphSAGE with a mean readout on trees vs cycles. */
    static double graphClassification(boolean gin) {
        List<SyntheticGraphs.Graph> train = SyntheticGraphs.treesAndCycles(60, 1);
        List<SyntheticGraphs.Graph> test = SyntheticGraphs.treesAndCycles(40, 2);
        SplittableRandom r = new SplittableRandom(5);
        MultiInputLayer l1 = gin ? new GinLayer(1, 16, 16, r) : new SageLayer(1, 16, Activation.RELU, false, r);
        MultiInputLayer l2 = gin ? new GinLayer(16, 16, 16, r) : new SageLayer(16, 16, Activation.RELU, false, r);
        GraphReadout readout = new GraphReadout(gin ? GraphReadout.Mode.SUM : GraphReadout.Mode.MEAN);
        Var head = Var.param(Tensor.normal(Shape.of(16, 2), 0, 0.3f, r));
        List<Var> params = new ArrayList<>(l1.parameters());
        params.addAll(l2.parameters());
        params.add(head);
        Optimizer opt = Optimizer.adam(0.01f);
        SyntheticGraphs.Batch b = SyntheticGraphs.batch(train);
        for (int step = 0; step < 300; step++) {
            Var h = l2.forward(List.of(l1.forward(List.of(Var.of(b.features()), Var.of(b.adjacency())), true),
                    Var.of(b.adjacency())), true);
            Var logits = readout.forward(List.of(h, Var.of(b.membership())), true).matmul(head);
            boolean[] all = new boolean[b.labels().length];
            java.util.Arrays.fill(all, true);
            loss(logits, b.labels(), all, 2).backward();
            opt.step(params);
        }
        SyntheticGraphs.Batch t = SyntheticGraphs.batch(test);
        Var h = l2.forward(List.of(l1.forward(List.of(Var.of(t.features()), Var.of(t.adjacency())), false),
                Var.of(t.adjacency())), false);
        Var logits = readout.forward(List.of(h, Var.of(t.membership())), false).matmul(head);
        boolean[] none = new boolean[t.labels().length];
        return accuracy(logits, t.labels(), none, false);
    }

    @Test
    void ginWithSumReadoutTellsTreesFromCycles() {
        double gin = graphClassification(true);
        double sageMean = graphClassification(false);
        System.out.printf("graph classification: GIN+sum %.3f, SAGE+mean %.3f%n", gin, sageMean);
        assertTrue(gin >= 0.85, "GIN " + gin);
        assertTrue(gin > sageMean + 0.2, "GIN " + gin + " SAGE " + sageMean);
    }

    @Test
    void readoutModesPoolPerGraph() {
        Var h = Var.of(Tensor.of(Shape.of(3, 1), 1, 2, 5));
        Var m = Var.of(Tensor.of(Shape.of(2, 3), 1, 1, 0, 0, 0, 1));
        assertEquals(3f, new GraphReadout(GraphReadout.Mode.SUM).forward(List.of(h, m), false).value().toArray()[0]);
        assertEquals(1.5f, new GraphReadout(GraphReadout.Mode.MEAN).forward(List.of(h, m), false).value().toArray()[0]);
        float[] max = new GraphReadout(GraphReadout.Mode.MAX).forward(List.of(h, m), false).value().toArray();
        assertEquals(2f, max[0]);
        assertEquals(5f, max[1]);
    }
}
