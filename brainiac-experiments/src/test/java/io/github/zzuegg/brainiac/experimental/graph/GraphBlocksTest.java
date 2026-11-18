package io.github.zzuegg.brainiac.experimental.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.AlgorithmTraces;
import io.github.zzuegg.brainiac.experimental.data.FamilyTreeGraph;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

class GraphBlocksTest {

    /** A triangle 0-1-2 with a tail 2-3. */
    private static Tensor triangleWithTail() {
        float[] a = new float[16];
        int[][] edges = {{0, 1}, {1, 2}, {0, 2}, {2, 3}};
        for (int[] e : edges) {
            a[e[0] * 4 + e[1]] = 1;
            a[e[1] * 4 + e[0]] = 1;
        }
        return Tensor.of(Shape.of(4, 4), a);
    }

    @Test
    void randomWalkEncodingSeesTriangles() {
        float[] rw = GraphPositionalEncodings.randomWalk(triangleWithTail(), 3).toArray();
        // 3-step return probability: positive on triangle nodes, zero on the tail node
        assertTrue(rw[0 * 3 + 2] > 0 && rw[1 * 3 + 2] > 0);
        assertEquals(0f, rw[3 * 3 + 2], 1e-6f);
        assertEquals(0f, rw[0], 1e-6f);   // no self loops: 1-step return is 0
    }

    @Test
    void laplacianEncodingIsPerGraphAndOrthogonal() {
        SyntheticGraphs.Batch b = SyntheticGraphs.batch(SyntheticGraphs.treesAndCycles(2, 1));
        Tensor pe = GraphPositionalEncodings.laplacian(b.adjacency(), b.membership(), 3);
        int n = b.adjacency().shape().dim(0);
        float[] v = pe.toArray();
        float[] m = b.membership().toArray();
        double cross = 0;
        for (int i = 0; i < n; i++) {
            if (m[i] > 0) {
                cross += v[i * 3] * v[i * 3 + 1];
            }
        }
        assertEquals(0, cross, 1e-3);
        Tensor flipped = GraphPositionalEncodings.randomSignFlip(pe, new SplittableRandom(3));
        assertEquals(pe.abs().sum().item(), flipped.abs().sum().item(), 1e-4);
    }

    @Test
    void gpsLayerKeepsShapeAndMasksAttention() {
        SplittableRandom r = new SplittableRandom(1);
        GraphGpsLayer layer = new GraphGpsLayer(8, 2, true, r);
        Var x = Var.of(Tensor.normal(Shape.of(4, 8), 0, 1, r));
        Var out = layer.forward(List.of(x, Var.of(triangleWithTail())), false);
        assertEquals(Shape.of(4, 8), out.shape());
        // with a block mask, node 3 (alone in its block, no edges) is unaffected by the others
        float[] mask = {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1};
        Tensor empty = Tensor.zeros(Shape.of(4, 4));
        Var a1 = layer.attend(x, Tensor.of(Shape.of(4, 4), mask));
        Var x2 = Var.of(x.value().add(Tensor.of(Shape.of(4, 1), 5, 5, 5, 0)));
        Var a2 = layer.attend(x2, Tensor.of(Shape.of(4, 4), mask));
        assertEquals(a1.value().slice(0, 3, 4).sum().item(), a2.value().slice(0, 3, 4).sum().item(), 1e-4);
        assertNotNull(layer.forward(List.of(x, Var.of(empty)), false));
    }

    @Test
    void sheafDiffusionGradientsMatchFiniteDifferences() {
        SplittableRandom r = new SplittableRandom(2);
        Tensor adj = triangleWithTail();
        for (SheafDiffusionLayer.Maps maps : SheafDiffusionLayer.Maps.values()) {
            SheafDiffusionLayer layer = new SheafDiffusionLayer(2, 2, maps, r);
            Tensor x = Tensor.normal(Shape.of(4, 4), 0, 1, r);
            GradCheck.Result g = GradCheck.check(v -> layer.forward(List.of(v.get(0), Var.of(adj)), true).square().sum(),
                    List.of(x), 1e-3f, 3e-2f);
            assertTrue(g.passed(), maps + " " + g);
        }
    }

    @Test
    void orthogonalRestrictionMapsAreOrthogonal() {
        SplittableRandom r = new SplittableRandom(4);
        SheafDiffusionLayer layer = new SheafDiffusionLayer(3, 2, SheafDiffusionLayer.Maps.ORTHOGONAL, r);
        Var q = layer.restrictionMaps(Var.of(Tensor.normal(Shape.of(5, 12), 0, 1, r)));
        Tensor qtq = q.value().transpose().matmul(q.value());
        Tensor eye = Linalg.identity(3, qtq.dtype(), qtq.device()).reshape(1, 3, 3);
        assertTrue(qtq.sub(eye).abs().max(-1, false).max(-1, false).max(0, false).item() < 1e-4);
    }

    @Test
    void pairNormKeepsTotalPairwiseScale() {
        SplittableRandom r = new SplittableRandom(5);
        Var x = Var.of(Tensor.normal(Shape.of(10, 4), 3, 0.5f, r));
        Tensor y = new PairNorm(1f).forward(x).value();
        assertEquals(0, y.mean(0, false).abs().max(0, false).item(), 1e-4);
        assertEquals(1, y.square().sum(1, false).mean().item(), 1e-3);
    }

    @Test
    void dropEdgeIsSymmetricAndOffAtEvaluation() {
        Tensor adj = SyntheticGraphs.communities(2, 10, 0.5, 0.1, 0, 1).adjacencyTensor();
        DropEdge d = new DropEdge(0.5, new SplittableRandom(1));
        Tensor dropped = d.apply(adj, true);
        assertEquals(0, dropped.sub(dropped.transpose()).abs().sum().item(), 0);
        assertTrue(dropped.sum().item() < adj.sum().item());
        assertTrue(d.apply(adj, false) == adj);
    }

    @Test
    void gcn2BetaShrinksWithDepth() {
        SplittableRandom r = new SplittableRandom(6);
        assertTrue(new Gcn2Layer(4, 0.1f, 0.5f, 1, r).beta() > new Gcn2Layer(4, 0.1f, 0.5f, 32, r).beta());
    }

    @Test
    void rewiringAddsRelaysAndWidensTheSpectralGap() {
        SyntheticGraphs.Batch b = SyntheticGraphs.batch(SyntheticGraphs.neighboursMatch(2, 2, 1));
        GraphRewiring.Rewired v = GraphRewiring.virtualNode(b.adjacency(), b.features(), b.membership());
        int n = b.adjacency().shape().dim(0);
        assertEquals(n + 2, v.adjacency().shape().dim(0));
        assertEquals(b.adjacency().sum().item() + 2 * n, v.adjacency().sum().item(), 1e-3);
        Tensor tree = SyntheticGraphs.neighboursMatch(3, 1, 2).getFirst().adjacencyTensor();
        Tensor rewired = GraphRewiring.fosr(tree, null, 4);
        assertEquals(tree.sum().item() + 8, rewired.sum().item(), 1e-3);
        assertTrue(spectralGap(rewired) > spectralGap(tree));
    }

    private static double spectralGap(Tensor adj) {
        int n = adj.shape().dim(0);
        float[] a = adj.toArray();
        float[] lap = new float[n * n];
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i] += a[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            lap[i * n + i] = 1;
            for (int j = 0; j < n; j++) {
                if (a[i * n + j] != 0) {
                    lap[i * n + j] = (float) (-1 / Math.sqrt(d[i] * d[j]));
                }
            }
        }
        return Linalg.eigh(Tensor.of(Shape.of(n, n), lap))[0].toArray()[1];
    }

    @Test
    void randomNodeFeaturesAreFreshEachCall() {
        RandomNodeFeatures rnf = new RandomNodeFeatures(3, new SplittableRandom(1));
        Var x = Var.of(Tensor.ones(Shape.of(5, 2)));
        Tensor a = rnf.forward(x).value();
        Tensor b = rnf.forward(x).value();
        assertEquals(Shape.of(5, 5), a.shape());
        assertTrue(a.sub(b).abs().sum().item() > 0);
    }

    @Test
    void subgraphEnsembleSeparatesSkipLinks() {
        SyntheticGraphs.Batch b = SyntheticGraphs.batch(SyntheticGraphs.circularSkipLinks(11, new int[] {2, 3}, 1, 1));
        SubgraphEnsembleGnn esan = new SubgraphEnsembleGnn(1, 8, 4, 3, SubgraphEnsembleGnn.Policy.NODE_DELETED, 2,
                new SplittableRandom(1));
        Tensor out = esan.forward(List.of(Var.of(b.features()), Var.of(b.adjacency()), Var.of(b.membership())), false)
                .value();
        assertEquals(Shape.of(2, 4), out.shape());
        // untrained, the two 4-regular graphs already embed differently
        assertTrue(out.slice(0, 0, 1).sub(out.slice(0, 1, 2)).abs().sum().item() > 1e-4);
    }

    @Test
    void pnaGradientsMatchFiniteDifferences() {
        SplittableRandom r = new SplittableRandom(8);
        PnaLayer pna = new PnaLayer(3, 4, 1f, r);
        Tensor adj = triangleWithTail();
        GradCheck.Result g = GradCheck.check(v -> pna.forward(List.of(v.get(0), Var.of(adj)), true).square().sum(),
                List.of(Tensor.normal(Shape.of(4, 3), 0, 1, r)), 1e-3f, 3e-2f);
        assertTrue(g.passed(), g.toString());
    }

    @Test
    void poolingCoarsensEveryGraph() {
        SyntheticGraphs.Batch b = SyntheticGraphs.batch(SyntheticGraphs.treesAndCycles(4, 2));
        SplittableRandom r = new SplittableRandom(9);
        Var x = Var.of(Tensor.normal(Shape.of(b.adjacency().shape().dim(0), 6), 0, 1, r));
        DiffPool dp = new DiffPool(6, 3, r);
        DiffPool.Pooled p = dp.pool(x, b.adjacency(), b.membership());
        assertEquals(Shape.of(4, 3, 6), p.features().shape());
        assertEquals(Shape.of(4, 3, 3), p.adjacency().shape());
        assertTrue(p.auxiliaryLoss().value().item() > 0);
        for (TopKPooling.Score s : TopKPooling.Score.values()) {
            TopKPooling.Pooled t = new TopKPooling(6, 0.5, s, r).pool(x, b.adjacency(), b.membership());
            assertEquals(4, t.membership().shape().dim(0));
            assertEquals(t.kept().length, t.features().shape().dim(0));
            float[] m = t.membership().sum(1, false).toArray();
            for (float k : m) {
                assertTrue(k >= 3);
            }
        }
    }

    @Test
    void walksFollowEdges() {
        Tensor adj = SyntheticGraphs.communities(2, 8, 0.6, 0.1, 0, 3).adjacencyTensor();
        float[] a = adj.toArray();
        for (int[] w : RandomWalkEmbedding.walks(adj, 2, 10, 0.5, 2, 1)) {
            for (int t = 1; t < w.length; t++) {
                assertEquals(1f, a[w[t - 1] * 16 + w[t]]);
            }
        }
        for (int[] w : RandomWalkSequenceLayer.sampleWalks(adj, 6, new SplittableRandom(2))) {
            for (int t = 1; t < w.length; t++) {
                assertTrue(a[w[t - 1] * 16 + w[t]] == 1 || w[t] == w[t - 1]);
            }
        }
    }

    @Test
    void knowledgeGraphScoresAreBoundedAndLossFinite() {
        KnowledgeGraphEmbedding kg = new KnowledgeGraphEmbedding(KnowledgeGraphEmbedding.Model.ROTATE, 5, 2, 4, 6f,
                new SplittableRandom(1));
        float[] s = kg.scoreTails(0, 1);
        assertEquals(5, s.length);
        for (float v : s) {
            assertTrue(v <= 6f);
        }
        FamilyTreeGraph.Data d = FamilyTreeGraph.generate(3, 0.3, 1);
        Var loss = kg.loss(d.train().subList(0, 4).stream().map(t -> new int[] {t[0] % 5, t[1] % 2, t[2] % 5})
                .toList(), 3, 1f, new SplittableRandom(2));
        assertTrue(Float.isFinite(loss.value().item()));
    }

    @Test
    void familyTreeTestTriplesFollowFromTraining() {
        FamilyTreeGraph.Data d = FamilyTreeGraph.generate(6, 0.3, 4);
        assertTrue(d.test().size() > 10);
        java.util.Set<Long> train = new java.util.HashSet<>();
        for (int[] t : d.train()) {
            train.add(KnowledgeGraphEmbedding.key(t[0], t[1], t[2]));
        }
        for (int[] t : d.test()) {
            if (t[1] == FamilyTreeGraph.SIBLING) {
                assertTrue(train.contains(KnowledgeGraphEmbedding.key(t[2], FamilyTreeGraph.SIBLING, t[0])));
            }
            if (t[1] == FamilyTreeGraph.CHILD) {
                assertTrue(train.contains(KnowledgeGraphEmbedding.key(t[2], FamilyTreeGraph.PARENT, t[0])));
            }
        }
    }

    @Test
    void tracesAreCorrect() {
        AlgorithmTraces.Trace bf = AlgorithmTraces.generate(AlgorithmTraces.Algorithm.BELLMAN_FORD, 10, 3);
        int[] pi = bf.output();
        assertEquals(bf.source(), pi[bf.source()]);
        for (int v = 0; v < 10; v++) {
            assertTrue(v == bf.source() || bf.weights()[v * 10 + pi[v]] > 0);
        }
        AlgorithmTraces.Trace prim = AlgorithmTraces.generate(AlgorithmTraces.Algorithm.PRIM, 10, 3);
        assertEquals(10, prim.steps());
        int edges = 0;
        for (int v = 0; v < 10; v++) {
            edges += prim.output()[v] != v ? 1 : 0;
        }
        assertEquals(9, edges);
        AlgorithmTraces.Trace bfs = AlgorithmTraces.generate(AlgorithmTraces.Algorithm.BFS, 12, 5);
        assertEquals(1, bfs.hints()[0][bfs.source()]);
    }

    @Test
    void srPairsDifferInSatisfiability() {
        SplittableRandom r = new SplittableRandom(1);
        RandomSatFormulas.Formula[] pair = RandomSatFormulas.srPair(8, r);
        boolean[] a = RandomSatFormulas.solve(8, pair[0].clauses());
        assertNotNull(a);
        assertTrue(RandomSatFormulas.satisfies(pair[0].clauses(), a));
        assertNull(RandomSatFormulas.solve(8, pair[1].clauses()));
        NeuroSat ns = new NeuroSat(8, r);
        NeuroSat.Batch b = NeuroSat.batch(List.of(pair));
        assertEquals(Shape.of(2), ns.logits(b, 3).shape());
    }

    @Test
    void processorRunsAndDecodes() {
        AlgorithmicProcessor p = new AlgorithmicProcessor(1, 8, 2, AlgorithmicProcessor.Aggregation.MAX,
                new SplittableRandom(1));
        AlgorithmTraces.Trace t = AlgorithmTraces.generate(AlgorithmTraces.Algorithm.PRIM, 6, 1);
        Tensor w = Tensor.of(Shape.of(6, 6), t.weights());
        List<Var> states = p.run(Var.of(Tensor.ones(Shape.of(6, 1))), w, 3);
        assertEquals(3, states.size());
        assertEquals(Shape.of(6, 6), p.pointerLogits(states.getLast(), w).shape());
    }

    @Test
    void labelPropagationRecoversCommunities() {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 20, 0.3, 0.02, 0, 1);
        boolean[] train = new boolean[60];
        for (int i = 0; i < 60; i += 10) {
            train[i] = true;
        }
        float[] lp = CorrectAndSmooth.labelPropagation(g.adjacencyTensor(), g.nodeLabels(), train, 3, 0.9f, 30)
                .argmax(1).toArray();
        int right = 0;
        for (int i = 0; i < 60; i++) {
            right += Math.round(lp[i]) == g.nodeLabels()[i] ? 1 : 0;
        }
        assertTrue(right > 50, "LP correct: " + right);
    }
}
