package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/** Wave 7 items 7.1.2, 7.1.3 and 7.1.12. */
class NodeShootoutTest {

    @Test
    void sgcMatchesGcnAndAppnpBeatsMlpAtFivePercentLabels() {
        Map<String, Double> r = GraphLearningShootout.decoupledPropagation(0.05, 2, 100, 11);
        System.out.printf("7.1.2 @5%% labels: SGC %.3f (%.0f ms), APPNP %.3f; GCN %.3f (%.0f ms), GAT %.3f, MLP %.3f%n",
                r.get("SGC"), r.get("SGC ms"), r.get("APPNP"), r.get("GCN"), r.get("GCN ms"), r.get("GAT"),
                r.get("MLP"));
        assertTrue(r.get("SGC") > r.get("GCN") - 0.06, r.toString());
        assertTrue(r.get("APPNP") > r.get("MLP") + 0.1, r.toString());
    }

    @Test
    void gcniiAndPairNormSurviveDepth32() {
        Map<String, Double> r = GraphLearningShootout.depthSweep(
                List.of("GCN", "GCN + residual", "GCN + PairNorm", "GCN + DropEdge", "GCNII"), new int[] {2, 32}, 80, 5);
        System.out.printf("7.1.3 depth 2 -> 32: GCN %.3f -> %.3f (dist %.3f -> %.3f), residual -> %.3f, PairNorm -> %.3f,"
                + " DropEdge -> %.3f, GCNII %.3f -> %.3f (dist %.3f)%n", r.get("GCN @ 2"), r.get("GCN @ 32"),
                r.get("GCN @ 2 dist"), r.get("GCN @ 32 dist"), r.get("GCN + residual @ 32"),
                r.get("GCN + PairNorm @ 32"), r.get("GCN + DropEdge @ 32"), r.get("GCNII @ 2"), r.get("GCNII @ 32"),
                r.get("GCNII @ 32 dist"));
        assertTrue(r.get("GCNII @ 32") > r.get("GCN @ 32") + 0.3, r.toString());
        assertTrue(r.get("GCN + PairNorm @ 32") > r.get("GCN @ 32") + 0.3, r.toString());
        assertTrue(r.get("GCN @ 32 dist") < r.get("GCN @ 2 dist") / 2, r.toString());
    }

    @Test
    void randomWalkEmbeddingsClassifyNodes() {
        Map<String, Double> r = GraphLearningShootout.randomWalkEmbeddings(new double[][] {{1, 1}, {1, 0.5}, {1, 2}},
                2);
        System.out.printf("7.1.12 link AUC: DeepWalk %.3f, node2vec q=0.5 %.3f, q=2 %.3f; GAE %.3f, CN %.3f, spectral %.3f."
                + " Node acc: DeepWalk %.3f vs spectral %.3f%n", r.get("DeepWalk link AUC"),
                r.get("node2vec p=1.0 q=0.5 link AUC"), r.get("node2vec p=1.0 q=2.0 link AUC"), r.get("GAE link AUC"),
                r.get("common neighbours link AUC"), r.get("spectral link AUC"), r.get("DeepWalk node acc"),
                r.get("spectral node acc"));
        assertTrue(r.get("DeepWalk link AUC") > r.get("common neighbours link AUC"), r.toString());
        assertTrue(r.get("DeepWalk node acc") > r.get("spectral node acc"), r.toString());
    }
}
