package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Wave 7 items 7.1.6 and 7.1.18. */
class DiffusionAndSmoothingTest {

    @Test
    void grandImprovesWithIntegrationTime() {
        Map<String, Double> r = GraphLearningShootout.graphDiffusion(new float[] {2, 8}, 80, 7);
        System.out.printf("7.1.6 GRAND T=2 %.3f (%.0f evals), T=8 %.3f (%.0f evals); GCN %.3f, GAT %.3f, GCNII-8 %.3f%n",
                r.get("GRAND T=2.0"), r.get("GRAND T=2.0 evals"), r.get("GRAND T=8.0"), r.get("GRAND T=8.0 evals"),
                r.get("GCN"), r.get("GAT"), r.get("GCNII (8)"));
        assertTrue(r.get("GRAND T=8.0") > r.get("GRAND T=2.0"), r.toString());
        assertTrue(r.get("GRAND T=8.0") > 0.5, r.toString());
    }

    @Test
    void correctAndSmoothLiftsTheMlp() {
        Map<String, Double> r = GraphLearningShootout.correctAndSmooth(new double[] {0.5, 2.0}, 0.8f, 100, 3);
        System.out.printf("7.1.18 noise 0.5: MLP+C&S %.3f vs MLP %.3f, GCN %.3f, APPNP %.3f, LP %.3f. noise 2.0: MLP+C&S"
                + " %.3f vs MLP %.3f, GCN %.3f, LP %.3f. Train ms: MLP+C&S %.0f, GCN %.0f, GAT %.0f%n",
                r.get("MLP + C&S @ 0.5"), r.get("MLP @ 0.5"), r.get("GCN @ 0.5"), r.get("APPNP @ 0.5"), r.get("LP @ 0.5"),
                r.get("MLP + C&S @ 2.0"), r.get("MLP @ 2.0"), r.get("GCN @ 2.0"), r.get("LP @ 2.0"),
                r.get("MLP + C&S ms"), r.get("GCN ms"), r.get("GAT ms"));
        assertTrue(r.get("MLP + C&S @ 0.5") > r.get("MLP @ 0.5") + 0.03, r.toString());
        assertTrue(r.get("LP @ 2.0") > r.get("MLP @ 2.0") + 0.3, r.toString());
    }
}
