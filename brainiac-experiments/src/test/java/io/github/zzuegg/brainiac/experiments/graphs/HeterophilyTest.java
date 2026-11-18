package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/** Wave 7 items 7.1.4 and 7.1.5. */
class HeterophilyTest {

    @Test
    void heterophilyLayersStayAboveGcnWhenNeighboursDiffer() {
        Map<String, Double> r = GraphLearningShootout.heterophilySweep(
                List.of("MLP", "GCN", "SAGE", "H2GCN", "GPR-GNN", "Sheaf diag", "Sheaf orth"), new double[] {0.1, 0.9},
                1, 100, 3);
        System.out.printf("7.1.4/7.1.5 h=0.1: H2GCN %.3f, GPR-GNN %.3f, sheaf diag %.3f, sheaf orth %.3f; GCN %.3f,"
                + " SAGE %.3f, MLP %.3f. h=0.9: H2GCN %.3f, GPR %.3f, sheaf %.3f/%.3f; GCN %.3f, MLP %.3f%n",
                r.get("H2GCN @ 0.1"), r.get("GPR-GNN @ 0.1"), r.get("Sheaf diag @ 0.1"), r.get("Sheaf orth @ 0.1"),
                r.get("GCN @ 0.1"), r.get("SAGE @ 0.1"), r.get("MLP @ 0.1"), r.get("H2GCN @ 0.9"),
                r.get("GPR-GNN @ 0.9"), r.get("Sheaf diag @ 0.9"), r.get("Sheaf orth @ 0.9"), r.get("GCN @ 0.9"),
                r.get("MLP @ 0.9"));
        assertTrue(r.get("GCN @ 0.1") < r.get("MLP @ 0.1"), r.toString());
        assertTrue(r.get("H2GCN @ 0.1") > r.get("GCN @ 0.1") + 0.1, r.toString());
        assertTrue(r.get("GPR-GNN @ 0.1") > r.get("GCN @ 0.1"), r.toString());
        assertTrue(r.get("Sheaf diag @ 0.1") > r.get("GCN @ 0.1"), r.toString());
        assertTrue(r.get("GCN @ 0.9") > r.get("MLP @ 0.9") + 0.2, r.toString());
    }
}
