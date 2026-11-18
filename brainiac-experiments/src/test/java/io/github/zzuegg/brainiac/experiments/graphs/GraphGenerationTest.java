package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Wave 7 item 7.1.16. */
class GraphGenerationTest {

    @Test
    void discreteDiffusionGeneratesCommunityGraphs() {
        Map<String, Double> r = GraphGeneration.compare(false, 48, 24, 200, 1);
        System.out.printf("7.1.16 SBM: valid DiGress %.3f vs ER %.3f, GraphRNN-lite %.3f, VGAE %.3f (real %.3f). Spectral"
                + " MMD DiGress %.3f vs ER %.3f, GraphRNN %.3f, VGAE %.3f. Degree MMD DiGress %.3f vs ER %.3f%n",
                r.get("DiGress valid"), r.get("Erdos-Renyi valid"), r.get("GraphRNN-lite valid"), r.get("VGAE valid"),
                r.get("held-out valid"), r.get("DiGress spectral MMD"), r.get("Erdos-Renyi spectral MMD"),
                r.get("GraphRNN-lite spectral MMD"), r.get("VGAE spectral MMD"), r.get("DiGress degree MMD"),
                r.get("Erdos-Renyi degree MMD"));
        assertTrue(r.get("DiGress valid") > r.get("Erdos-Renyi valid"), r.toString());
    }

    @Test
    void gridsAsAPlanarFamily() {
        Map<String, Double> r = GraphGeneration.compare(true, 48, 24, 150, 2);
        System.out.printf("7.1.16 grids: valid (connected, degree <= 4) DiGress %.3f vs ER %.3f, GraphRNN-lite %.3f,"
                + " VGAE %.3f; degree MMD DiGress %.3f vs ER %.3f%n", r.get("DiGress valid"), r.get("Erdos-Renyi valid"),
                r.get("GraphRNN-lite valid"), r.get("VGAE valid"), r.get("DiGress degree MMD"),
                r.get("Erdos-Renyi degree MMD"));
        assertTrue(r.get("DiGress degree MMD") < 0.5, r.toString());
    }
}
