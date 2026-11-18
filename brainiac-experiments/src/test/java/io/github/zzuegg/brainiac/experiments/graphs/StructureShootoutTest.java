package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Wave 7 item 7.1.1. */
class StructureShootoutTest {

    @Test
    void randomWalkEncodingsCountTriangles() {
        Map<String, Double> r = GraphStructureShootout.positionalEncodings(30, 80, 1);
        System.out.printf("7.1.1 triangle MAE: GIN+RWSE %.3f, GPS+RWSE %.3f, GIN+LapPE %.3f, GPS %.3f; GIN %.3f, GCN %.3f,"
                + " mean %.3f. Trees/cycles: GPS+RWSE %.3f, GPS+LapPE %.3f vs GIN %.3f. SBM: GIN+LapPE %.3f, GPS+LapPE"
                + " %.3f vs GCN %.3f%n", r.get("triangles MAE GIN + RWSE"), r.get("triangles MAE GPS + RWSE"),
                r.get("triangles MAE GIN + LapPE"), r.get("triangles MAE GPS"), r.get("triangles MAE GIN"),
                r.get("triangles MAE GCN"), r.get("triangles MAE predict-mean"), r.get("trees/cycles GPS + RWSE"),
                r.get("trees/cycles GPS + LapPE"), r.get("trees/cycles GIN"), r.get("SBM GIN + LapPE"),
                r.get("SBM GPS + LapPE"), r.get("SBM GCN"));
        assertTrue(r.get("triangles MAE GIN + RWSE") < r.get("triangles MAE GIN") * 0.6, r.toString());
        assertTrue(r.get("triangles MAE GPS + RWSE") < r.get("triangles MAE GIN"), r.toString());
    }
}
