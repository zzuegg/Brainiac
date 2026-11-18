package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Wave 7 item 7.1.15. */
class NeuroSatTest {

    @Test
    void neuroSatClassifiesSrPairs() {
        Map<String, Double> r = GraphReasoning.neuroSat(60, 6, new int[] {6, 12}, 8, new int[] {8, 24}, 200, 1);
        System.out.printf("7.1.15 SR(6) accuracy: NeuroSAT T=8 %.3f, T=24 %.3f vs GIN %.3f; decoded T=8 %.3f, T=24 %.3f"
                + " vs two random assignments %.3f, WalkSAT %.3f. SR(12): NeuroSAT %.3f, GIN %.3f, decoded %.3f vs"
                + " random %.3f%n",
                r.get("NeuroSAT n=6 T=8"), r.get("NeuroSAT n=6 T=24"), r.get("GIN n=6"), r.get("NeuroSAT decoded n=6 T=8"),
                r.get("NeuroSAT decoded n=6 T=24"),
                r.get("two random assignments solved n=6"), r.get("WalkSAT solved n=6"), r.get("NeuroSAT n=12 T=8"),
                r.get("GIN n=12"), r.get("NeuroSAT decoded n=12 T=8"), r.get("two random assignments solved n=12"));
        assertTrue(r.get("NeuroSAT n=6 T=8") > 0.6, r.toString());
        assertTrue(r.get("NeuroSAT decoded n=6 T=24") > r.get("two random assignments solved n=6") + 0.4, r.toString());
    }
}
