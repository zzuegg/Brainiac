package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class CircuitExperimentsTest {

    @Test
    void attributionPatchingTracksExactPatching() {
        PatchingAttribution.Outcome o = PatchingAttribution.run(400, 1);
        System.out.printf("AtP on recall (acc %.2f, %d nodes): Spearman AtP %.3f / AtP* %.3f, top-3 hits %d / %d, "
                + "time exact %.0f ms vs AtP %.0f ms vs AtP* %.0f ms%n", o.accuracy(), o.nodes(), o.atpSpearman(),
                o.atpStarSpearman(), o.atpTop3(), o.atpStarTop3(), o.exactMillis(), o.atpMillis(), o.atpStarMillis());
        assertTrue(o.accuracy() > 0.8, o.toString());
        assertTrue(o.atpSpearman() > 0.5, o.toString());
        assertTrue(o.atpTop3() >= 2, o.toString());
    }

    @Test
    void weightSparseTrainingLeavesSmallerCircuits() {
        WeightSparseCircuits.Outcome o = WeightSparseCircuits.run(30, 1);
        System.out.printf("Bracket matching: circuit at acc >= %.3f: weight-sparse %d vs dense pruned %d of %d weights "
                + "(acc sparse %.3f, dense %.3f)%n", o.target(), o.sparseCircuit(), o.denseCircuit(),
                o.totalWeights(), o.sparseAccuracy(), o.denseAccuracy());
        assertTrue(o.sparseAccuracy() > 0.85, o.toString());
        assertTrue(o.sparseCircuit() < o.denseCircuit(), o.toString());
    }
}
