package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class RepresentationComparisonTest {

    @Test
    void ckaMatchesLayersAcrossSeeds() {
        RepresentationComparison.Similarity s = RepresentationComparison.similarity(32, 3, 600, 1);
        s.correspondence().forEach((k, v) -> System.out.printf("Layer correspondence %-24s %.3f%n", k, v));
        System.out.printf("Stitching penalty per layer %s, linear CKA %s, accuracy %.3f%n",
                Arrays.toString(s.stitchPenalty()), Arrays.toString(s.stitchCka()), s.accuracy());
        assertTrue(s.accuracy() > 0.5, "the networks learn");
        assertTrue(s.correspondence().get("linear CKA") > 0.4, "CKA mostly matches layers");
    }

    @Test
    void alignmentShrinksTheBarrier() {
        for (int width : new int[] {16, 64}) {
            RepresentationComparison.Connectivity c = RepresentationComparison.connectivity(width, 700, 1);
            System.out.printf("Connectivity width %d: barrier naive %.3f weight-matched %.3f activation-matched %.3f;"
                    + " accuracy A %.3f merged naive %.3f weight %.3f activation %.3f SLERP %.3f ensemble %.3f%n",
                    width, c.barrierNaive(), c.barrierWeight(), c.barrierActivation(), c.accuracyA(), c.mergedNaive(),
                    c.mergedWeight(), c.mergedActivation(), c.slerp(), c.ensemble());
            assertTrue(Math.min(c.barrierWeight(), c.barrierActivation()) < c.barrierNaive(),
                    "aligning lowers the barrier");
        }
    }
}
