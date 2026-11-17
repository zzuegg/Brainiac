package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class OneStepFrontierTest {
    static final ToyPoints RING = OneStepGeneratorsTest.RING;
    static final int IT = 700;

    private static String fmt(OneStepGenerators.Score s) {
        return String.format("%s %.3f/%.2f/%d", s.method(), s.distance(), s.nearShare(), s.modes());
    }

    @Test
    void dmdDistilsAOneStepGenerator() {
        long t0 = System.nanoTime();
        OneStepGenerators.Score[] dmd = OneStepFrontier.dmd(RING, IT, IT, 256, 1);
        OneStepGenerators.Score mean = OneStepGenerators.meanFlow(RING, IT, 256, 1);
        System.out.printf("one-step, 4-blob ring (distance/near/modes): %s; baselines %s, %s [%.1f s]%n",
                fmt(dmd[0]), fmt(dmd[1]), fmt(mean), (System.nanoTime() - t0) / 1e9);
        assertTrue(dmd[0].modes() >= 4 && dmd[0].distance() < 0.45, "DMD covers the ring in one step");
    }

    @Test
    void momentMatchingAndContinuousConsistency() {
        long t0 = System.nanoTime();
        OneStepGenerators.Score[] imm = OneStepFrontier.imm(RING, IT, 256, 1);
        OneStepGenerators.Score[] scm = OneStepFrontier.scm(RING, IT, 256, 1);
        OneStepGenerators.Score cm = OneStepGenerators.consistency(RING, IT, 256, 1);
        System.out.printf("few-step, 4-blob ring: %s, %s, %s; %s, %s, %s; baseline %s [%.1f s]%n", fmt(imm[0]),
                fmt(imm[1]), fmt(imm[2]), fmt(scm[0]), fmt(scm[1]), fmt(scm[2]), fmt(cm),
                (System.nanoTime() - t0) / 1e9);
        assertTrue(imm[0].modes() >= 4 && scm[0].modes() >= 4, "both cover the ring");
        assertTrue(imm[2].distance() < imm[0].distance() && imm[2].distance() < 0.4, "IMM improves with steps");
        assertTrue(scm[0].distance() < 0.8, "sCM learns a one-step map");
    }

    @Test
    void driftingAgainstAGan() {
        long t0 = System.nanoTime();
        OneStepGenerators.Score drift = OneStepFrontier.drifting(RING, IT, 256, 1);
        OneStepGenerators.Score gan = OneStepFrontier.gan(RING, IT, 256, 1);
        System.out.printf("one-step, 4-blob ring: %s; baseline %s [%.1f s]%n", fmt(drift), fmt(gan),
                (System.nanoTime() - t0) / 1e9);
        assertTrue(drift.modes() >= 4 && drift.distance() < 0.35, "drifting covers the ring in one step");
    }
}
