package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class StateTrackingTest {

    @Test
    void negativeEigenvaluesSolveParity() {
        MixerShootout.Task task = MixerShootout.parity(8);
        List<MixerShootout.Result> results = DeltaFamilyShootout.run(DeltaFamilyShootout.trackingLineup(),
                List.of("Gated DeltaNet", "DeltaProduct n=1", "DeltaProduct n=2"), List.of(task), 16, 250, 1);
        System.out.print(MixerShootout.table(results));
        double gdn = DeltaFamilyShootout.accuracy(results, "Gated DeltaNet", task.name());
        double one = DeltaFamilyShootout.accuracy(results, "DeltaProduct n=1", task.name());
        double two = DeltaFamilyShootout.accuracy(results, "DeltaProduct n=2", task.name());
        System.out.printf("parity(8): DeltaProduct n=2 %.3f, n=1 %.3f (beta in (0,2)) vs Gated DeltaNet %.3f (beta in "
                + "(0,1)) (chance 0.5)%n", two, one, gdn);
        assertTrue(two > 0.9);
        assertTrue(two > gdn + 0.3);
        assertTrue(one >= gdn);
    }

    @Test
    void twoHouseholderStepsComposePermutations() {
        MixerShootout.Task task = DeltaFamilyShootout.permutations(3);
        List<MixerShootout.Result> results = DeltaFamilyShootout.run(DeltaFamilyShootout.trackingLineup(),
                List.of("DeltaProduct n=1", "DeltaProduct n=2"), List.of(task), 16, 600, 1);
        System.out.print(MixerShootout.table(results));
        double one = DeltaFamilyShootout.accuracy(results, "DeltaProduct n=1", task.name());
        double two = DeltaFamilyShootout.accuracy(results, "DeltaProduct n=2", task.name());
        System.out.printf("S3 composition of 3: DeltaProduct n=2 %.3f vs n=1 %.3f (chance 0.167)%n", two, one);
        assertTrue(two > 0.95);
        assertTrue(two > one + 0.05);
    }
}
