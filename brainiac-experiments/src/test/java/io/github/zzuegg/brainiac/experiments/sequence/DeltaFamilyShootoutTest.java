package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DeltaFamilyShootoutTest {

    @Test
    void kimiDeltaRecallsBetterThanScalarGates() {
        MixerShootout.Task task = MixerShootout.recall(8, 8, 4);
        List<MixerShootout.Result> results = DeltaFamilyShootout.run(DeltaFamilyShootout.recallLineup(),
                List.of("KDA", "Gated DeltaNet", "Mamba-2"), List.of(task), 16, 300, 1);
        System.out.print(MixerShootout.table(results));
        double kda = DeltaFamilyShootout.accuracy(results, "KDA", task.name());
        double gdn = DeltaFamilyShootout.accuracy(results, "Gated DeltaNet", task.name());
        double mamba = DeltaFamilyShootout.accuracy(results, "Mamba-2", task.name());
        System.out.printf("recall 4 pairs: KDA %.3f vs Gated DeltaNet %.3f vs Mamba-2 %.3f (chance 0.125)%n", kda, gdn,
                mamba);
        assertTrue(kda > 0.8);
        assertTrue(kda > gdn + 0.1);
        assertTrue(kda > mamba);
    }

    @Test
    void logLinearStatesRecallMorePairsThanOneState() {
        MixerShootout.Task task = MixerShootout.recall(16, 16, 12);
        List<MixerShootout.Result> results = DeltaFamilyShootout.run(DeltaFamilyShootout.recallLineup(),
                List.of("gated linear", "log-linear"), List.of(task), 32, 300, 1);
        System.out.print(MixerShootout.table(results));
        double linear = DeltaFamilyShootout.accuracy(results, "gated linear", task.name());
        double logLinear = DeltaFamilyShootout.accuracy(results, "log-linear", task.name());
        System.out.printf("recall 12 pairs: log-linear (O(log T) states) %.3f vs gated linear (one state) %.3f "
                + "(chance 0.0625)%n", logLinear, linear);
        assertTrue(logLinear > linear);
    }
}
