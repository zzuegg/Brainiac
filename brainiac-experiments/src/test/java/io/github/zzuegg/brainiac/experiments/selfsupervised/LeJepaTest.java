package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Extra;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Options;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Report;

import org.junit.jupiter.api.Test;

/** Item 10.3.6: LeJEPA with SIGReg on both encoders, against VICReg in its place and I-JEPA. */
class LeJepaTest {

    private static String line(Report r) {
        return String.format("%s probe %.3f kNN %.3f rank %.1f unif %.2f %.1f ms/step%s", r.name(),
                r.evaluation().probeAccuracy(), r.evaluation().knnAccuracy(), r.evaluation().effectiveRank(),
                r.evaluation().uniformity(), r.millisPerStep(),
                Double.isNaN(r.extra()) ? "" : String.format(" purity %.3f", r.extra()));
    }

    @Test
    void lejepaLearnsWithoutTeacherOrPredictor() {
        Report mlp = SelfSupervisedBenchmark.run(Extra.LEJEPA, Options.of(150, 1));
        Report vic = SelfSupervisedBenchmark.run(Extra.LEJEPA_VICREG, Options.of(150, 1));
        Report patches = SelfSupervisedBenchmark.run(Extra.LEJEPA_PATCHES, Options.of(60, 1));
        SelfSupervisedBenchmark.Result ijepa = SelfSupervisedBenchmark.run(Method.IJEPA, 60, 1);
        System.out.println(line(mlp) + " | " + line(vic) + " | " + line(patches)
                + String.format(" | I-JEPA-lite probe %.3f", ijepa.probeAccuracy()));
        assertTrue(mlp.evaluation().probeAccuracy() > 1.0f / 6 + 0.15f, "LeJEPA beats chance");
        assertTrue(mlp.evaluation().effectiveRank() > 4, "LeJEPA does not collapse");
        assertTrue(mlp.lastLoss() < mlp.firstLoss(), "LeJEPA's loss falls");
    }
}
