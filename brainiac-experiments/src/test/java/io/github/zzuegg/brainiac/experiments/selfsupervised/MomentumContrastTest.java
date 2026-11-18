package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Extra;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Options;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Report;

import org.junit.jupiter.api.Test;

/** Item 10.3.3: MoCo v2 and v3 against SimCLR at a small batch of 16. */
class MomentumContrastTest {

    private static String line(Report r) {
        return String.format("%s probe %.3f kNN %.3f rank %.1f unif %.2f %.1f ms/step%s", r.name(),
                r.evaluation().probeAccuracy(), r.evaluation().knnAccuracy(), r.evaluation().effectiveRank(),
                r.evaluation().uniformity(), r.millisPerStep(),
                Double.isNaN(r.extra()) ? "" : String.format(" purity %.3f", r.extra()));
    }

    @Test
    void mocoHoldsUpAtSmallBatch() {
        Report moco = SelfSupervisedBenchmark.run(Extra.MOCO_V2, Options.of(200, 1).batch(16));
        Report v3 = SelfSupervisedBenchmark.run(Extra.MOCO_V3, Options.of(200, 1).batch(16));
        Report simclr = SelfSupervisedBenchmark.report(Method.SIMCLR, Options.of(200, 1).batch(16));
        System.out.println(line(moco) + " | " + line(v3) + " | " + line(simclr));
        assertTrue(moco.evaluation().probeAccuracy() > 1.0f / 6 + 0.15f, "MoCo v2 beats chance");
        assertTrue(v3.evaluation().probeAccuracy() > 1.0f / 6 + 0.1f, "MoCo v3 beats chance");
    }
}
