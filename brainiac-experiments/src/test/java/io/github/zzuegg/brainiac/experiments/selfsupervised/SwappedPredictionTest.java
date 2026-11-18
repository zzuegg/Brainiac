package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Extra;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Options;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Report;

import org.junit.jupiter.api.Test;

/** Item 10.3.2: SwAV with and without multi-crop, against DINO and SimCLR. */
class SwappedPredictionTest {

    private static String line(Report r) {
        return String.format("%s probe %.3f kNN %.3f rank %.1f unif %.2f %.1f ms/step%s", r.name(),
                r.evaluation().probeAccuracy(), r.evaluation().knnAccuracy(), r.evaluation().effectiveRank(),
                r.evaluation().uniformity(), r.millisPerStep(),
                Double.isNaN(r.extra()) ? "" : String.format(" purity %.3f", r.extra()));
    }

    @Test
    void swavLearnsWithBalancedCodes() {
        Report swav = SelfSupervisedBenchmark.run(Extra.SWAV, Options.of(150, 1));
        Report multi = SelfSupervisedBenchmark.run(Extra.SWAV_MULTI_CROP, Options.of(150, 1));
        Report dino = SelfSupervisedBenchmark.report(Method.DINO, Options.of(150, 1));
        Report simclr = SelfSupervisedBenchmark.report(Method.SIMCLR, Options.of(150, 1));
        System.out.println(line(swav) + " | " + line(multi) + " | " + line(dino) + " | " + line(simclr));
        assertTrue(swav.evaluation().probeAccuracy() > 1.0f / 6 + 0.15f, "SwAV beats chance");
        assertTrue(swav.extra() > 1.0 / 6, "prototype assignments carry class information");
    }
}
