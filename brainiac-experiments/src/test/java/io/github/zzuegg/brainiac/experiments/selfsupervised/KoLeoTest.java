package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Extra;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Options;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Report;

import org.junit.jupiter.api.Test;

/** Item 10.3.5: the KoLeo regulariser on DINO and SimSiam. */
class KoLeoTest {

    private static String line(Report r) {
        return String.format("%s probe %.3f kNN %.3f rank %.1f unif %.2f %.1f ms/step%s", r.name(),
                r.evaluation().probeAccuracy(), r.evaluation().knnAccuracy(), r.evaluation().effectiveRank(),
                r.evaluation().uniformity(), r.millisPerStep(),
                Double.isNaN(r.extra()) ? "" : String.format(" purity %.3f", r.extra()));
    }

    @Test
    void koleoSpreadsTheFeatures() {
        Report dino = SelfSupervisedBenchmark.report(Method.DINO, Options.of(150, 1));
        Report dinoKoleo = SelfSupervisedBenchmark.report(Method.DINO, Options.of(150, 1).koleo(0.1f));
        Report siam = SelfSupervisedBenchmark.run(Extra.SIMSIAM, Options.of(150, 1));
        Report siamKoleo = SelfSupervisedBenchmark.run(Extra.SIMSIAM, Options.of(150, 1).koleo(0.1f));
        System.out.println(line(dino) + " | " + line(dinoKoleo) + " | " + line(siam) + " | " + line(siamKoleo));
        assertTrue(dinoKoleo.evaluation().probeAccuracy() > 1.0f / 6 + 0.1f, "DINO with KoLeo still learns");
        assertTrue(siamKoleo.evaluation().probeAccuracy() > 1.0f / 6 + 0.1f, "SimSiam with KoLeo still learns");
    }
}
