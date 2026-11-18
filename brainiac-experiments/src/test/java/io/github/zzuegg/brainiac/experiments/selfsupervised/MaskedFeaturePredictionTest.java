package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Extra;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Options;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Report;

import org.junit.jupiter.api.Test;

/** Item 10.3.4: data2vec and iBOT on the patch transformer, against MAE. */
class MaskedFeaturePredictionTest {

    private static String line(Report r) {
        return String.format("%s probe %.3f kNN %.3f rank %.1f unif %.2f %.1f ms/step%s", r.name(),
                r.evaluation().probeAccuracy(), r.evaluation().knnAccuracy(), r.evaluation().effectiveRank(),
                r.evaluation().uniformity(), r.millisPerStep(),
                Double.isNaN(r.extra()) ? "" : String.format(" purity %.3f", r.extra()));
    }

    @Test
    void teacherFeatureTargetsTrainThePatchEncoder() {
        Report d2v = SelfSupervisedBenchmark.run(Extra.DATA2VEC, Options.of(80, 1));
        Report ibot = SelfSupervisedBenchmark.run(Extra.IBOT, Options.of(80, 1));
        SelfSupervisedBenchmark.Result mae = SelfSupervisedBenchmark.run(Method.MAE, 80, 1);
        System.out.println(line(d2v) + " | " + line(ibot) + String.format(" | MAE probe %.3f (random %.3f)",
                mae.probeAccuracy(), mae.randomProbeAccuracy()));
        assertTrue(d2v.lastLoss() < d2v.firstLoss(), "data2vec's regression loss falls");
        assertTrue(d2v.evaluation().probeAccuracy() > 1.0f / 6 + 0.1f, "data2vec beats chance");
        assertTrue(ibot.evaluation().probeAccuracy() > 1.0f / 6 + 0.1f, "iBOT beats chance");
    }
}
