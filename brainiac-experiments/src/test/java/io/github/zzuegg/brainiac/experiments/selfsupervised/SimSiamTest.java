package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Extra;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Options;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Report;

import org.junit.jupiter.api.Test;

/** Item 10.3.1: SimSiam and its stop-gradient and predictor ablations, against BYOL. */
class SimSiamTest {

    private static String line(Report r) {
        return String.format("%s probe %.3f kNN %.3f rank %.1f", r.name(), r.evaluation().probeAccuracy(),
                r.evaluation().knnAccuracy(), r.evaluation().effectiveRank());
    }

    @Test
    void simSiamLearnsWithoutNegativesOrMomentum() {
        Report simSiam = SelfSupervisedBenchmark.run(Extra.SIMSIAM, Options.of(150, 1));
        Report noStop = SelfSupervisedBenchmark.run(Extra.SIMSIAM_NO_STOP_GRADIENT, Options.of(150, 1));
        Report noPred = SelfSupervisedBenchmark.run(Extra.SIMSIAM_NO_PREDICTOR, Options.of(150, 1));
        Report byol = SelfSupervisedBenchmark.report(Method.BYOL, Options.of(150, 1));
        System.out.println(line(simSiam) + " | " + line(noStop) + " | " + line(noPred) + " | " + line(byol)
                + " (random encoder 0.341)");
        assertTrue(simSiam.evaluation().probeAccuracy() > 1.0f / 6 + 0.15f, "SimSiam beats chance");
        assertTrue(simSiam.lastLoss() < simSiam.firstLoss(), "SimSiam's loss falls");
    }
}
