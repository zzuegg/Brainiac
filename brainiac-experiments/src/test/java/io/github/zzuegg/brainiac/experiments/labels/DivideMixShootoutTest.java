package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Noise;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Result;

import org.junit.jupiter.api.Test;

class DivideMixShootoutTest {

    @Test
    void divideMixAtFortyPercentSymmetricNoise() {
        Result ce = NoisyLabelShootout.run(Method.CE, Noise.SYMMETRIC, 0.4, 600, 1);
        Result dm = NoisyLabelShootout.run(Method.DIVIDEMIX, Noise.SYMMETRIC, 0.4, 600, 1);
        RobustLossShootoutTest.print("sym 0.4", ce);
        RobustLossShootoutTest.print("sym 0.4", dm);
        assertTrue(dm.finalAccuracy() > ce.finalAccuracy(), "DivideMix beats CE");
        assertTrue(dm.splitPrecision() > 0.8, "the GMM split is mostly clean");
    }
}
