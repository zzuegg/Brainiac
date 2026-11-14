package io.github.zzuegg.brainiac.examples;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.TrainingResult;

import org.junit.jupiter.api.Test;

/** The milestone 1 acceptance checks, run through the public examples. */
class AcceptanceTest {

    @Test
    void xorReachesLossBelowOneHundredthWithin2000Epochs() {
        TrainingResult r = XorExample.train(XorExample.model(), false);
        assertTrue(r.finalLoss() < 0.01, "loss " + r.finalLoss());
    }

    @Test
    void irisReachesAtLeast95PercentTestAccuracy() {
        IrisExample.Result r = IrisExample.run(IrisExample.DATA_FILE, false);
        float accuracy = r.testEvaluation().get(Metric.ACCURACY);
        assertTrue(accuracy >= 0.95f, "test accuracy " + accuracy);
    }

    @Test
    void spiralIsLearned() {
        Dataset data = SpiralExample.data(100, 3, 1);
        TrainingResult r = SpiralExample.train(SpiralExample.model(3), data, false);
        float accuracy = r.last().metrics().get("accuracy");
        assertTrue(accuracy >= 0.9f, "accuracy " + accuracy);
    }

    @Test
    void sameSeedGivesBitIdenticalTraining() {
        Model a = XorExample.model();
        Model b = XorExample.model();
        XorExample.train(a, false);
        XorExample.train(b, false);
        for (int i = 0; i < a.parameters().size(); i++) {
            assertArrayEquals(a.parameters().get(i).value().toArray(), b.parameters().get(i).value().toArray());
        }
    }
}
