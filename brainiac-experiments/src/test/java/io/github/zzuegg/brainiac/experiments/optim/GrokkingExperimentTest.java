package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class GrokkingExperimentTest {

    @Test
    void modularAdditionSetupMemorisesAndWeightDecayKeepsWeightsSmall() {
        Tensor[] d = GrokkingExperiment.modularAddition(13, 0.5, 1);
        assertEquals(85, d[0].shape().dim(0));
        assertEquals(169 - 85, d[2].shape().dim(0));
        float[] x = d[0].slice(0, 0, 1).toArray();
        int a = -1;
        int b = -1;
        for (int i = 0; i < 13; i++) {
            a = x[i] == 1 ? i : a;
            b = x[13 + i] == 1 ? i : b;
        }
        assertEquals((a + b) % 13, Math.round(d[1].toArray()[0]));

        GrokkingExperiment.Curve curve = GrokkingExperiment.run(23, 0.5, 128, 400, 3e-3f, 1f, 50, 1);
        System.out.println("train " + curve.trainAccuracy() + "\ntest " + curve.testAccuracy() + "\nnorm "
                + curve.weightNorm());
        assertTrue(curve.memorisedAt() > 0, "the training pairs are memorised");
        GrokkingExperiment.Curve noDecay = GrokkingExperiment.run(23, 0.5, 128, 400, 3e-3f, 0f, 400, 1);
        assertTrue(curve.weightNorm().getLast() < noDecay.weightNorm().getLast(),
                "weight decay, the driver of grokking, keeps the weights smaller");
    }
}
