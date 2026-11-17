package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BitNetExperimentTest {

    @Test
    void quantizationAwareTrainingBeatsPostTrainingTernarization() {
        float[] y = new float[1000];
        Tensor x = BitNetExperiment.spirals(1000, new SplittableRandom(9), y);
        Model fp = BitNetExperiment.model(false, 1);
        Model qat = BitNetExperiment.model(true, 1);
        BitNetExperiment.train(fp, 600, 2);
        BitNetExperiment.train(qat, 600, 2);
        double floatAcc = BitNetExperiment.accuracy(fp, x, y);
        double qatAcc = BitNetExperiment.accuracy(qat, x, y);
        BitNetExperiment.ternarizeHidden(fp);
        double ptqAcc = BitNetExperiment.accuracy(fp, x, y);
        System.out.printf("BitNet spirals accuracy: float %.3f, ternary QAT %.3f, post-training ternary %.3f%n", floatAcc,
                qatAcc, ptqAcc);
        assertTrue(floatAcc > 0.95, "float " + floatAcc);
        assertTrue(qatAcc > 0.9 && qatAcc > ptqAcc, "QAT " + qatAcc + " vs post-training ternary " + ptqAcc);

        float[] packed = BitNetExperiment.packedForward(qat, x);
        float[] fake = qat.forward(Var.of(x)).value().toArray();
        int agree = 0;
        for (int i = 0; i < 1000; i++) {
            agree += (packed[2 * i] > packed[2 * i + 1]) == (fake[2 * i] > fake[2 * i + 1]) ? 1 : 0;
        }
        System.out.printf("BitNet packed additions-only inference agrees on %d/1000 predictions%n", agree);
        assertEquals(1000, agree, 5, "packed additions-only inference gives the same predictions");
    }
}
