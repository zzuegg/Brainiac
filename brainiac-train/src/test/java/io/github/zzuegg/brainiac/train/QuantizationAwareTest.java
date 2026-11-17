package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Quantization;
import io.github.zzuegg.brainiac.tensor.DType;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class QuantizationAwareTest {

    /** Three interleaved spirals: needs a curved boundary, so coarse weights hurt. */
    private static Dataset spirals(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] x = new float[samples][];
        float[][] y = new float[samples][3];
        for (int i = 0; i < samples; i++) {
            int c = i % 3;
            double r = random.nextDouble();
            double angle = c * 2 * Math.PI / 3 + r * 4 + random.nextGaussian() * 0.15;
            x[i] = new float[] {(float) (r * Math.cos(angle)), (float) (r * Math.sin(angle))};
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    private static Model model() {
        return Model.sequential().input(2).dense(48, Activation.TANH).dense(48, Activation.TANH)
                .dense(3, Activation.SOFTMAX).seed(5).build();
    }

    private static Trainer trainer(Model m, int epochs) {
        return Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f)).batchSize(32).epochs(epochs)
                .metric(Metric.ACCURACY).seed(3);
    }

    private static float accuracy(Model m, Dataset test) {
        return trainer(m, 1).evaluate(test).get(Metric.ACCURACY);
    }

    private static void assertOnGrid(Model m, Quantization q) {
        for (Quantization.Weight w : Quantization.weights(m)) {
            float[] v = w.weight().value().toArray();
            float[] again = q.quantize(w.weight().value(), w.axis()).toArray();
            for (int i = 0; i < v.length; i++) {
                assertEquals(again[i], v[i], 1e-6f);
            }
        }
    }

    @Test
    void trainingForTwoBitsBeatsRoundingAfterwards() {
        Dataset train = spirals(900, 1);
        Dataset test = spirals(300, 2);
        Model base = model();
        trainer(base, 60).fit(train);
        float full = accuracy(base, test);
        assertTrue(full > 0.9f, "float accuracy " + full);

        Quantization q = Quantization.bits(2);
        Model rounded = base.copy(base.device()).quantize(q);
        float ptq = accuracy(rounded, test);

        Model aware = base.copy(base.device());
        trainer(aware, 20).quantizationAware(q).validation(test).fit(train);
        assertEquals(q, aware.quantization());
        assertOnGrid(aware, q);
        float qat = accuracy(aware, test);
        assertTrue(qat > ptq + 0.05f && qat > 0.8f, "2-bit: rounded " + ptq + ", trained " + qat + ", float " + full);
    }

    @Test
    void worksWithMixedPrecisionAndKeepsFloat32Weights() {
        Model m = model();
        Quantization q = Quantization.int4().groups(16);
        trainer(m, 30).mixedPrecision(DType.BFLOAT16).quantizationAware(q).fit(spirals(600, 1));
        assertEquals(DType.FLOAT32, m.dtype());
        assertOnGrid(m, q);
        float acc = accuracy(m, spirals(300, 2));
        assertTrue(acc > 0.8f, "accuracy " + acc);
    }
}
