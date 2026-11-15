package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class PrecisionTest {

    private static Dataset clusters(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] centers = {{-1, -1}, {1, -1}, {0, 1}};
        float[][] x = new float[samples][];
        float[][] y = new float[samples][3];
        for (int i = 0; i < samples; i++) {
            int c = i % 3;
            x[i] = new float[] {centers[c][0] + (float) random.nextGaussian() * 0.25f,
                centers[c][1] + (float) random.nextGaussian() * 0.25f};
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    private static Model model() {
        return Model.sequential().input(2).dense(16, Activation.TANH).dense(3, Activation.SOFTMAX).seed(4).build();
    }

    private static Trainer trainer(Model m) {
        return Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.05f)).batchSize(16).epochs(15)
                .metric(Metric.ACCURACY).seed(2);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void mixedPrecisionTrainsAndKeepsFloat32Weights(DType type) {
        Model m = model();
        Trainer t = trainer(m).mixedPrecision(type);
        t.fit(clusters(300, 1));
        for (Var p : m.parameters()) {
            assertEquals(DType.FLOAT32, p.dtype());
        }
        float accuracy = t.evaluate(clusters(90, 2)).get(Metric.ACCURACY);
        assertTrue(accuracy > 0.95f, type + " accuracy " + accuracy);
        assertEquals(type == DType.FLOAT16 ? 65536f : 1f, t.lossScale());
    }

    @Test
    void overflowingStepsAreSkippedAndTheScaleShrinks() {
        Model m = model();
        Trainer t = trainer(m).epochs(3).mixedPrecision(DType.FLOAT16, 1e9f, 0);
        t.fit(clusters(300, 1));
        assertTrue(t.skippedSteps() > 0, "skipped " + t.skippedSteps());
        assertTrue(t.lossScale() < 1e9f, "scale " + t.lossScale());
        float accuracy = t.evaluate(clusters(90, 2)).get(Metric.ACCURACY);
        assertTrue(accuracy > 0.9f, "accuracy " + accuracy);
    }

    @Test
    void float64ModelTrainsLikeFloat32() {
        Model single = model();
        Model wide = model().to(DType.FLOAT64);
        assertEquals(DType.FLOAT64, wide.dtype());
        Trainer a = trainer(single);
        Trainer b = trainer(wide);
        a.fit(clusters(300, 1));
        b.fit(clusters(300, 1));
        for (Var p : wide.parameters()) {
            assertEquals(DType.FLOAT64, p.dtype());
        }
        Tensor x = clusters(30, 3).features();
        assertEquals(DType.FLOAT64, wide.predict(x).dtype());
        assertArrayEquals(single.predict(x).toArray(), wide.predict(x).toArray(), 2e-3f);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void savedModelsKeepTheirNumberFormat(DType type, @TempDir Path dir) throws IOException {
        Model m = model().to(type);
        Path file = dir.resolve("m.brainiac");
        m.save(file);
        Model back = Model.load(file);
        assertEquals(type, back.dtype());
        for (int i = 0; i < m.parameters().size(); i++) {
            assertArrayEquals(m.parameters().get(i).value().toDoubleArray(),
                    back.parameters().get(i).value().toDoubleArray(), 0);
        }
        Tensor x = Tensor.of(Shape.of(2, 2), 0.5f, -0.25f, 1, 0);
        assertArrayEquals(m.predict(x).toDoubleArray(), back.predict(x).toDoubleArray(), 0);
    }
}
