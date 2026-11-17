package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SoftHebbLayerTest {

    static final float[][] CENTERS = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-0.6f, -0.6f, 0.53f}};

    /** Noisy points around four directions, with one-hot cluster targets. */
    static float[][][] clusters(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][3];
        float[][] y = new float[n][4];
        for (int i = 0; i < n; i++) {
            int c = i % 4;
            for (int k = 0; k < 3; k++) {
                x[i][k] = CENTERS[c][k] + (float) r.nextGaussian() * 0.15f;
            }
            y[i][c] = 1;
        }
        return new float[][][] {x, y};
    }

    @Test
    void unitsBecomePrototypesOfTheClusters() {
        SoftHebbLayer layer = new SoftHebbLayer(3, 8, 0.1f, 0.1f, new SplittableRandom(2));
        for (int i = 0; i < 300; i++) {
            layer.learn(Tensor.of(clusters(16, i)[0]));
        }
        float[] w = layer.weight().toArray();   // [3, 8]
        for (float[] c : CENTERS) {
            double best = -1;
            double cn = Math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
            for (int u = 0; u < 8; u++) {
                double dot = 0;
                double wn = 0;
                for (int k = 0; k < 3; k++) {
                    dot += w[k * 8 + u] * c[k];
                    wn += w[k * 8 + u] * w[k * 8 + u];
                }
                best = Math.max(best, dot / (cn * Math.sqrt(wn)));
            }
            assertTrue(best > 0.9, "a unit points at every cluster: " + best);
        }
    }

    @Test
    void worksInAModelTrainedByTheCoreTrainer() {
        Model model = Model.sequential().input(3).add(SoftHebbLayer.spec(8, 0.1f, 0.1f))
                .dense(4, Activation.SOFTMAX).seed(1).build();
        float[][][] d = clusters(400, 7);
        Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.05f)).epochs(15).batchSize(32)
                .fit(Dataset.of(d[0], d[1]));
        float[][][] t = clusters(200, 8);
        float[] p = model.predict(Tensor.of(t[0])).toArray();
        int correct = 0;
        for (int i = 0; i < 200; i++) {
            int best = 0;
            for (int c = 1; c < 4; c++) {
                if (p[i * 4 + c] > p[i * 4 + best]) {
                    best = c;
                }
            }
            correct += t[1][i][best] == 1 ? 1 : 0;
        }
        assertTrue(correct > 180, "readout on Hebbian features: " + correct + "/200");
        assertTrue(model.parameters().stream().noneMatch(v -> v.shape().equals(Shape.of(3, 8))),
                "the Hebbian weight is a buffer, not a parameter");
    }
}
