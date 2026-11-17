package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Modern Hopfield networks, product-key memory and Engram n-gram memory. */
class AssociativeMemoryTest {

    @Test
    void modernHopfieldRetrievesFarBeyondClassicalCapacity() {
        int d = 64;
        int n = 200;   // a classical Hopfield net stores about 0.14 * 64 = 9 patterns
        SplittableRandom r = new SplittableRandom(1);
        float[] x = new float[n * d];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextBoolean() ? 1 : -1;
        }
        int queries = 20;
        float[] q = new float[queries * d];
        for (int i = 0; i < queries; i++) {
            for (int j = 0; j < d; j++) {
                float v = x[i * d + j];
                q[i * d + j] = r.nextDouble() < 0.15 ? -v : v;   // 15% of bits flipped
            }
        }
        Tensor stored = Tensor.of(Shape.of(n, d), x);
        float[] got = ModernHopfield.retrieve(Var.of(stored), Var.of(Tensor.of(Shape.of(queries, d), q)), 1, 1)
                .value().toArray();
        int modernExact = 0;
        int classicalExact = 0;
        for (int i = 0; i < queries; i++) {
            boolean ok = true;
            boolean classicalOk = true;
            for (int j = 0; j < d; j++) {
                ok &= Math.signum(got[i * d + j]) == x[i * d + j];
                // classical: sign(W q) with W = sum x x^T
                double h = 0;
                for (int p = 0; p < n; p++) {
                    double overlap = 0;
                    for (int k = 0; k < d; k++) {
                        overlap += x[p * d + k] * q[i * d + k];
                    }
                    h += x[p * d + j] * overlap;
                }
                classicalOk &= Math.signum(h) == x[i * d + j];
            }
            modernExact += ok ? 1 : 0;
            classicalExact += classicalOk ? 1 : 0;
        }
        assertEquals(queries, modernExact);
        assertTrue(classicalExact < queries / 2, "classical " + classicalExact);
    }

    @Test
    void hopfieldPoolingSolvesMultipleInstanceLearning() {
        int items = 8;
        int f = 6;
        SplittableRandom random = new SplittableRandom(2);
        ModernHopfield pool = ModernHopfield.pooling(f, 2, 8, 0, random.split());
        Dense head = new Dense(16, 2, Activation.IDENTITY, Init.XAVIER, random.split());
        List<Var> params = new ArrayList<>(pool.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(0.02f);
        float[] witness = {2, -2, 2, -2, 2, -2};
        SplittableRandom data = new SplittableRandom(5);
        int batch = 16;
        double correct = 0;
        for (int step = 0; step < 220; step++) {
            float[] x = new float[batch * items * f];
            float[] y = new float[batch];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < items * f; i++) {
                    x[b * items * f + i] = (float) data.nextGaussian();
                }
                if (data.nextBoolean()) {
                    y[b] = 1;
                    int at = data.nextInt(items);
                    for (int k = 0; k < f; k++) {
                        x[(b * items + at) * f + k] = witness[k] + 0.3f * (float) data.nextGaussian();
                    }
                }
            }
            Var logits = head.forward(pool.forward(Var.of(Tensor.of(Shape.of(batch, items, f), x))));
            if (step < 200) {
                Functional.crossEntropy(logits, Tensor.of(y)).backward();
                opt.step(params);
            } else {
                float[] p = logits.value().argmax(1).toArray();
                for (int b = 0; b < batch; b++) {
                    correct += p[b] == y[b] ? 1 : 0;
                }
            }
        }
        assertTrue(correct / (20.0 * batch) > 0.9, "bag accuracy " + correct / (20.0 * batch));
    }

    @Test
    void hopfieldLayerWorksInAModel() {
        Model m = Model.sequential().input(5).add(ModernHopfield.layerSpec(16, 4)).dense(2, Activation.SOFTMAX).seed(1)
                .build();
        assertEquals(2, m.predict(new float[] {1, 2, 3, 4, 5}).length);
    }

    @Test
    void productKeyMemoryMemorizesWithSparseUpdates() {
        int n = 128;
        int f = 16;
        int classes = 8;
        SplittableRandom r = new SplittableRandom(3);
        Tensor x = Tensor.normal(Shape.of(n, f), 0, 1, r);
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            y[i] = r.nextInt(classes);
        }
        ProductKeyMemory pkm = new ProductKeyMemory(f, 16, 16, classes, 2, 4, new SplittableRandom(4));
        Optimizer opt = Optimizer.adam(0.05f);
        for (int step = 0; step < 150; step++) {
            Functional.crossEntropy(pkm.forward(Var.of(x)), Tensor.of(y)).backward();
            if (step == 0) {
                // only the rows that were read receive gradient: at most rows * heads * topK of 256
                float[] g = pkm.values().grad().abs().sum(1, false).toArray();
                int touched = 0;
                for (float v : g) {
                    touched += v > 0 ? 1 : 0;
                }
                assertTrue(touched <= n * 2 * 4 && touched < 256, "touched " + touched);
            }
            opt.step(pkm.parameters());
        }
        float[] p = pkm.forward(Var.of(x)).value().argmax(1).toArray();
        int correct = 0;
        for (int i = 0; i < n; i++) {
            correct += p[i] == y[i] ? 1 : 0;
        }
        assertTrue(correct > 0.95 * n, "memorized " + correct + " of " + n);
    }

    @Test
    void engramLooksUpWhatAUnigramCannotPredict() {
        assertTrue(engramAccuracy(new int[] {2}) > 0.95);
        assertTrue(engramAccuracy(new int[] {1}) < 0.6);
    }

    /** Next token = table[previous two tokens]; returns accuracy from n-gram lookups plus a linear readout. */
    private static double engramAccuracy(int[] orders) {
        int v = 6;
        int t = 12;
        SplittableRandom r = new SplittableRandom(7);
        int[][] rule = new int[v][v];
        for (int[] row : rule) {
            for (int j = 0; j < v; j++) {
                row[j] = r.nextInt(v);
            }
        }
        EngramMemory engram = new EngramMemory(orders, 2, 97, 16, 16, new SplittableRandom(8));
        Dense head = new Dense(16, v, Activation.IDENTITY, Init.XAVIER, new SplittableRandom(9));
        List<Var> params = new ArrayList<>(engram.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(0.03f);
        int batch = 8;
        double correct = 0;
        int total = 0;
        for (int step = 0; step < 220; step++) {
            float[] ids = new float[batch * t];
            float[] next = new float[batch * t];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < t; i++) {
                    ids[b * t + i] = r.nextInt(v);
                }
                for (int i = 0; i < t; i++) {
                    next[b * t + i] = i == 0 ? -1 : rule[(int) ids[b * t + i - 1]][(int) ids[b * t + i]];
                }
            }
            Var logits = head.forward(engram.forward(Var.of(Tensor.of(Shape.of(batch, t), ids))).reshape(batch * t,
                    16));
            if (step < 200) {
                Functional.crossEntropy(logits, Tensor.of(next)).backward();
                opt.step(params);
            } else {
                float[] p = logits.value().argmax(1).toArray();
                for (int i = 0; i < p.length; i++) {
                    if (next[i] >= 0) {
                        correct += p[i] == next[i] ? 1 : 0;
                        total++;
                    }
                }
            }
        }
        // fusing into a hidden state keeps the shape
        Var h = Var.of(Tensor.zeros(Shape.of(1, 3, 16)));
        assertEquals(Shape.of(1, 3, 16), engram.fuse(h, Var.of(Tensor.of(Shape.of(1, 3), 1, 2, 3))).shape());
        return correct / total;
    }
}
