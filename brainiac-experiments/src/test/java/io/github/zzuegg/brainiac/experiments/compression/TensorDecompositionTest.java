package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.TensorDecomposition;
import io.github.zzuegg.brainiac.experimental.compress.TensorTrainDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TensorDecompositionTest {

    @Test
    void fullRankDecompositionsAreExact() {
        SplittableRandom r = new SplittableRandom(3);
        Tensor w = Tensor.normal(Shape.of(16, 16), 0, 1, r);
        Dense dense = new Dense(16, 16, Activation.IDENTITY, w, Tensor.zeros(Shape.of(16)));
        TensorTrainDense tt = TensorTrainDense.from(dense, new int[] {4, 4}, new int[] {4, 4}, 16);
        Tensor x = Tensor.normal(Shape.of(5, 16), 0, 1, r);
        float[] expect = x.matmul(w).toArray();
        assertArrayEquals(expect, tt.multiply(Var.of(x)).value().toArray(), 1e-3f);

        Tensor k = Tensor.normal(Shape.of(6, 4, 3, 3), 0, 1, r);
        TensorDecomposition.Tucker2 tk = TensorDecomposition.tucker2(k, 6, 4);
        assertArrayEquals(k.toArray(), tk.rebuild().toArray(), 1e-3f);

        double[] x3 = new double[2 * 2 * 2];
        TensorDecomposition.Cp exact = TensorDecomposition.cp(
                TensorDecomposition.cp(randomVector(8, r), new int[] {2, 2, 2}, 1, 30, 1).rebuild(),
                new int[] {2, 2, 2}, 1, 30, 2);
        double[] target = exact.rebuild();
        System.arraycopy(target, 0, x3, 0, 8);
        double[] again = TensorDecomposition.cp(x3, new int[] {2, 2, 2}, 1, 30, 5).rebuild();
        for (int i = 0; i < 8; i++) {
            assertTrue(Math.abs(again[i] - x3[i]) < 1e-4, "rank-1 CP recovered");
        }
    }

    private static double[] randomVector(int n, SplittableRandom r) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.nextGaussian();
        }
        return v;
    }

    @Test
    void tensorTrainAgainstSvd() {
        List<CompactRow> dense = CompactModels.denseDecompositions(8, 80, 30, 1);
        print("8.5.12 MLP 256x256 middle (acc @weights, ms):", dense);
        for (CompactRow row : dense) {
            if (row.method().endsWith("+ft")) {
                assertTrue(row.metric() > 0.3, "fine-tuned decomposition works: " + row);
            }
        }
    }

    static void print(String head, List<CompactRow> rows) {
        StringBuilder line = new StringBuilder(head);
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f @%.0f %.2fms;", r.method(), r.metric(), r.size(), r.millis()));
        }
        System.out.println(line);
    }
}
