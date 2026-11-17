package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.Random;
import org.junit.jupiter.api.Test;

class SparseMatrixTest {

    private static Tensor random(int rows, int cols, Random r, double density) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < v.length; i++) {
            v[i] = r.nextDouble() < density ? (float) r.nextGaussian() : 0;
        }
        return Tensor.of(Shape.of(rows, cols), v);
    }

    @Test
    void productMatchesTheDenseOne() {
        Random r = new Random(1);
        Tensor a = random(7, 5, r, 0.3);
        Tensor b = random(5, 4, r, 1);
        SparseMatrix s = SparseMatrix.fromDense(a, 0);
        assertArrayEquals(a.toArray(), s.toDense().value().toArray(), 1e-6f);
        assertArrayEquals(a.matmul(b).toArray(), s.matmul(Var.of(b)).value().toArray(), 1e-5f);
        assertArrayEquals(a.transpose().matmul(random(7, 3, new Random(2), 1)).toArray(),
                s.transpose().matmul(Var.of(random(7, 3, new Random(2), 1))).value().toArray(), 1e-5f);
    }

    @Test
    void gradientsMatchTheDenseOnes() {
        Random r = new Random(3);
        Tensor a = random(6, 5, r, 0.4);
        Tensor b = random(5, 3, r, 1);
        SparseMatrix s = SparseMatrix.fromDense(a, 0);
        Var values = Var.param(s.values().value());
        Var bs = Var.param(b);
        s.withValues(values).matmul(bs).square().sum().backward();
        Var ad = Var.param(a);
        Var bd = Var.param(b);
        ad.matmul(bd).square().sum().backward();
        assertArrayEquals(bd.grad().toArray(), bs.grad().toArray(), 1e-4f);
        // the values' gradient is the dense gradient at the stored entries
        float[] dense = ad.grad().toArray();
        float[] rows = s.rowIds().toArray();
        float[] cols = s.colIds().toArray();
        float[] g = values.grad().toArray();
        for (int i = 0; i < g.length; i++) {
            assertEquals(dense[(int) rows[i] * 5 + (int) cols[i]], g[i], 1e-4f);
        }
    }

    @Test
    void repeatedEntriesAddUpAndEmptyWorks() {
        SparseMatrix s = SparseMatrix.of(2, 2, new int[] {0, 0}, new int[] {1, 1}, new float[] {1, 2},
                Tensor.zeros(Shape.of(1)).device());
        assertArrayEquals(new float[] {0, 3, 0, 0}, s.toDense().value().toArray());
        SparseMatrix empty = SparseMatrix.fromDense(Tensor.zeros(Shape.of(3, 2)), 0);
        assertEquals(1, empty.nonZeros());
        assertArrayEquals(new float[6], empty.matmul(Var.of(Tensor.ones(Shape.of(2, 2)))).value().toArray());
    }
}
