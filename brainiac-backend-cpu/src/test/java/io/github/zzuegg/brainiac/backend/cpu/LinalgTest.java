package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class LinalgTest {

    private static final SplittableRandom R = new SplittableRandom(3);

    private static Tensor random(int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, R).to(DType.FLOAT64);
    }

    /** Batched symmetric positive definite matrices: m m^T + n I. */
    private static Tensor spd(int batch, int n) {
        Tensor m = random(batch, n, n);
        return m.matmul(m.transpose()).add(Linalg.identity(n, DType.FLOAT64, m.device()).mul(n));
    }

    private static void close(Tensor expected, Tensor actual, double tol) {
        assertEquals(expected.shape(), actual.shape());
        assertArrayEquals(expected.toDoubleArray(), actual.toDoubleArray(), tol);
    }

    @Test
    void eighReconstructsAndSorts() {
        Tensor a = random(2, 6, 6);
        a = a.add(a.transpose());
        Tensor[] eig = Linalg.eigh(a);
        assertEquals(Shape.of(2, 6), eig[0].shape());
        Tensor v = eig[1];
        close(a, v.mul(eig[0].reshape(2, 1, 6)).matmul(v.transpose()), 1e-10);
        close(Linalg.identity(6, DType.FLOAT64, a.device()).expand(Shape.of(2, 6, 6)).add(0f),
                v.transpose().matmul(v), 1e-10);
        double[] values = eig[0].toDoubleArray();
        for (int i = 1; i < 6; i++) {
            assertTrue(values[i - 1] <= values[i]);
        }
    }

    @Test
    void symmetricPowerGivesRoots() {
        Tensor a = spd(1, 5);
        Tensor root = Linalg.symmetricPower(a, 0.5, 0);
        close(a, root.matmul(root), 1e-10);
        Tensor quarter = Linalg.symmetricPower(a, -0.25, 1e-12);
        Tensor q4 = quarter.matmul(quarter).matmul(quarter).matmul(quarter);
        close(Linalg.inverse(a), q4, 1e-10);
    }

    @Test
    void choleskyAndTriangularSolves() {
        Tensor a = spd(3, 4);
        Tensor l = Linalg.cholesky(a);
        close(a, l.matmul(l.transpose()), 1e-12);
        assertEquals(0, l.toDoubleArray()[1]);   // above the diagonal
        Tensor b = random(3, 4, 2);
        Tensor y = Linalg.solveTriangular(l, b, true);
        close(b, l.matmul(y), 1e-12);
        Tensor u = l.transpose().add(0f);
        Tensor z = Linalg.solveTriangular(u, b, false);
        close(b, u.matmul(z), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(a.neg()));
    }

    @Test
    void solveInverseAndPowers() {
        Tensor a = random(2, 5, 5).add(Linalg.identity(5, DType.FLOAT64, Device.getDefault()).mul(3));
        Tensor b = random(2, 5, 3);
        close(b, a.matmul(Linalg.solve(a, b)), 1e-12);
        close(Linalg.identity(5, DType.FLOAT64, a.device()).expand(a.shape()).add(0f),
                a.matmul(Linalg.inverse(a)), 1e-12);
        close(a.matmul(a).matmul(a).matmul(a).matmul(a), Linalg.matrixPower(a, 5), 1e-9);
        close(Linalg.inverse(a.matmul(a)), Linalg.matrixPower(a, -2), 1e-10);
        close(Linalg.identity(5, DType.FLOAT64, a.device()).expand(a.shape()).add(0f), Linalg.matrixPower(a, 0), 0);
    }

    @Test
    void lstsqMatchesTheNormalEquationsAndRidge() {
        Tensor a = random(20, 4);
        Tensor xTrue = random(4, 2);
        Tensor b = a.matmul(xTrue);
        close(xTrue, Linalg.lstsq(a, b, 0), 1e-10);
        Tensor ridge = Linalg.lstsq(a, b, 0.5);
        Tensor gram = a.transpose().matmul(a).add(Linalg.identity(4, DType.FLOAT64, a.device()).mul(0.5f));
        close(a.transpose().matmul(b), gram.matmul(ridge), 1e-10);
    }
}
