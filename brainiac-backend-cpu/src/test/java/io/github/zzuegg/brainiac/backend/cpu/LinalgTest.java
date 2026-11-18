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
    void svdMatchesPyTorchAndReconstructs() {
        Tensor a = Tensor.of(Shape.of(2, 4, 3), new double[] {-2.31041, -0.37325, -1.06082, 0.99951, -0.88403,
            -1.27555, -0.62322, -0.86644, 0.23872, 0.73353, -1.32188, 1.31481, -1.4875, -0.71358, 0.52, -0.95289,
            0.67486, 1.04687, 0.1543, -0.74989, -0.72339, 0.08637, -1.7464, 1.04101}, DType.FLOAT64, Device.cpu());
        // torch.linalg.svdvals
        double[] values = {2.89577, 1.89556, 1.80051, 2.34970, 2.00386, 1.07962};
        for (Tensor m : new Tensor[] {a, a.transpose().add(0f)}) {
            Tensor[] usv = Linalg.svd(m);
            int rows = m.shape().dim(1);
            int cols = m.shape().dim(2);
            assertEquals(Shape.of(2, rows, 3), usv[0].shape());
            assertEquals(Shape.of(2, 3), usv[1].shape());
            assertEquals(Shape.of(2, 3, cols), usv[2].shape());
            assertArrayEquals(values, usv[1].toDoubleArray(), 1e-5);
            close(m, usv[0].mul(usv[1].reshape(2, 1, 3)).matmul(usv[2]), 1e-10);
            Tensor eye = Linalg.identity(3, DType.FLOAT64, a.device()).expand(Shape.of(2, 3, 3)).add(0f);
            close(eye, usv[0].transpose().matmul(usv[0]), 1e-10);
            close(eye, usv[2].matmul(usv[2].transpose()), 1e-10);
        }
        // rank-deficient: one singular value is zero
        Tensor low = random(5, 2).matmul(random(2, 4));
        Tensor[] usv = Linalg.svd(low);
        assertEquals(0, usv[1].toDoubleArray()[3], 1e-10);
        close(low, usv[0].mul(usv[1].reshape(1, 4)).matmul(usv[2]), 1e-10);
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

    @Test
    void logDeterminants() {
        Tensor a = Tensor.of(Shape.of(2, 2, 2), new double[] {2, 1, 1, 3, 0, 1, 1, 0}, DType.FLOAT64,
                Device.getDefault());
        Tensor[] sl = Linalg.slogdet(a);
        assertArrayEquals(new double[] {1, -1}, sl[0].toDoubleArray());
        assertEquals(Math.log(5), sl[1].toDoubleArray()[0], 1e-12);
        assertEquals(0, sl[1].toDoubleArray()[1], 1e-12);
        Tensor singular = Tensor.of(Shape.of(2, 2), new double[] {1, 2, 2, 4}, DType.FLOAT64, Device.getDefault());
        assertEquals(0, Linalg.slogdet(singular)[0].item());
        // 60 x 60 with determinant 1e300 overflows as a product but not as a log
        Tensor big = Linalg.identity(60, DType.FLOAT64, Device.getDefault()).mul(1e5f);
        assertEquals(60 * Math.log(1e5), Linalg.slogdet(big)[1].toDoubleArray()[0], 1e-6);
        Tensor p = spd(3, 5);
        assertArrayEquals(Linalg.slogdet(p)[1].toDoubleArray(), Linalg.logdetPositiveDefinite(p).toDoubleArray(),
                1e-9);
    }

    @Test
    void linearAssignmentFindsTheCheapestMatching() {
        Tensor cost = Tensor.of(Shape.of(3, 3), 4, 1, 3, 2, 0, 5, 3, 2, 2);
        assertArrayEquals(new int[] {1, 0, 2}, Linalg.linearAssignment(cost));
        // against brute force on random rectangular problems
        for (int trial = 0; trial < 20; trial++) {
            Tensor c = Tensor.uniform(Shape.of(4, 6), 0, 1, R);
            int[] got = Linalg.linearAssignment(c);
            double[] v = c.toDoubleArray();
            double best = bruteForce(v, 6, 0, new boolean[6]);
            double sum = 0;
            java.util.Set<Integer> cols = new java.util.HashSet<>();
            for (int r = 0; r < 4; r++) {
                sum += v[r * 6 + got[r]];
                cols.add(got[r]);
            }
            assertEquals(4, cols.size());
            assertEquals(best, sum, 1e-9);
        }
        assertThrows(IllegalArgumentException.class, () -> Linalg.linearAssignment(Tensor.zeros(Shape.of(3, 2))));
    }

    private static double bruteForce(double[] c, int m, int row, boolean[] used) {
        if (row == c.length / m) {
            return 0;
        }
        double best = Double.POSITIVE_INFINITY;
        for (int j = 0; j < m; j++) {
            if (!used[j]) {
                used[j] = true;
                best = Math.min(best, c[row * m + j] + bruteForce(c, m, row + 1, used));
                used[j] = false;
            }
        }
        return best;
    }

    @Test
    void gammaFunctions() {
        Tensor x = Tensor.of(Shape.of(4), new double[] {0.5, 10, -0.5, 1}, DType.FLOAT64, Device.getDefault());
        assertArrayEquals(new double[] {0.5723649429247001, 12.801827480081469, 1.2655121234846454, 0},
                io.github.zzuegg.brainiac.tensor.Special.lgamma(x).toDoubleArray(), 1e-12);
        assertArrayEquals(new double[] {-1.9635100260214235, 2.251752589066721, 0.03648997397857652,
            -0.5772156649015329}, io.github.zzuegg.brainiac.tensor.Special.digamma(x).toDoubleArray(), 1e-12);
        assertArrayEquals(new double[] {Math.PI * Math.PI / 2, 0.10516633568168575, 8.934802200544704,
            Math.PI * Math.PI / 6}, io.github.zzuegg.brainiac.tensor.Special.trigamma(x).toDoubleArray(), 1e-8);
        assertEquals(Double.POSITIVE_INFINITY, io.github.zzuegg.brainiac.tensor.Special.lgamma(-2.0));
    }

    @Test
    void reshapeInfersOneDimension() {
        Tensor t = Tensor.zeros(Shape.of(2, 3, 4));
        assertEquals(Shape.of(6, 4), t.reshape(-1, 4).shape());
        assertEquals(Shape.of(24), t.reshape(-1).shape());
        assertThrows(IllegalArgumentException.class, () -> t.reshape(-1, 5));
        assertThrows(IllegalArgumentException.class, () -> t.reshape(-1, -1));
        assertEquals(Shape.of(2, 7, 4), t.shape().withDim(-2, 7));
    }
}
