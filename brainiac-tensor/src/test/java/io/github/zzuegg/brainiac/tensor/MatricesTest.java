package io.github.zzuegg.brainiac.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Host linear algebra against NumPy and PyTorch, and its own identities. */
class MatricesTest {

    private static final double[][] A = {{4, 1, 2, 0.5}, {0, 3, -1, 2}, {1, 0.5, 2, 1}, {2, -1, 0, 1}};

    private static void close(double[][] want, double[][] got, double tol) {
        for (int i = 0; i < want.length; i++) {
            assertArrayEquals(want[i], got[i], tol, "row " + i);
        }
    }

    @Test
    void luSolvesAndGivesTheDeterminant() {
        assertEquals(46, Matrices.determinant(A), 1e-10);
        double[] b = {1, 2, 3, 4};
        double[] x = Matrices.solve(A, b);
        for (int i = 0; i < 4; i++) {
            double s = 0;
            for (int j = 0; j < 4; j++) {
                s += A[i][j] * x[j];
            }
            assertEquals(b[i], s, 1e-12);
        }
        close(Matrices.identity(4), Matrices.multiply(A, Matrices.inverse(A)), 1e-12);
        assertEquals(0, Matrices.determinant(new double[][] {{1, 2}, {2, 4}}));
    }

    @Test
    void qrAndCholesky() {
        double[][] tall = {{1, 2}, {3, 4}, {5, 6}};
        Matrices.Qr f = Matrices.qr(tall);
        close(tall, Matrices.multiply(f.q(), f.r()), 1e-12);
        double[][] qtq = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    qtq[i][j] += f.q()[k][i] * f.q()[k][j];
                }
            }
        }
        close(Matrices.identity(2), qtq, 1e-12);
        assertEquals(0, f.r()[1][0]);
        double[][] spd = {{4, 2, 0.6}, {2, 5, 1}, {0.6, 1, 3}};
        double[][] l = Matrices.cholesky(spd);
        double[][] lt = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                lt[i][j] = l[j][i];
            }
        }
        close(spd, Matrices.multiply(l, lt), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> Matrices.cholesky(new double[][] {{1, 2}, {2, 1}}));
    }

    @Test
    void eigenvalues() {
        double[][] w = Matrices.eigenvalues(A);   // numpy.linalg.eigvals
        assertArrayEquals(new double[] {1.2986025764122435, 1.2986025764122435, 2.103356023067363, 5.299438824108151}, w[0], 1e-10);
        assertArrayEquals(new double[] {-1.562193284002903, 1.562193284002903, 0, 0}, w[1], 1e-10);
        double[][] rotation = Matrices.eigenvalues(new double[][] {{0, -2}, {2, 0}});
        assertArrayEquals(new double[] {0, 0}, rotation[0], 1e-12);
        assertArrayEquals(new double[] {-2, 2}, rotation[1], 1e-12);
        double[][] sym = {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
        Matrices.SymmetricEigen e = Matrices.symmetricEigen(sym);
        assertArrayEquals(new double[] {2 - Math.sqrt(2), 2, 2 + Math.sqrt(2)}, e.values(), 1e-12);
        // A V = V diag(values)
        double[][] av = Matrices.multiply(sym, e.vectors());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(e.vectors()[i][j] * e.values()[j], av[i][j], 1e-12);
            }
        }
    }

    @Test
    void matrixExponential() {
        double[] want = {137.2295322007489, 50.88819078033011, 68.42902936290014, 53.97791178346279, 24.035294712261347,
            16.37578752751562, 1.6485360151784523, 15.577288443965525, 61.395536139431826, 21.280705876408486,
            34.46787218210302, 25.552603486657116, 59.11697187740137, 15.281490076146772, 32.25706903196583,
            20.407892512533234};   // torch.linalg.matrix_exp
        double[][] e = Matrices.exp(A);
        for (int i = 0; i < 16; i++) {
            assertEquals(want[i], e[i / 4][i % 4], 1e-9 * Math.abs(want[i]));
        }
        // a rotation generator gives a rotation
        double[][] r = Matrices.exp(new double[][] {{0, -1}, {1, 0}});
        assertEquals(Math.cos(1), r[0][0], 1e-13);
        assertEquals(Math.sin(1), r[1][0], 1e-13);
    }

    @Test
    void lanczosFindsTheExtremePairsOfAPathLaplacian() {
        // the path graph's Laplacian has eigenvalues 2 - 2 cos(pi k / n)
        int n = 300;
        Matrices.Operator laplacian = (in, out) -> {
            for (int i = 0; i < n; i++) {
                double deg = (i > 0 ? 1 : 0) + (i < n - 1 ? 1 : 0);
                out[i] = deg * in[i] - (i > 0 ? in[i - 1] : 0) - (i < n - 1 ? in[i + 1] : 0);
            }
        };
        Matrices.Eigenpairs small = Matrices.lanczos(laplacian, n, 3, false, 300, 1);
        for (int k = 0; k < 3; k++) {
            assertEquals(2 - 2 * Math.cos(Math.PI * k / n), small.values()[k], 1e-8);
            assertTrue(small.residuals()[k] < 1e-6, "residual " + small.residuals()[k]);
        }
        Matrices.Eigenpairs big = Matrices.lanczos(laplacian, n, 2, true, n, 2);
        assertEquals(2 - 2 * Math.cos(Math.PI * (n - 1) / n), big.values()[0], 1e-6);
        // the random symmetric case against the dense solver
        SplittableRandom r = new SplittableRandom(3);
        double[][] s = new double[40][40];
        for (int i = 0; i < 40; i++) {
            for (int j = 0; j <= i; j++) {
                s[i][j] = s[j][i] = r.nextDouble(-1, 1);
            }
        }
        double[] all = Matrices.symmetricEigen(s).values();
        Matrices.Eigenpairs top = Matrices.lanczos((in, out) -> {
            double[][] col = new double[40][1];
            for (int i = 0; i < 40; i++) {
                col[i][0] = in[i];
            }
            double[][] p = Matrices.multiply(s, col);
            for (int i = 0; i < 40; i++) {
                out[i] = p[i][0];
            }
        }, 40, 2, true, 40, 4);
        assertEquals(all[39], top.values()[0], 1e-10);
        assertEquals(all[38], top.values()[1], 1e-10);
    }
}
