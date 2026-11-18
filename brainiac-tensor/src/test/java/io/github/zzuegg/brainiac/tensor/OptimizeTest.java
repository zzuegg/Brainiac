package io.github.zzuegg.brainiac.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Nelder–Mead and the simplex method on textbook problems. */
class OptimizeTest {

    @Test
    void nelderMeadFindsTheRosenbrockMinimum() {
        Optimize.Minimum m = Optimize.nelderMead(x -> Math.pow(1 - x[0], 2) + 100 * Math.pow(x[1] - x[0] * x[0], 2),
                new double[] {-1.2, 1}, 0.1, 5000, 1e-10);
        assertTrue(m.converged());
        assertArrayEquals(new double[] {1, 1}, m.x(), 1e-4);
        Optimize.Minimum q = Optimize.nelderMead(x -> {
            double s = 0;
            for (int i = 0; i < x.length; i++) {
                s += (i + 1) * (x[i] - i) * (x[i] - i);
            }
            return s;
        }, new double[6], 0.5, 20_000, 1e-10);
        assertArrayEquals(new double[] {0, 1, 2, 3, 4, 5}, q.x(), 1e-3);
    }

    @Test
    void linearPrograms() {
        // scipy.optimize.linprog docs: min -x0 + 4 x1, -3 x0 + x1 <= 6, x0 + 2 x1 <= 4, x1 >= -3 (x1 = y - 3)
        // with x1 shifted: x1 = y - 3, y >= 0: min -x0 + 4y - 12, -3x0 + y <= 9, x0 + 2y <= 10
        Optimize.LinearSolution s = Optimize.linearProgram(new double[] {-1, 4},
                new double[][] {{-3, 1}, {1, 2}}, new double[] {9, 10}, null, null);
        assertEquals(Optimize.Status.OPTIMAL, s.status());
        assertEquals(-10 - 12, s.value() - 12, 1e-9);   // linprog gives -22 at x = (10, -3)
        assertArrayEquals(new double[] {10, 0}, s.x(), 1e-9);
        // equality and a negative right-hand side: min x + y, x + y = 2, x - y >= 1 (as -x + y <= -1)
        Optimize.LinearSolution e = Optimize.linearProgram(new double[] {1, 2},
                new double[][] {{-1, 1}}, new double[] {-1}, new double[][] {{1, 1}}, new double[] {2});
        assertEquals(Optimize.Status.OPTIMAL, e.status());
        assertArrayEquals(new double[] {2, 0}, e.x(), 1e-9);
        assertEquals(Optimize.Status.INFEASIBLE, Optimize.linearProgram(new double[] {1},
                new double[][] {{1}}, new double[] {-1}, null, null).status());
        assertEquals(Optimize.Status.UNBOUNDED, Optimize.linearProgram(new double[] {-1, 0},
                new double[][] {{-1, 1}}, new double[] {1}, null, null).status());
    }

    @Test
    void cmaEsSolvesRosenbrockAndARotatedEllipsoid() {
        Optimize.Minimum r = Optimize.cmaEs(x -> {
            double s = 0;
            for (int i = 0; i + 1 < x.length; i++) {
                s += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
            }
            return s;
        }, new double[5], 0.5, 3000, 1e-12, 1);
        assertArrayEquals(new double[] {1, 1, 1, 1, 1}, r.x(), 1e-4);
        // badly scaled and rotated: f = sum 10^(6 i/(n-1)) (sum_j<=i x_j)^2
        Optimize.Minimum e = Optimize.cmaEs(x -> {
            double s = 0;
            double prefix = 0;
            for (int i = 0; i < x.length; i++) {
                prefix += x[i] - 2;
                s += Math.pow(10, 6.0 * i / (x.length - 1)) * prefix * prefix;
            }
            return s;
        }, new double[6], 1, 3000, 1e-14, 2);
        assertTrue(e.converged(), e.toString());
        assertArrayEquals(new double[] {2, 2, 2, 2, 2, 2}, e.x(), 1e-4);
    }

    @Test
    void badlyScaledCuttingPlanesStaySolvable() {
        // RiskSLIM-like: a box-bounded variable set, cuts whose coefficients span 16 decades
        java.util.SplittableRandom r = new java.util.SplittableRandom(5);
        for (int trial = 0; trial < 150; trial++) {
            int n = 6;
            int cuts = 30;
            double[] feasible = new double[n];
            for (int j = 0; j < n; j++) {
                feasible[j] = r.nextDouble(0, 3);
            }
            double[][] a = new double[cuts + n][n];
            double[] b = new double[cuts + n];
            for (int i = 0; i < cuts; i++) {
                double dot = 0;
                for (int j = 0; j < n; j++) {
                    a[i][j] = (r.nextBoolean() ? 1 : -1) * Math.pow(10, r.nextDouble(-8, 8));
                    dot += a[i][j] * feasible[j];
                }
                b[i] = dot + Math.abs(dot) * 1e-3 * r.nextDouble();
            }
            for (int j = 0; j < n; j++) {
                a[cuts + j][j] = 1;
                b[cuts + j] = 10;
            }
            double[] c = new double[n];
            for (int j = 0; j < n; j++) {
                c[j] = r.nextDouble(-1, 1) * Math.pow(10, r.nextDouble(-3, 3));
            }
            Optimize.LinearSolution s = Optimize.linearProgram(c, a, b, null, null);
            assertEquals(Optimize.Status.OPTIMAL, s.status(), "trial " + trial);
            double atFeasible = 0;
            for (int j = 0; j < n; j++) {
                atFeasible += c[j] * feasible[j];
                assertTrue(s.x()[j] >= 0);
            }
            assertTrue(s.value() <= atFeasible + 1e-6 * Math.max(1, Math.abs(atFeasible)), "trial " + trial);
            for (int i = 0; i < a.length; i++) {
                double dot = 0;
                double scale = 0;
                for (int j = 0; j < n; j++) {
                    dot += a[i][j] * s.x()[j];
                    scale = Math.max(scale, Math.abs(a[i][j] * s.x()[j]));
                }
                assertTrue(dot <= b[i] + 1e-7 * Math.max(1, Math.max(scale, Math.abs(b[i]))), "trial " + trial + " row " + i + " dot " + dot + " b " + b[i] + " scale " + scale + " x " + java.util.Arrays.toString(s.x()) + " a " + java.util.Arrays.toString(a[i]));
            }
        }
    }

    @Test
    void quadraticPrograms() {
        // the OSQP README problem: min x'Px/2 + q'x, P = [[4,1],[1,2]], q = [1,1],
        // 1 <= x0 + x1 <= 1, 0 <= x0 <= 0.7, 0 <= x1 <= 0.7; solution (0.3, 0.7)
        Optimize.QuadraticSolution s = Optimize.quadraticProgram(new double[][] {{4, 1}, {1, 2}}, new double[] {1, 1},
                new double[][] {{1, 1}, {1, 0}, {0, 1}}, new double[] {1, 0, 0}, new double[] {1, 0.7, 0.7}, 1e-9, 20000);
        assertTrue(s.converged());
        assertArrayEquals(new double[] {0.3, 0.7}, s.x(), 1e-6);
        assertEquals(1.88, s.value(), 1e-6);
        // on the probability simplex (as EasyMKL): min |x - c|^2 / 2 is the projection of c
        double[] c = {0.9, 0.5, -0.2, 0.1};
        double[][] eye = Matrices.identity(4);
        double[][] rows = new double[5][];
        rows[0] = new double[] {1, 1, 1, 1};
        System.arraycopy(eye, 0, rows, 1, 4);
        double inf = Double.POSITIVE_INFINITY;
        Optimize.QuadraticSolution proj = Optimize.quadraticProgram(eye, new double[] {-0.9, -0.5, 0.2, -0.1}, rows,
                new double[] {1, 0, 0, 0, 0}, new double[] {1, inf, inf, inf, inf}, 1e-10, 20000);
        // the projection shifts by tau = 0.2: (0.7, 0.3, 0, 0)
        assertArrayEquals(new double[] {0.7, 0.3, 0, 0}, proj.x(), 1e-6);
        // unconstrained: P x = -q
        Optimize.QuadraticSolution free = Optimize.quadraticProgram(new double[][] {{2, 0}, {0, 4}}, new double[] {-2, -8},
                null, null, null, 1e-10, 1000);
        assertArrayEquals(new double[] {1, 2}, free.x(), 1e-6);
    }
}
