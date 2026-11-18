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
}
