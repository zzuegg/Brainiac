package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

/** Triangles, diagonals, cross and Kronecker products, logaddexp, xlogy and quantiles; values from PyTorch. */
class MatrixAndStatisticsOpsTest {

    private static final Var X = Var.of(Tensor.of(Shape.of(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 10));

    private static float[] v(Var x) {
        return x.value().toArray();
    }

    @Test
    void triangles() {
        assertArrayEquals(new float[] {1, 0, 0, 4, 5, 0, 7, 8, 10}, v(X.tril(0)));
        assertArrayEquals(new float[] {0, 2, 3, 0, 0, 6, 0, 0, 0}, v(X.triu(1)));
        assertArrayEquals(new float[] {0, 0, 0, 4, 0, 0, 7, 8, 0}, v(X.tril(-1)));
    }

    @Test
    void diagonalAndTrace() {
        assertArrayEquals(new float[] {0, 4}, v(Var.of(Tensor.of(Shape.of(2, 3), 0, 1, 2, 3, 4, 5)).diagonal()));
        assertEquals(16f, X.trace().value().item(), 0f);
        Var batch = Var.of(Tensor.of(Shape.of(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8));
        assertArrayEquals(new float[] {5, 13}, v(batch.trace()));
    }

    @Test
    void crossAndKronecker() {
        assertArrayEquals(new float[] {-3, 6, -3},
                v(Var.of(Tensor.of(1, 2, 3)).cross(Var.of(Tensor.of(4, 5, 6)), 0)));
        assertArrayEquals(new float[] {0, 1, 0, 2, 1, 0, 2, 0, 0, 3, 0, 4, 3, 0, 4, 0},
                v(Var.of(Tensor.of(Shape.of(2, 2), 1, 2, 3, 4)).kron(Var.of(Tensor.of(Shape.of(2, 2), 0, 1, 1, 0)))));
    }

    @Test
    void logaddexpAndXlogy() {
        assertArrayEquals(new float[] {1000.6931762695312f, 3.0067152976989746f},
                v(Var.of(Tensor.of(1000, -2)).logaddexp(Var.of(Tensor.of(1000, 3)))), 1e-4f);
        Var x = Var.param(Tensor.of(0, 2));
        Var y = Var.param(Tensor.of(0, 3));
        Var r = x.xlogy(y);
        assertArrayEquals(new float[] {0, 2.1972246170043945f}, v(r), 1e-6f);
        r.sum().backward();
        // d/dx = log y (0 where x is 0 by the convention), d/dy = x / y
        assertArrayEquals(new float[] {0, (float) Math.log(3)}, x.grad().toArray(), 1e-6f);
        assertArrayEquals(new float[] {0, 2f / 3}, y.grad().toArray(), 1e-6f);
    }

    @Test
    void quantilesAndMedians() {
        Var x = Var.of(Tensor.of(Shape.of(2, 5), 3, 1, 4, 1, 5, 9, 2, 6, 5, 3));
        assertArrayEquals(new float[] {1.4f, 3.4f}, v(x.quantile(0.3, 1, false)), 1e-6f);
        assertArrayEquals(new float[] {3, 5}, v(x.median(-1, false)));
        assertEquals(Shape.of(2, 1), x.median(1, true).shape());
        assertArrayEquals(new float[] {2.5f}, v(Var.of(Tensor.of(1, 2, 3, 4)).median(0, false)));
    }
}
