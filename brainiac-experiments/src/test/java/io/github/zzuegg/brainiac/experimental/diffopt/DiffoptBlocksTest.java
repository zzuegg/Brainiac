package io.github.zzuegg.brainiac.experimental.diffopt;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

class DiffoptBlocksTest {

    @Test
    void qpLayerGradientsMatchFiniteDifferences() {
        // min 1/2 z^T Q z + q^T z  s.t.  G z <= h, A z = b, with one active inequality
        Tensor qm = Tensor.of(Shape.of(3, 3), 2f, 0.3f, 0f, 0.3f, 1.5f, 0.2f, 0f, 0.2f, 1f);
        Tensor q = Tensor.of(Shape.of(2, 3), -1f, -2f, 0.5f, 0.4f, -1f, -1.5f);
        Tensor g = Tensor.of(Shape.of(2, 3), 1f, 1f, 0f, 0f, -1f, 1f);
        Tensor h = Tensor.of(Shape.of(1, 2), 0.5f, 2f);
        Tensor a = Tensor.of(Shape.of(1, 3), 1f, -1f, 1f);
        Tensor b = Tensor.of(Shape.of(1, 1), 0.2f);
        Tensor weights = Tensor.of(Shape.of(2, 3), 0.3f, -1f, 0.7f, 1.1f, 0.2f, -0.4f);
        GradCheck.Result res = GradCheck.check(v -> QpLayer.solve(v.get(0), v.get(1), v.get(2), v.get(3), v.get(4),
                v.get(5)).mul(weights).sum(), List.of(qm, q, g, h, a, b), 1e-3f, 3e-2f);
        assertTrue(res.passed(), res.toString());
    }

    @Test
    void softSortsApproachTheHardSort() {
        Var s = Var.of(Tensor.of(Shape.of(1, 5), 0.3f, 2f, -1f, 0.9f, 1.5f));
        float[] sorted = {2f, 1.5f, 0.9f, 0.3f, -1f};
        for (SoftSorting.Method m : SoftSorting.Method.values()) {
            float[] v = SoftSorting.apply(SoftSorting.permutation(s, m, 0.01), s).value().toArray();
            assertArrayEquals(sorted, v, 0.05f, m.name());
        }
        assertArrayEquals(sorted, SoftSorting.fastSoftSort(s, 1e-3).value().toArray(), 1e-3f);
        assertArrayEquals(new float[] {2, 5, 1, 3, 4}, SoftSorting.fastSoftRank(s, 1e-3).value().toArray(), 1e-3f);
        GradCheck.Result res = GradCheck.check(v -> SoftSorting.fastSoftRank(v.get(0), 1.0)
                .mul(Tensor.of(Shape.of(1, 5), 1f, -2f, 0.5f, 3f, 1f)).sum(),
                List.of(Tensor.of(Shape.of(1, 5), 0.3f, 2f, -1f, 0.9f, 1.5f)));
        assertTrue(res.passed(), res.toString());
    }

    @Test
    void solversAndPerturbationsBehave() {
        double[] cost = {1, 9, 9, 1, 9, 9, 1, 1, 1};
        double[] path = PerturbedOptimizers.shortestPathCells(cost, 3, false);
        assertArrayEquals(new double[] {1, 0, 0, 1, 0, 0, 1, 1, 1}, path, 1e-12);
        double[] y = PerturbedOptimizers.ranking().solve(new double[] {0.1, 3, -2});
        assertArrayEquals(new double[] {2, 3, 1}, y, 1e-12);
        double[] m = PerturbedOptimizers.assignment(2).solve(new double[] {0, 5, 4, 0});
        assertArrayEquals(new double[] {0, 1, 1, 0}, m, 1e-12);
        double[] w = DecisionFocused.gridPaths(3).solve(new double[] {1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1});
        double total = 0;
        for (double v : w) {
            total += v;
        }
        assertEquals(4, total, 1e-12);
        double[] port = DecisionFocused.portfolio(4, 0.5).solve(new double[] {-1, -3, -2, 0});
        assertArrayEquals(new double[] {0, 0.5, 0.5, 0}, port, 1e-9);
    }
}
