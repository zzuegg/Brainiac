package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class GraphsTest {

    /** A path 0 - 1 - 2, both directions. */
    private static Graphs.Graph path(Tensor x) {
        return new Graphs.Graph(x, Tensor.of(new float[] {0, 1, 1, 2}), Tensor.of(new float[] {1, 0, 2, 1}));
    }

    @Test
    void propagateAndGcnByHand() {
        Tensor x = Tensor.of(Shape.of(3, 1), 1, 10, 100);
        Graphs.Graph g = path(x);
        assertArrayEquals(new float[] {10, 101, 10}, Graphs.propagate(Var.of(x), g.sources(), g.targets())
                .value().toArray());
        // degrees with self loops: 2, 3, 2
        Var out = Graphs.gcn(g, Var.of(x), Var.of(Tensor.of(Shape.of(1, 1), 1)), null);
        double n0 = 1 / 2.0 * 1 + 1 / Math.sqrt(6) * 10;
        double n1 = 1 / Math.sqrt(6) * 1 + 1 / 3.0 * 10 + 1 / Math.sqrt(6) * 100;
        assertArrayEquals(new float[] {(float) n0, (float) n1}, java.util.Arrays.copyOf(out.value().toArray(), 2), 1e-4f);
    }

    @Test
    void sageAndGatHaveGradients() {
        SplittableRandom r = new SplittableRandom(1);
        Tensor x = Tensor.normal(Shape.of(3, 2), 0, 1, r).to(DType.FLOAT64);
        Graphs.Graph g = path(x);
        List<Tensor> params = List.of(x, Tensor.normal(Shape.of(2, 3), 0, 1, r).to(DType.FLOAT64),
                Tensor.normal(Shape.of(2, 3), 0, 1, r).to(DType.FLOAT64), Tensor.normal(Shape.of(3, 1), 0, 1, r)
                        .to(DType.FLOAT64), Tensor.normal(Shape.of(3, 1), 0, 1, r).to(DType.FLOAT64));
        GradCheck.Result s = GradCheck.check(v -> Graphs.sage(g, v.get(0), v.get(1), v.get(2), null).square().sum(),
                params.subList(0, 3));
        assertTrue(s.passed(), s.toString());
        GradCheck.Result a = GradCheck.check(v -> Graphs.gat(g, v.get(0), v.get(1), v.get(3), v.get(4)).square().sum(),
                params);
        assertTrue(a.passed(), a.toString());
        // attention weights of each node sum to one: with h = x, outputs are convex combinations
        Var h = Graphs.gat(g, Var.of(Tensor.of(Shape.of(3, 1), 1, 1, 1)), Var.of(Tensor.of(Shape.of(1, 1), 1)),
                Var.of(Tensor.of(Shape.of(1, 1), 0.3f)), Var.of(Tensor.of(Shape.of(1, 1), -0.2f)));
        assertArrayEquals(new float[] {1, 1, 1}, h.value().toArray(), 1e-6f);
    }

    @Test
    void batchingShiftsEdgesAndPools() {
        Graphs.Graph a = path(Tensor.of(Shape.of(3, 1), 1, 2, 3));
        Graphs.Graph b = new Graphs.Graph(Tensor.of(Shape.of(2, 1), 10, 20), Tensor.of(new float[] {0}),
                Tensor.of(new float[] {1}));
        Graphs.Batch batch = Graphs.batch(List.of(a, b));
        assertEquals(5, batch.graph().size());
        assertArrayEquals(new float[] {0, 1, 1, 2, 3}, batch.graph().sources().toArray());
        assertArrayEquals(new float[] {1, 0, 2, 1, 4}, batch.graph().targets().toArray());
        assertArrayEquals(new float[] {2, 15}, Graphs.pool(Var.of(batch.graph().nodes()), batch, true).value().toArray());
        assertArrayEquals(new float[] {6, 30}, Graphs.pool(Var.of(batch.graph().nodes()), batch, false).value().toArray());
    }
}
