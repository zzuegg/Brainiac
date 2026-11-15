package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class GraphTest {

    private static final Tensor X = Tensor.uniform(Shape.of(5, 4), -1, 1, new SplittableRandom(9));

    @Test
    void chainGraphMatchesSequentialModel() {
        Node in = Node.input(4);
        Model graph = Model.graph(in, in.dense(6, Activation.TANH).dense(2)).seed(3).build();
        Model seq = Model.sequential().input(4).dense(6, Activation.TANH).dense(2).seed(3).build();
        assertArrayEquals(seq.predict(X).toArray(), graph.predict(X).toArray());
        assertEquals(seq.summary(), graph.summary());
    }

    @Test
    void skipConnectionAddsTheInputBack() {
        Node in = Node.input(4);
        Dense zero = new Dense(4, 4, Activation.IDENTITY, Tensor.zeros(Shape.of(4, 4)), Tensor.zeros(Shape.of(4)));
        Model m = Model.graph(in, in.layer(zero).add(in)).build();
        assertArrayEquals(X.toArray(), m.predict(X).toArray(), 1e-7f);
    }

    @Test
    void concatJoinsBranchesSideBySide() {
        Node in = Node.input(4);
        Node a = in.dense(3);
        Node b = in.dense(2, Activation.TANH);
        Node joined = a.concat(b, in);
        assertEquals(9, joined.width());
        Model m = Model.graph(in, joined).seed(5).build();
        Tensor out = m.predict(X);
        assertEquals(Shape.of(5, 9), out.shape());
        assertArrayEquals(X.toArray(), out.slice(1, 5, 9).toArray());
        assertEquals("""
                Input(4)
                #1 Dense(4 -> 3, identity) <- input  params: 15
                #2 Dense(4 -> 2, tanh) <- input  params: 10
                #3 Concat <- #1, #2, input
                Total params: 25""", m.summary());
    }

    @Test
    void sharedLayerIsCountedOnce() {
        Dense shared = new Dense(3, 3, Activation.TANH, Init.XAVIER, new SplittableRandom(1));
        Node in = Node.input(3);
        Model m = Model.graph(in, in.layer(shared).layer(shared)).build();
        assertEquals(1, m.layers().size());
        assertEquals(12, m.parameterCount());
        Tensor once = shared.forward(Var.of(X.slice(1, 0, 3))).value();
        Tensor twice = shared.forward(Var.of(once)).value();
        assertArrayEquals(twice.toArray(), m.predict(X.slice(1, 0, 3)).toArray(), 1e-6f);
    }

    @Test
    void gradientsFlowThroughAllBranches() {
        Node in = Node.input(4);
        Node h = in.dense(4, Activation.TANH);
        Model m = Model.graph(in, h.dense(4).add(h).concat(in).dense(1)).seed(2).build();
        Var loss = m.forward(Var.of(X), true).square().mean();
        loss.backward();
        for (Var p : m.parameters()) {
            assertTrue(p.grad() != null && p.grad().abs().sum().item() > 0, "no gradient for a parameter");
        }
    }

    @Test
    void sliceAndConcatGradientsMatchFiniteDifferences() {
        Tensor w = Tensor.uniform(Shape.of(5, 6), -1, 1, new SplittableRandom(4));
        GradCheck.Result r = GradCheck.check(
                v -> Var.concat(1, v.get(0).slice(1, 1, 3), v.get(1)).mul(w).sum(),
                List.of(X, X));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void rejectsInvalidGraphs() {
        Node in = Node.input(4);
        Node other = Node.input(4);
        assertThrows(IllegalArgumentException.class, () -> in.dense(3).add(in));
        assertThrows(IllegalArgumentException.class, () -> Model.graph(in, other.dense(2)).build());
        assertThrows(IllegalArgumentException.class, () -> Model.graph(in.dense(2), in.dense(2)));
        assertThrows(IllegalStateException.class, () -> Model.graph(in, in).build());
    }
}
