package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Reading and replacing what each step of a graph model computes. */
class ForwardHookTest {

    private static final Tensor X = Tensor.uniform(Shape.of(5, 4), -1, 1, new SplittableRandom(9));

    @Test
    void graphListsEveryStepWithItsInputs() {
        Node in = Node.input(4);
        Dense shared = new Dense(4, 4, Activation.TANH, Init.XAVIER, new SplittableRandom(1));
        Node a = in.layer(shared);
        Node b = a.layer(shared);
        Model m = Model.graph(in, b.add(in)).build();
        List<Model.GraphStep> g = m.graph();
        assertEquals(3, g.size());
        assertEquals(Model.StepKind.LAYER, g.get(0).kind());
        assertSame(shared, g.get(0).layer());
        assertSame(shared, g.get(1).layer());   // every use, unlike layers()
        assertEquals(List.of(1), g.get(1).inputs());
        assertEquals(Model.StepKind.ADD, g.get(2).kind());
        assertEquals(List.of(2, 0), g.get(2).inputs());
        assertNull(g.get(2).layer());
        assertEquals(1, m.layers().size());
    }

    @Test
    void observingChangesNothing() {
        Model m = Model.sequential().input(4).dense(6, Activation.TANH).dense(2).seed(3).build();
        List<Tensor> seen = new ArrayList<>();
        Var out = m.forward(Var.of(X), false, (step, value) -> {
            seen.add(value.value());
            return value;
        });
        assertEquals(2, seen.size());
        assertEquals(Shape.of(5, 6), seen.get(0).shape());
        assertArrayEquals(m.predict(X).toArray(), out.value().toArray());
        assertArrayEquals(out.value().toArray(), seen.get(1).toArray());
    }

    @Test
    void patchingReplacesAStepsValue() {
        Node in = Node.input(4);
        Node hidden = in.dense(4, Activation.TANH);
        Model m = Model.graph(in, hidden.add(in)).seed(2).build();
        // zero the hidden branch: the residual leaves only the input
        Var out = m.forward(Var.of(X), false, (step, value) ->
                step.index() == 0 ? Var.of(Tensor.zeros(value.shape())) : value);
        assertArrayEquals(X.toArray(), out.value().toArray(), 1e-7f);
    }
}
