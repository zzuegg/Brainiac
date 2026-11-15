package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LearnableActivationTest {

    private static final Tensor X = Tensor.uniform(Shape.of(5, 4), -2, 2, new SplittableRandom(1));

    @TempDir
    Path dir;

    @Test
    void preluComputesPerUnitSlopes() {
        Activation prelu = Activation.prelu(0.5f).bind(2);
        prelu.parameters().get(0).assign(Tensor.of(0.5f, 0.1f));
        float[] y = prelu.forward(Var.of(Tensor.of(Shape.of(2, 2), -2, -2, 3, 3))).value().toArray();
        assertArrayEquals(new float[] {-1f, -0.2f, 3f, 3f}, y, 1e-6f);
    }

    @Test
    void swishStartsAsSilu() {
        Var x = Var.of(X);
        float[] expected = x.mul(x.sigmoid()).value().toArray();
        assertArrayEquals(expected, Activation.swish().bind(4).forward(x).value().toArray(), 1e-6f);
    }

    @Test
    void parameterGradientsMatchFiniteDifferences() {
        for (Activation template : List.of(Activation.prelu(), Activation.swish(0.7f))) {
            Activation a = template.bind(4);
            Var p = a.parameters().get(0);
            p.assign(Tensor.uniform(Shape.of(4), 0.2f, 1.5f, new SplittableRandom(4)));
            Tensor weights = Tensor.uniform(Shape.of(5, 4), -1, 1, new SplittableRandom(5));
            Supplier<Var> loss = () -> a.forward(Var.of(X)).mul(weights).sum();
            p.zeroGrad();
            loss.get().backward();
            float[] analytic = p.grad().toArray();
            float[] values = p.value().toArray();
            for (int i = 0; i < values.length; i++) {
                float h = 1e-2f;
                float[] plus = values.clone();
                plus[i] += h;
                p.assign(Tensor.of(plus));
                float up = loss.get().value().item();
                float[] minus = values.clone();
                minus[i] -= h;
                p.assign(Tensor.of(minus));
                float down = loss.get().value().item();
                p.assign(Tensor.of(values));
                assertEquals((up - down) / (2 * h), analytic[i], 2e-2, template + " parameter " + i);
            }
        }
    }

    @Test
    void eachLayerGetsItsOwnParameters() {
        Activation prelu = Activation.prelu();
        Model m = Model.sequential().input(4).dense(8, prelu).dense(3, prelu).seed(1).build();
        Dense first = (Dense) m.layers().get(0);
        Dense second = (Dense) m.layers().get(1);
        assertNotSame(first.activation(), second.activation());
        assertEquals(Shape.of(8), first.activation().parameters().get(0).shape());
        assertEquals(Shape.of(3), second.activation().parameters().get(0).shape());
        // weights + biases + one slope per unit
        assertEquals(4 * 8 + 8 + 8 + 8 * 3 + 3 + 3, m.parameterCount());
        assertTrue(m.parameters().contains(first.activation().parameters().get(0)));
        assertTrue(m.summary().contains("params: 48"), m.summary());
    }

    @Test
    void bindingTheSameCopyTwiceSharesIt() {
        Activation shared = Activation.prelu().bind(4);
        assertSame(shared, shared.bind(4));
        Node in = Node.input(4);
        Node out = in.dense(4, shared).dense(4, shared);
        Model m = Model.graph(in, out).seed(2).build();
        // two weight matrices and biases, but only one set of slopes
        assertEquals(4 * 4 * 2 + 4 * 2 + 4, m.parameterCount());
        assertEquals(5, m.parameters().size());
        assertThrows(IllegalArgumentException.class, () -> shared.bind(5));
    }

    @Test
    void standaloneActivationStepsAreLearnable() {
        Model seq = Model.sequential().input(4).dense(6).activation(Activation.prelu()).seed(3).build();
        assertEquals(4 * 6 + 6 + 6, seq.parameterCount());

        Node in = Node.input(4);
        Model graph = Model.graph(in, in.activation(Activation.swish()).dense(2)).seed(3).build();
        assertEquals(4 + 4 * 2 + 2, graph.parameterCount());
    }

    @Test
    void trainingMovesTheActivationParameters() {
        Model m = Model.sequential().input(4).dense(3, Activation.prelu()).seed(4).build();
        Var slope = ((Dense) m.layers().get(0)).activation().parameters().get(0);
        Tensor target = Tensor.uniform(Shape.of(5, 3), -1, 1, new SplittableRandom(6));
        Loss.MSE.compute(m.forward(Var.of(X), true), target).backward();
        assertTrue(slope.grad().abs().sum().item() > 0, "slope got no gradient");
    }

    @Test
    void unboundTemplateCannotRun() {
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> Activation.prelu().forward(Var.of(X)));
        assertTrue(e.getMessage().contains("bind"), e.getMessage());
    }

    @Test
    void customLearnableActivation() {
        Activation scaledTanh = Activation.withParameters("scaled-tanh",
                width -> List.of(Tensor.full(Shape.of(1), 2f)),
                (x, p) -> x.tanh().mul(p.get(0)));
        Model m = Model.sequential().input(4).dense(4, scaledTanh).seed(5).build();
        assertEquals(4 * 4 + 4 + 1, m.parameterCount());
        assertTrue(m.summary().contains("scaled-tanh"), m.summary());
    }

    @Test
    void learnedValuesSurviveLoadWeights() throws IOException {
        Model m = Model.sequential().input(4).dense(3, Activation.prelu()).seed(6).build();
        Var slope = ((Dense) m.layers().get(0)).activation().parameters().get(0);
        slope.assign(Tensor.of(0.1f, 0.2f, 0.3f));
        Path file = dir.resolve("prelu.brainiac");
        m.save(file);
        Model fresh = Model.sequential().input(4).dense(3, Activation.prelu()).seed(7).build();
        fresh.loadWeights(file);
        assertArrayEquals(m.predict(X).toArray(), fresh.predict(X).toArray());
    }
}
