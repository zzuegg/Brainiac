package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

class ModelTest {

    private static Model xorShaped(long seed) {
        return Model.sequential()
                .input(2)
                .dense(8, Activation.TANH)
                .dense(1, Activation.SIGMOID)
                .seed(seed)
                .build();
    }

    @Test
    void buildsLayersWithMatchingSizes() {
        Model m = xorShaped(42);
        assertEquals(2, m.inputSize());
        assertEquals(1, m.outputSize());
        assertEquals(2, m.layers().size());
        assertEquals(4, m.parameters().size());
        assertEquals(2 * 8 + 8 + 8 * 1 + 1, m.parameterCount());
        assertEquals("""
                Input(2)
                Dense(2 -> 8, tanh)  params: 24
                Dense(8 -> 1, sigmoid)  params: 9
                Total params: 33""", m.summary());
    }

    @Test
    void predictsSingleSamplesAndBatches() {
        Model m = xorShaped(42);
        float[] one = m.predict(1, 0);
        assertEquals(1, one.length);
        assertTrue(one[0] > 0 && one[0] < 1);
        float[][] many = m.predict(new float[][] {{0, 0}, {1, 0}});
        assertEquals(2, many.length);
        assertEquals(one[0], many[1][0], 1e-6);
        assertEquals(Shape.of(2, 1), m.predict(Tensor.of(new float[][] {{0, 0}, {1, 1}})).shape());
        assertThrows(IllegalArgumentException.class, () -> m.predict(1, 2, 3));
    }

    @Test
    void sameSeedGivesSameWeights() {
        assertArrayEquals(xorShaped(7).predict(0.3f, -0.2f), xorShaped(7).predict(0.3f, -0.2f));
    }

    @Test
    void softmaxOutputsAreProbabilities() {
        Model m = Model.sequential().input(4).dense(16, Activation.RELU).dense(3, Activation.SOFTMAX).seed(1).build();
        float[] p = m.predict(0.5f, -1, 2, 0);
        assertEquals(1f, p[0] + p[1] + p[2], 1e-5);
        for (float v : p) {
            assertTrue(v > 0);
        }
    }

    @Test
    void acceptsActivationAndCustomLayers() {
        Layer doubler = new Layer() {
            @Override
            public Var forward(Var input) {
                return input.mul(2);
            }
        };
        Model m = Model.sequential().input(2).dense(3).layer(Activation.RELU).layer(doubler).seed(3).build();
        assertEquals(3, m.outputSize());
        assertEquals(3, m.layers().size());
    }

    @Test
    void forwardRecordsGradientsForAllParameters() {
        Model m = xorShaped(1);
        Var loss = Loss.MSE.compute(m.forward(Var.of(Tensor.of(new float[][] {{0, 1}, {1, 1}}))),
                Tensor.of(Shape.of(2, 1), 1, 0));
        loss.backward();
        for (Var p : m.parameters()) {
            assertNotNull(p.grad());
            assertEquals(p.shape(), p.grad().shape());
        }
    }

    @Test
    void explainsBuilderMistakes() {
        assertThrows(IllegalStateException.class, () -> Model.sequential().dense(3));
        assertThrows(IllegalStateException.class, () -> Model.sequential().input(2).build());
        assertThrows(IllegalStateException.class, () -> Model.sequential().input(2).input(3));
        assertThrows(IllegalArgumentException.class, () -> Model.sequential().input(0));
    }

    @Test
    void denseRejectsWrongInputWidth() {
        Dense d = new Dense(3, 2, Activation.RELU, Init.HE, new java.util.SplittableRandom(1));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> d.forward(Var.of(Tensor.zeros(Shape.of(4, 2)))));
        assertEquals("Dense(2) expects input [*, 3], got [4, 2]", e.getMessage());
    }

    @Test
    void lossesRejectMismatchedShapes() {
        Var p = Var.of(Tensor.zeros(Shape.of(2, 3)));
        for (Loss loss : List.of(Loss.MSE, Loss.CROSS_ENTROPY, Loss.BINARY_CROSS_ENTROPY)) {
            assertThrows(IllegalArgumentException.class, () -> loss.compute(p, Tensor.zeros(Shape.of(3))));
        }
    }

    @Test
    void crossEntropyIsLowForConfidentCorrectPredictions() {
        Tensor target = Tensor.of(Shape.of(1, 3), 0, 1, 0);
        float good = Loss.CROSS_ENTROPY.compute(Var.of(Tensor.of(Shape.of(1, 3), 0.01f, 0.98f, 0.01f)), target).value().item();
        float bad = Loss.CROSS_ENTROPY.compute(Var.of(Tensor.of(Shape.of(1, 3), 0.98f, 0.01f, 0.01f)), target).value().item();
        assertEquals(-Math.log(0.98), good, 1e-4);
        assertTrue(bad > 4);
    }
}
