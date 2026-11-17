package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Neural Turing Machine and Differentiable Neural Computer. */
class MemoryMachinesTest {

    private static Var c(float... v) {
        return Var.of(Tensor.of(Shape.of(1, 1, v.length), v));
    }

    @Test
    void ntmAddressingFindsByContentThenShifts() {
        Var memory = Var.of(Tensor.of(Shape.of(1, 4, 3), 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0));
        Var previous = c(1, 0, 0, 0);
        Var key = c(0, 0, 1);
        Var strong = c(50);
        float[] content = NeuralTuringMachine.address(memory, key, strong, c(1), c(0, 1, 0), c(1), previous).value()
                .toArray();
        assertEquals(2, argmax(content));
        assertTrue(content[2] > 0.95f);
        float[] shifted = NeuralTuringMachine.address(memory, key, strong, c(1), c(0, 0, 1), c(1), previous).value()
                .toArray();
        assertEquals(3, argmax(shifted));
        // gate 0: ignore content, shift the previous weighting back by one (circularly)
        float[] located = NeuralTuringMachine.address(memory, key, strong, c(0), c(1, 0, 0), c(1), previous).value()
                .toArray();
        assertEquals(3, argmax(located));
        // sharpening makes a blurred weighting peakier
        float[] blurred = NeuralTuringMachine.address(memory, key, c(1), c(1), c(0.2f, 0.6f, 0.2f), c(1), previous)
                .value().toArray();
        float[] sharp = NeuralTuringMachine.address(memory, key, c(1), c(1), c(0.2f, 0.6f, 0.2f), c(5), previous)
                .value().toArray();
        assertTrue(sharp[2] > blurred[2]);
    }

    private static int argmax(float[] x) {
        int b = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[b]) {
                b = i;
            }
        }
        return b;
    }

    /** Copy: show L random bit vectors, a delimiter, then reproduce them. Returns {first loss, last loss}. */
    static float[] copyLosses(Layer machine, int iterations, float lr) {
        int bits = 3;
        int len = 3;
        int steps = 2 * len + 1;
        int batch = 8;
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom data = new SplittableRandom(4);
        float first = 0;
        float last = 0;
        for (int it = 0; it < iterations; it++) {
            float[] x = new float[batch * steps * (bits + 1)];
            float[] y = new float[batch * steps * bits];
            float[] mask = new float[batch * steps * bits];
            for (int b = 0; b < batch; b++) {
                for (int t = 0; t < len; t++) {
                    for (int k = 0; k < bits; k++) {
                        float bit = data.nextBoolean() ? 1 : 0;
                        x[(b * steps + t) * (bits + 1) + k] = bit;
                        y[(b * steps + len + 1 + t) * bits + k] = bit;
                        mask[(b * steps + len + 1 + t) * bits + k] = 1;
                    }
                }
                x[(b * steps + len) * (bits + 1) + bits] = 1;
            }
            Var p = machine.forward(Var.of(Tensor.of(Shape.of(batch, steps, bits + 1), x))).sigmoid();
            Tensor t = Tensor.of(Shape.of(batch, steps, bits), y);
            Tensor m = Tensor.of(Shape.of(batch, steps, bits), mask);
            Var bce = p.add(1e-6f).log().mul(t).add(p.neg().add(1 + 1e-6f).log().mul(t.neg().add(1f))).neg();
            Var loss = bce.mul(m).sum().mul(1f / (batch * len * bits));
            float v = loss.value().item();
            if (it < 10) {
                first += v / 10;
            }
            if (it >= iterations - 10) {
                last += v / 10;
            }
            loss.backward();
            opt.step(machine.parameters());
        }
        return new float[] {first, last};
    }

    @Test
    void ntmLearnsToCopy() {
        float[] l = copyLosses(new NeuralTuringMachine(4, 3, 8, 6, 32, new SplittableRandom(1)), 400, 0.01f);
        assertTrue(l[1] < 0.4f * l[0], "copy loss " + l[0] + " -> " + l[1]);
    }

    @Test
    void dncLearnsToCopy() {
        float[] l = copyLosses(new DifferentiableNeuralComputer(4, 3, 8, 6, 1, 32, new SplittableRandom(1)), 300,
                0.01f);
        assertTrue(l[1] < 0.5f * l[0], "copy loss " + l[0] + " -> " + l[1]);
    }

    private static DifferentiableNeuralComputer.Interface iface(float[] readKey, float readStrength, float[] value,
            float writeGate, float[] modes) {
        int w = value.length;
        return new DifferentiableNeuralComputer.Interface(
                Var.of(Tensor.of(Shape.of(1, 1, w), readKey)), c(readStrength), c(new float[w]), c(1),
                c(1, 1, 1, 1), Var.of(Tensor.of(Shape.of(1, 1, w), value)), c(0), c(1), c(writeGate),
                Var.of(Tensor.of(Shape.of(1, 1, 3), modes)));
    }

    @Test
    void dncAllocatesFreeSlotsAndReadsInWriteOrder() {
        float[][] items = {{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
        float[] content = {0, 1, 0};
        DifferentiableNeuralComputer.State s = DifferentiableNeuralComputer.initialState(1, 6, 4, 1);
        for (float[] item : items) {
            s = DifferentiableNeuralComputer.step(s, iface(new float[4], 1, item, 1, content));
        }
        // look up the first item by content, then follow the temporal links forward and back
        float[] none = new float[4];
        s = DifferentiableNeuralComputer.step(s, iface(items[0], 50, none, 0, content));
        assertArrayEquals(items[0], s.reads().value().toArray(), 0.05f);
        // three different slots are in use (usage counts the previous step's writes)
        float[] usage = s.usage().value().toArray();
        int used = 0;
        for (float u : usage) {
            used += u > 0.9f ? 1 : 0;
        }
        assertEquals(3, used);
        s = DifferentiableNeuralComputer.step(s, iface(none, 1, none, 0, new float[] {0, 0, 1}));
        assertArrayEquals(items[1], s.reads().value().toArray(), 0.05f);
        s = DifferentiableNeuralComputer.step(s, iface(none, 1, none, 0, new float[] {0, 0, 1}));
        assertArrayEquals(items[2], s.reads().value().toArray(), 0.05f);
        s = DifferentiableNeuralComputer.step(s, iface(none, 1, none, 0, new float[] {1, 0, 0}));
        assertArrayEquals(items[1], s.reads().value().toArray(), 0.05f);
    }
}
