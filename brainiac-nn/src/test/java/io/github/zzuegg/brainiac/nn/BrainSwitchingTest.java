package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Brain regions whose neurons choose their activation by their input, as a SwitchingDense does. */
class BrainSwitchingTest {

    private static final Tensor X = Tensor.of(Shape.of(3, 4), -1.2f, 0.3f, 2.0f, -0.4f, 0.9f, -2.2f, 0.1f, 1.4f,
            0.0f, 0.6f, -0.8f, -1.7f);

    @TempDir
    Path dir;

    private static float[] random(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) r.nextGaussian();
        }
        return v;
    }

    private static Brain brain(boolean hard) {
        Brain.Switching sw = new Brain.Switching(List.of(Activation.TANH, Activation.RELU, Activation.SIGMOID), hard);
        return Brain.builder().region("h", 5).activation(Activation.GELU).switching(sw)
                .connect(Brain.INPUT, "h", 1).build().create(Shape.of(4), new SplittableRandom(1));
    }

    @Test
    void aFullyConnectedSwitchingRegionIsASwitchingDense() {
        for (boolean hard : new boolean[] {false, true}) {
            Brain b = brain(hard);
            float[] w = random(20, 2);
            float[] v = random(20, 3);
            float[][] own = {random(5, 4), random(15, 5), random(15, 6)};
            float[] bias = random(5, 7);
            b.setWeightValues(0, w);
            b.setControlValues(0, v);
            b.setSwitchingValues("h", own);
            b.setBiasValues("h", bias);
            SwitchingDense d = new SwitchingDense(4, 5,
                    new Activation[] {Activation.TANH, Activation.RELU, Activation.SIGMOID}, hard,
                    Tensor.of(Shape.of(4, 5), w), Tensor.of(Shape.of(5), bias), Tensor.of(Shape.of(4, 5), v),
                    Tensor.of(Shape.of(5), own[0]), Tensor.of(Shape.of(3, 5), own[1]), Tensor.of(Shape.of(3, 5), own[2]));
            assertArrayEquals(d.forward(Var.of(X)).value().toArray(), b.forward(Var.of(X)).value().toArray(), 1e-5f,
                    "hard " + hard);
        }
    }

    @Test
    void neuronsThatDoNotSwitchKeepTheRegionActivation() throws IOException {
        Brain b = brain(false);
        boolean[] units = {true, false, true, false, false};
        b.setSwitchingUnits("h", units);
        assertArrayEquals(units, b.switchingUnits("h"));
        float[] all = b.forward(Var.of(X)).value().toArray();

        Brain plain = Brain.builder().region("h", 5).activation(Activation.GELU)
                .connect(Brain.INPUT, "h", 1).build().create(Shape.of(4), new SplittableRandom(9));
        plain.setWeightValues(0, b.weightValues(0));
        plain.setBiasValues("h", b.biasValues("h"));
        float[] fixed = plain.forward(Var.of(X)).value().toArray();
        for (int i = 0; i < 3; i++) {
            for (int u = 0; u < 5; u++) {
                if (!units[u]) {
                    assertEquals(fixed[i * 5 + u], all[i * 5 + u], 1e-6, "unit " + u);
                }
            }
        }

        // saved and loaded, with the same choice of switching neurons
        Model m = Model.sequential().input(4).add(b.spec()).dense(2).seed(3).build();
        Brain inModel = (Brain) m.layers().get(0);
        inModel.setSwitchingUnits("h", units);
        m.save(dir.resolve("switching.brainiac"));
        Model loaded = Model.load(dir.resolve("switching.brainiac"));
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray(), 1e-6f);
        assertArrayEquals(units, ((Brain) loaded.layers().get(0)).switchingUnits("h"));
    }

    @Test
    void theControlWeightsLearnInSoftMode() {
        Brain b = brain(false);
        Var loss = b.forward(Var.of(X)).square().sum();
        loss.backward();
        List<Var> params = b.parameters();
        assertEquals(6, params.size(), "weights, bias, control weights, control bias, scales, offsets");
        for (Var p : params) {
            float[] g = p.grad().toArray();
            assertEquals(true, Arrays.stream(box(g)).anyMatch(x -> x != 0), "a gradient reaches " + p.shape());
        }
    }

    private static Float[] box(float[] a) {
        Float[] out = new Float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i];
        }
        return out;
    }

    @Test
    void rejectsLearnableOrTooFewCandidates() {
        assertThrows(IllegalArgumentException.class, () -> new Brain.Switching(List.of(Activation.RELU), false));
        assertThrows(IllegalArgumentException.class, () -> Brain.builder().region("h", 2)
                .activation(Activation.RELU).switching(new Brain.Switching(List.of(Activation.RELU,
                        Activation.prelu()), false)));
        Brain plain = Brain.builder().region("h", 2).connect(Brain.INPUT, "h", 1).build()
                .create(Shape.of(4), new SplittableRandom(1));
        assertThrows(IllegalArgumentException.class, () -> plain.controlValues(0));
        assertThrows(IllegalArgumentException.class, () -> plain.switchingValues("h"));
    }
}
