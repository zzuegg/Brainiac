package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.Rope;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RopeScalingTest {

    private final Tensor x = Tensor.normal(Shape.of(1, 2, 5, 16), 0, 1, new SplittableRandom(1));

    @Test
    void noScalingIsPlainRotary() {
        RopeScaling none = RopeScaling.of(16, 10000, RopeScaling.Method.NONE, 1, 128);
        assertArrayEquals(Functional.rotary(Var.of(x), 10000, 3).value().toArray(),
                none.apply(Var.of(x), 3).value().toArray(), 1e-5f);
    }

    @Test
    void yarnKeepsFastBandsAndInterpolatesSlowOnes() {
        RopeScaling yarn = RopeScaling.of(64, 10000, RopeScaling.Method.YARN, 8, 256);
        double[] base = yarn.baseFrequencies();
        double[] f = yarn.frequencies();
        assertEquals(base[0], f[0], 1e-12);                       // many rotations: kept
        assertEquals(base[31] / 8, f[31], 1e-12);                 // under one rotation: interpolated
        for (int i = 1; i < f.length; i++) {
            assertTrue(f[i] / base[i] <= f[i - 1] / base[i - 1] + 1e-12, "ramp is monotone");
        }
        assertEquals(0.1 * Math.log(8) + 1, yarn.mscale(), 1e-6);   // the core keeps it as a float
        // the slowest band at position 8p has the angle the unscaled one had at p
        assertEquals(base[31] * 100, f[31] * 800, 1e-9);
    }

    @Test
    void ntkScalesTheBaseAndDynamicNtkOnlyBeyondTheContext() {
        RopeScaling ntk = RopeScaling.of(16, 10000, RopeScaling.Method.NTK, 4, 128);
        double[] f = ntk.frequencies();
        assertEquals(1, f[0], 1e-12);                            // the fastest band never changes
        assertEquals(Math.pow(10000 * Math.pow(4, 16 / 14.0), -14.0 / 16), f[7], 1e-12);
        RopeScaling dyn = RopeScaling.of(16, 10000, RopeScaling.Method.DYNAMIC_NTK, 4, 128);
        assertArrayEquals(dyn.baseFrequencies(), dyn.frequencies(100), 1e-12);
        assertTrue(dyn.frequencies(400)[7] < dyn.baseFrequencies()[7]);
        RopeScaling linear = RopeScaling.of(16, 10000, RopeScaling.Method.LINEAR, 2, 128);
        assertEquals(0.5, linear.frequencies()[0], 1e-12);
    }

    @Test
    void freqFactorsDropIntoTheCoreRope() {
        RopeScaling yarn = RopeScaling.of(16, 10000, RopeScaling.Method.YARN, 4, 32).bands(8, 1);
        Rope core = new Rope(new Rope.Settings(16, 10000, "default", 1, 1, 1, 32, null)
                .withFreqFactors(yarn.freqFactors()));
        int[][] pos = {{7, 8, 9, 10, 11}};
        float[] expected = yarn.apply(Var.of(x), 7).value().div((float) yarn.mscale()).toArray();
        assertArrayEquals(expected, core.apply(Var.of(x), pos).value().toArray(), 1e-4f);
    }

    @Test
    void yarnAndDynamicNtkMatchTheCoreRope() {
        int[][] pos = {{300, 301, 302, 303, 304}};
        RopeScaling yarn = RopeScaling.of(16, 10000, RopeScaling.Method.YARN, 4, 64).bands(16, 2);
        Rope coreYarn = new Rope(new Rope.Settings(16, 10000, "yarn", 4, 1, 1, 64, null).withYarn(16, 2, 0));
        assertArrayEquals(coreYarn.frequencies(), yarn.frequencies(), 0);
        assertArrayEquals(coreYarn.apply(Var.of(x), pos).value().toArray(), yarn.apply(Var.of(x), 300).value().toArray(),
                1e-6f);
        // dynamic NTK: the core grows its base once positions pass the original context
        RopeScaling dyn = RopeScaling.of(16, 10000, RopeScaling.Method.DYNAMIC_NTK, 4, 64);
        Rope coreDyn = new Rope(new Rope.Settings(16, 10000, "dynamic", 4, 1, 1, 64, null));
        float[] expected = coreDyn.apply(Var.of(x), pos).value().toArray();
        assertArrayEquals(coreDyn.frequencies(), dyn.frequencies(305), 0);
        assertArrayEquals(expected, dyn.apply(Var.of(x), 300).value().toArray(), 1e-6f);
        // the same base in closed form: theta (s L / L0 - (s - 1))^(d / (d - 2))
        double base = 10000 * Math.pow(4 * 305 / 64.0 - 3, 16 / 14.0);
        assertEquals(Math.pow(base, -14.0 / 16), dyn.frequencies(305)[7], 1e-6 * dyn.frequencies(305)[7]);
    }
}
