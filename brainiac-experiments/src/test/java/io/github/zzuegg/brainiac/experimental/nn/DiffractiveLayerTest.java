package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiffractiveLayerTest {
    private static Tensor field(int n, long seed) {
        return Tensor.normal(Shape.of(1, n, n, 2), 0, 1, new SplittableRandom(seed));
    }

    @Test
    void zeroDistanceIsIdentityAndFft2RoundTrips() {
        Tensor f = field(8, 1);
        float[] round = DiffractiveLayer.fft2(DiffractiveLayer.fft2(Var.of(f), false), true).value().toArray();
        assertArrayEquals(f.toArray(), round, 1e-4f);
        // wavelength below 1/sqrt(0.5) pixel: no frequency of an 8x8 grid is evanescent
        float[] same = DiffractiveLayer.propagate(Var.of(f), 1, 0).value().toArray();
        assertArrayEquals(f.toArray(), same, 1e-4f);
    }

    @Test
    void propagationConservesEnergyAndSpreadsAPoint() {
        int n = 16;
        Tensor f = field(n, 2);
        float before = DiffractiveLayer.intensity(Var.of(f)).sum().value().item();
        float after = DiffractiveLayer.intensity(DiffractiveLayer.propagate(Var.of(f), 1, 7)).sum().value().item();
        assertEquals(before, after, 1e-3 * before);   // no evanescent part: the transfer is unitary
        float[] point = new float[n * n * 2];
        point[(8 * n + 8) * 2] = 1;
        float[] spread = DiffractiveLayer.intensity(DiffractiveLayer.propagate(
                Var.of(Tensor.of(Shape.of(1, n, n, 2), point)), 1, 7)).value().toArray();
        assertTrue(spread[8 * n + 8] < 0.5f, "a point source should diffract");
        // the phase mask alone does not change intensity
        DiffractiveLayer l = DiffractiveLayer.of(n, 1, 0, 3);
        float[] masked = DiffractiveLayer.intensity(l.forward(Var.of(f))).value().toArray();
        assertArrayEquals(DiffractiveLayer.intensity(Var.of(f)).value().toArray(), masked, 1e-3f);
    }

    @Test
    void phaseGradientsAreCorrect() {
        DiffractiveLayer l = DiffractiveLayer.of(4, 1.2, 2, 4);
        Tensor f = field(4, 5);
        Tensor weights = Tensor.uniform(Shape.of(1, 4, 4), 0, 1, new SplittableRandom(6));
        // the layer is mask then propagation; check that composite with the phases as input
        Tensor phi = Tensor.uniform(Shape.of(4, 4, 1), 0, 6, new SplittableRandom(7));
        GradCheck.Result g = GradCheck.check(v -> DiffractiveLayer.intensity(DiffractiveLayer.propagate(
                DiffractiveLayer.complexMultiply(Var.of(f), Var.concat(2, v.get(0).cos(), v.get(0).sin())),
                1.2, 2)).mul(weights).sum(), List.of(phi), 1e-3f, 2e-2f);
        assertTrue(g.passed(), g.toString());
        GradCheck.Result h = GradCheck.check(v -> DiffractiveLayer.intensity(l.forward(v.get(0))).mul(weights).sum(),
                List.of(f), 1e-3f, 2e-2f);
        assertTrue(h.passed(), h.toString());
        DiffractiveLayer.intensity(l.forward(Var.of(f))).mul(weights).sum().backward();
        assertTrue(l.phase().grad().abs().sum().item() > 0);
    }
}
