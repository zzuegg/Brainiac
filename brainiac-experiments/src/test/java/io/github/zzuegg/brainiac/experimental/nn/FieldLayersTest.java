package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FieldLayersTest {

    @Test
    void trigGradientsCheck() {
        Tensor x = Tensor.uniform(Shape.of(3, 4), -3, 3, new SplittableRandom(1));
        assertTrue(GradCheck.check(v -> v.get(0).sin().sum(), List.of(x)).passed());
        assertTrue(GradCheck.check(v -> v.get(0).cos().sum(), List.of(x)).passed());
    }

    @Test
    void positionalEncodingHasNerfFrequencies() {
        FourierFeatures pe = FourierFeatures.positional(2, 3, true);
        Var out = pe.forward(Var.of(Tensor.of(Shape.of(1, 2), 0.25f, 0.5f)));
        assertEquals(2 * 2 * 3 + 2, out.shape().dim(1));
        float[] v = out.value().toArray();
        assertEquals(Math.sin(Math.PI * 0.25), v[0], 1e-5);   // x, octave 0
        assertEquals(Math.sin(4 * Math.PI * 0.25), v[2], 1e-5);   // x, octave 2
        assertEquals(Math.cos(2 * Math.PI * 0.5), v[6 + 4], 1e-5);   // cos of y, octave 1
    }

    @Test
    void fourierFeaturesDropIntoAModel() {
        Model m = Model.sequential().input(2).add(FourierFeatures.spec(8, 1f, true)).dense(3).seed(1).build();
        assertEquals(3, m.predict(new float[] {0.1f, 0.2f}).length);
        GradCheck.Result r = GradCheck.check(v -> FourierFeatures.gaussian(2, 4, 1f, 3, false).forward(v.get(0))
                .square().sum(), List.of(Tensor.uniform(Shape.of(2, 2), 0, 1, new SplittableRandom(2))));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void splatRendererGradientsCheck() {
        GaussianSplats2d splats = GaussianSplats2d.of(3, 1, 4);
        Tensor px = Tensor.uniform(Shape.of(5, 2), 0.2f, 0.8f, new SplittableRandom(5));
        Var out = splats.forward(Var.of(px));
        assertEquals(Shape.of(5, 1), out.shape());
        // gradient with respect to the pixel positions, through the whole compositing chain
        GradCheck.Result r = GradCheck.check(v -> splats.forward(v.get(0)).sum(), List.of(px), 1e-3f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }
}
