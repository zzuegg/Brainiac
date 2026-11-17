package io.github.zzuegg.brainiac.nn.diffusion;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Heun steps and trajectories, clamped predictions, conditions and guidance. */
class SamplerOptionsTest {

    private static double maxDifference(Tensor a, Tensor b) {
        return a.sub(b).abs().max().item();
    }

    @Test
    void heunIsCloserToTheExactPathThanEuler() {
        Denoiser v = DiffusionTest.gaussianVelocity(0.3, 0.6);
        Tensor noise = Tensor.normal(Shape.of(64, 1), 0, 1, new SplittableRandom(1));
        Tensor exact = FlowMatching.sample(v, noise, 2000);
        List<Tensor> path = new ArrayList<>();
        Tensor heun = FlowMatching.sample(v, noise, 8, FlowMatching.Solver.HEUN, path);
        Tensor euler = FlowMatching.sample(v, noise, 8, FlowMatching.Solver.EULER, null);
        assertEquals(9, path.size());
        assertArrayEquals(noise.toArray(), path.getFirst().toArray());
        assertArrayEquals(heun.toArray(), path.getLast().toArray());
        double heunError = maxDifference(exact, heun);
        double eulerError = maxDifference(exact, euler);
        assertTrue(heunError < eulerError / 5, "heun " + heunError + ", euler " + eulerError);
    }

    @Test
    void clippingKeepsPredictionsInRange() {
        Diffusion d = new Diffusion(NoiseSchedule.cosine(200), Diffusion.Prediction.NOISE);
        Diffusion clipped = d.clipSample(1);
        // the data is at 1.5, outside the range: clamped predictions end at the bound
        Tensor x = clipped.sample(DiffusionTest.gaussian(d, 1.5, 0), Shape.of(8, 3), 20, 0, new SplittableRandom(1),
                DiffusionTest.CPU);
        for (float f : x.toArray()) {
            assertEquals(1, f, 1e-3);
        }
    }

    @Test
    void conditionsAndGuidance() {
        // predicts the condition itself, whatever the input
        Denoiser.Conditional model = (noisy, time, condition) -> Var.of(condition.reshape(noisy.shape().dim(0), 1)
                .expand(noisy.shape()).add(0f));
        Var noisy = Var.of(Tensor.zeros(Shape.of(2, 3)));
        Tensor time = Tensor.zeros(Shape.of(2));
        Tensor labels = Tensor.of(new float[] {1, 2});
        assertArrayEquals(new float[] {1, 1, 1, 2, 2, 2}, model.given(labels).apply(noisy, time).value().toArray());
        Tensor none = Tensor.of(new float[] {0.5f, 0.5f});
        // 0.5 + 3 * (label - 0.5)
        assertArrayEquals(new float[] {2, 2, 2, 5, 5, 5},
                model.guided(labels, none, 3).apply(noisy, time).value().toArray(), 1e-6f);
    }
}
