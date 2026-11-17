package io.github.zzuegg.brainiac.nn.diffusion;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Samplers driven by exact denoisers, which know the data: one point, or a normal
 * distribution. Whatever is wrong in a sampler shows up as samples off the data.
 */
class DiffusionTest {

    static final Device CPU = Device.cpu();

    /** The best prediction for data N(m, s^2), per value: what a perfectly trained model gives. */
    static Denoiser gaussian(Diffusion d, double m, double s) {
        return (noisy, time) -> {
            float[] x = noisy.value().toArray();
            float[] t = time.toArray();
            int per = x.length / t.length;
            float[] out = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                double ab = d.schedule().alphaBar((int) t[i / per]);
                double a = Math.sqrt(ab);
                double b = Math.sqrt(1 - ab);
                double v = a * a * s * s + b * b;
                double eps = b * (x[i] - a * m) / v;
                double x0 = m + a * s * s * (x[i] - a * m) / v;
                out[i] = (float) switch (d.prediction()) {
                    case NOISE -> eps;
                    case SAMPLE -> x0;
                    case VELOCITY -> a * eps - b * x0;
                };
            }
            return Var.of(Tensor.of(noisy.shape(), out, CPU));
        };
    }

    static double[] meanAndStd(Tensor samples) {
        float[] v = samples.toArray();
        double sum = 0;
        double sq = 0;
        for (float f : v) {
            sum += f;
            sq += (double) f * f;
        }
        double mean = sum / v.length;
        return new double[] {mean, Math.sqrt(sq / v.length - mean * mean)};
    }

    @ParameterizedTest
    @EnumSource(Diffusion.Prediction.class)
    void deterministicStepsFromAnyNoiseReachTheOnlyDataPoint(Diffusion.Prediction p) {
        Diffusion d = new Diffusion(NoiseSchedule.cosine(200), p);
        Tensor x = d.sample(gaussian(d, 1.5, 0), Shape.of(8, 3), 20, 0, new SplittableRandom(1), CPU);
        for (float f : x.toArray()) {
            assertEquals(1.5, f, 1e-3);
        }
    }

    @ParameterizedTest
    @EnumSource(Diffusion.Prediction.class)
    void samplesFollowTheDataDistribution(Diffusion.Prediction p) {
        for (float eta : new float[] {0, 1}) {
            Diffusion d = new Diffusion(NoiseSchedule.linear(1000, 1e-4, 0.02), p);
            // random steps put back the spread the mean prediction leaves out only when they are small
            int steps = eta == 0 ? 100 : 1000;
            Tensor x = d.sample(gaussian(d, -0.7, 0.4), Shape.of(4000, 1), steps, eta, new SplittableRandom(2), CPU);
            double[] ms = meanAndStd(x);
            assertEquals(-0.7, ms[0], 0.03, p + " eta " + eta + " mean");
            assertEquals(0.4, ms[1], 0.03, p + " eta " + eta + " std");
        }
    }

    @Test
    void anExactDenoiserHasNoLossOnItsOnlyPoint() {
        Diffusion d = new Diffusion(NoiseSchedule.cosine(100), Diffusion.Prediction.NOISE);
        Tensor data = Tensor.full(Shape.of(16, 4), 2f);
        assertEquals(0, d.loss(gaussian(d, 2, 0), data, new SplittableRandom(3)).value().item(), 1e-6);
        assertTrue(d.loss(gaussian(d, 0, 0), data, new SplittableRandom(3)).value().item() > 0.01);
    }

    @ParameterizedTest
    @EnumSource(Diffusion.Prediction.class)
    void aStepPerTokenTrainsLikeDiffusionForcing(Diffusion.Prediction p) {
        Diffusion d = new Diffusion(NoiseSchedule.cosine(100), p);
        Tensor data = Tensor.full(Shape.of(4, 6, 3), 2f);
        int[][] steps = d.randomTokenSteps(4, 6, new SplittableRandom(5));
        assertTrue(java.util.Arrays.stream(steps).flatMapToInt(java.util.Arrays::stream).distinct().count() > 6);
        assertEquals(0, d.loss(gaussian(d, 2, 0), data, steps, new SplittableRandom(3)).value().item(), 1e-6);
        assertTrue(d.loss(gaussian(d, 0, 0), data, steps, new SplittableRandom(3)).value().item() > 0.01);
        // the same step on every token is the per-example noise
        Tensor noise = Tensor.normal(data.shape(), 0, 1, new SplittableRandom(9));
        int[][] same = new int[4][6];
        for (int[] row : same) {
            java.util.Arrays.fill(row, 40);
        }
        assertArrayEquals(d.noisy(data, noise, new int[] {40, 40, 40, 40}).toArray(), d.noisy(data, noise, same).toArray(),
                1e-6f);
    }

    @Test
    void schedulesDecrease() {
        for (NoiseSchedule s : new NoiseSchedule[] {NoiseSchedule.cosine(1000), NoiseSchedule.linear(1000, 1e-4, 0.02)}) {
            assertEquals(1000, s.steps());
            assertTrue(s.alphaBar(0) > 0.99, s.toString());
            assertTrue(s.alphaBar(999) < 1e-3, s.toString());
        }
    }

    /** The best velocity for data N(m, s^2) on straight paths. */
    static Denoiser gaussianVelocity(double m, double s) {
        return (noisy, time) -> {
            float[] x = noisy.value().toArray();
            float[] t = time.toArray();
            int per = x.length / t.length;
            float[] out = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                double tt = t[i / per];
                double k = 1 - tt;
                double v = k * k * s * s + tt * tt;
                double r = x[i] - k * m;
                double x0 = m + k * s * s * r / v;
                double n = tt * r / v;
                out[i] = (float) (n - x0);
            }
            return Var.of(Tensor.of(noisy.shape(), out, CPU));
        };
    }

    @Test
    void flowMatchingFollowsTheVelocityToTheData() {
        Tensor x = FlowMatching.sample(gaussianVelocity(1.5, 0), Shape.of(8, 2), 10, new SplittableRandom(4), CPU);
        for (float f : x.toArray()) {
            assertEquals(1.5, f, 1e-4);
        }
        double[] ms = meanAndStd(FlowMatching.sample(gaussianVelocity(0.3, 0.6), Shape.of(4000, 1), 100,
                new SplittableRandom(5), CPU));
        assertEquals(0.3, ms[0], 0.03);
        assertEquals(0.6, ms[1], 0.03);
        Tensor data = Tensor.full(Shape.of(16, 4), -1f);
        assertEquals(0, FlowMatching.loss(gaussianVelocity(-1, 0), data, new SplittableRandom(6)).value().item(), 1e-5);
    }
}
