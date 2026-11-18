package io.github.zzuegg.brainiac.nn.diffusion;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * The latent-diffusion layers as building blocks of self-made networks: they train, run
 * without gradients to the same values, save and load, and their samplers reach the data.
 * (Their match with pretrained Stable Diffusion is tested in brainiac-diffusion.)
 */
class DiffusionLayersTest {

    static final Device CPU = Device.cpu();

    @TempDir
    Path dir;

    private static Tensor uniform(SplittableRandom r, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, r);
    }

    /** Plain gradient descent until the loss has halved, or fails after {@code steps}. */
    private static void fits(Layer layer, Supplier<Var> loss, int steps, float rate) {
        float first = Float.NaN;
        float last = Float.NaN;
        for (int i = 0; i < steps; i++) {
            Var l = loss.get();
            last = l.value().item();
            if (i == 0) {
                first = last;
            }
            if (last < first / 2) {
                return;
            }
            l.backward();
            for (Var p : layer.parameters()) {
                assertNotNull(p.grad(), "every parameter gets a gradient");
                p.assign(p.value().sub(p.grad().mul(rate)));
                p.zeroGrad();
            }
        }
        throw new AssertionError("loss went from " + first + " to " + last + " in " + steps + " steps");
    }

    private static Var mse(Var a, Tensor target) {
        return a.sub(Var.of(target)).square().mean();
    }

    @Test
    void unetFitsAndGivesTheSameValuesWithoutGradients() {
        SplittableRandom r = new SplittableRandom(1);
        UNet unet = new UNet(UNet.Settings.of(2, 2, new int[] {8, 16}, 6), r);
        Tensor x = uniform(r, 3, 2, 8, 8);
        Tensor t = Tensor.of(Shape.of(3), new float[] {3, 50, 400}, CPU);
        Tensor ctx = uniform(r, 3, 5, 6);
        Tensor target = uniform(r, 3, 2, 8, 8);
        Var withGrad = unet.forward(Var.of(x), Var.of(t), Var.of(ctx), null, false);
        assertTrue(withGrad.requiresGrad());
        Var without = Var.noGrad(() -> unet.forward(Var.of(x), Var.of(t), Var.of(ctx), null, false));
        assertFalse(without.requiresGrad());
        assertArrayEquals(withGrad.value().toArray(), without.value().toArray(), 1e-5f);
        fits(unet, () -> mse(unet.forward(Var.of(x), Var.of(t), Var.of(ctx), null, true), target), 200, 0.05f);
    }

    @Test
    void unetTakesSizeConditioningLikeStableDiffusionXl() {
        SplittableRandom r = new SplittableRandom(2);
        UNet.Settings s = new UNet.Settings(4, 4, new int[] {8, 16}, new boolean[] {false, true}, 1,
                new int[] {1, 2}, new int[] {1, 2}, 6, true, 4, 1e-5f, 4, 5, 6, true, 0);
        UNet unet = new UNet(s, r);
        Tensor x = uniform(r, 2, 4, 8, 8);
        Tensor t = Tensor.of(Shape.of(2), new float[] {10, 900}, CPU);
        Tensor ctx = uniform(r, 2, 3, 6);
        Tensor added = Tensor.concat(1, uniform(r, 2, 5), Tensor.of(Shape.of(2, 6),
                new float[] {64, 64, 0, 0, 64, 64, 32, 48, 4, 2, 32, 48}, CPU));
        Var out = unet.forward(List.of(Var.of(x), Var.of(t), Var.of(ctx), Var.of(added)), false);
        assertEquals(x.shape(), out.shape());
        mse(out, uniform(r, 2, 4, 8, 8)).backward();
        for (Var p : unet.parameters()) {
            assertNotNull(p.grad(), "every parameter, the size embedding's too, gets a gradient");
        }
        assertThrows(IllegalArgumentException.class,
                () -> unet.forward(List.of(Var.of(x), Var.of(t), Var.of(ctx)), false));
    }

    @Test
    void unetTrainsAsADenoiser() {
        SplittableRandom r = new SplittableRandom(3);
        UNet unet = new UNet(UNet.Settings.of(1, 1, new int[] {8, 8}, 0), r);
        Diffusion d = new Diffusion(NoiseSchedule.cosine(100), Diffusion.Prediction.NOISE);
        Tensor data = uniform(r, 4, 1, 4, 4);
        // the same noise and steps each time, so that the loss only falls if the model learns
        fits(unet, () -> d.loss(unet, data, new SplittableRandom(4)), 200, 0.05f);
    }

    @Test
    void autoencoderFitsAndEncodesSmaller() {
        SplittableRandom r = new SplittableRandom(5);
        AutoencoderKL vae = new AutoencoderKL(AutoencoderKL.Settings.of(3, 2, 8, 16), r);
        Tensor images = uniform(r, 2, 3, 8, 8);
        AutoencoderKL.Posterior p = vae.encode(Var.of(images));
        assertEquals(Shape.of(2, 2, 4, 4), p.mean().shape());
        assertTrue(Float.isFinite(p.kl().value().item()));
        assertEquals(Shape.of(2, 3, 8, 8), vae.decode(p.mean()).shape());
        assertEquals(images.shape(), vae.forward(Var.of(images)).shape());
        fits(vae, () -> vae.loss(Var.of(images), 1e-4f, new SplittableRandom(6)), 300, 0.05f);
    }

    @Test
    void clipTextEncoderIsCausalAndPoolsAtTheEndToken() {
        SplittableRandom r = new SplittableRandom(7);
        ClipTextEncoder clip = new ClipTextEncoder(new ClipTextEncoder.Settings(20, 16, 2, 2, 32, 8,
                Activation.QUICK_GELU, 1e-5f, -1, 4), r);
        Tensor a = Tensor.of(Shape.of(1, 6), new float[] {1, 5, 7, 19, 0, 0}, CPU);
        Tensor b = Tensor.of(Shape.of(1, 6), new float[] {1, 5, 7, 19, 3, 3}, CPU);
        ClipTextEncoder.Output oa = clip.encode(Var.of(a), false);
        ClipTextEncoder.Output ob = clip.encode(Var.of(b), false);
        assertEquals(Shape.of(1, 6, 16), oa.last().shape());
        assertEquals(Shape.of(1, 4), oa.pooled().shape());
        // tokens after the end token (19, the largest id) do not change it or what comes before
        assertArrayEquals(oa.pooled().value().toArray(), ob.pooled().value().toArray(), 1e-6f);
        assertArrayEquals(oa.penultimate().value().slice(1, 0, 4).toArray(),
                ob.penultimate().value().slice(1, 0, 4).toArray(), 1e-6f);
        Tensor target = uniform(r, 1, 4);
        fits(clip, () -> mse(clip.encode(Var.of(a), true).pooled(), target), 200, 0.1f);
    }

    @Test
    void layersWorkInAModel() {
        SplittableRandom r = new SplittableRandom(8);
        Model m = Model.sequential().input(4, 8, 8)
                .layer(new ResnetBlock(new ResnetBlock.Settings(4, 8, 0, 4, 1e-5f, 0.1f), r))
                .layer(new ImageAttention(new ImageAttention.Settings(8, 2, 4, 1e-6f), r))
                .layer(new SpatialTransformer(new SpatialTransformer.Settings(8, 2, 1, 0, 4, false), r))
                .build();
        Tensor out = m.predict(uniform(r, 2, 4, 8, 8));
        assertEquals(Shape.of(2, 8, 8, 8), out.shape());
    }

    @ParameterizedTest
    @EnumSource(SigmaSampler.Method.class)
    void samplersReachTheData(SigmaSampler.Method method) {
        for (Diffusion.Prediction p : Diffusion.Prediction.values()) {
            Diffusion d = new Diffusion(NoiseSchedule.scaledLinear(1000, 0.00085, 0.012), p);
            // one data point: every sampler ends on it
            double[] sigmas = SigmaSampler.karras(d.schedule(), 20);
            Tensor x = Tensor.normal(Shape.of(8, 2), 0, 1, new SplittableRandom(9)).mul((float) sigmas[0]);
            Tensor out = SigmaSampler.sample(SigmaSampler.model(DiffusionTest.gaussian(d, 1.5, 0), d.schedule(), p),
                    x, sigmas, method, new SplittableRandom(10), SigmaSampler.Listener.NONE);
            for (float f : out.toArray()) {
                assertEquals(1.5, f, 1e-3, method + " " + p);
            }
        }
    }

    @Test
    void samplersFollowTheDataDistribution() {
        Diffusion d = new Diffusion(NoiseSchedule.linear(1000, 1e-4, 0.02), Diffusion.Prediction.NOISE);
        for (SigmaSampler.Method method : SigmaSampler.Method.values()) {
            double[] sigmas = SigmaSampler.karras(d.schedule(), 200);
            Tensor x = Tensor.normal(Shape.of(4000, 1), 0, 1, new SplittableRandom(11)).mul((float) sigmas[0]);
            Tensor out = SigmaSampler.sample(SigmaSampler.model(DiffusionTest.gaussian(d, -0.7, 0.4), d.schedule(),
                    Diffusion.Prediction.NOISE), x, sigmas, method, new SplittableRandom(12),
                    SigmaSampler.Listener.NONE);
            double[] ms = DiffusionTest.meanAndStd(out);
            assertEquals(-0.7, ms[0], 0.03, method + " mean");
            assertEquals(0.4, ms[1], 0.04, method + " std");
        }
    }

    @Test
    void samplingStopsWhenTheListenerSaysSo() {
        Diffusion d = new Diffusion(NoiseSchedule.cosine(100), Diffusion.Prediction.NOISE);
        double[] sigmas = SigmaSampler.linear(d.schedule(), 5);
        int[] seen = {0};
        assertThrows(java.util.concurrent.CancellationException.class, () -> SigmaSampler.sample(
                SigmaSampler.model(DiffusionTest.gaussian(d, 0, 1), d.schedule(), Diffusion.Prediction.NOISE),
                Tensor.zeros(Shape.of(2, 2)), sigmas, SigmaSampler.Method.EULER, new SplittableRandom(1),
                (step, steps, denoised) -> ++seen[0] < 3));
        assertEquals(3, seen[0]);
    }

    @Test
    void timestepsAndNoiseLevelsConvert() {
        NoiseSchedule s = NoiseSchedule.scaledLinear(1000, 0.00085, 0.012);
        for (int t : new int[] {0, 1, 250, 999}) {
            assertEquals(t, s.timestep(s.sigma(t)), 1e-6);
        }
        assertEquals(14.6146, s.sigma(999), 1e-3);   // Stable Diffusion's largest noise level
        assertEquals(0.0292, s.sigma(0), 1e-4);
    }

    @Test
    void layersBuiltWithoutValuesCostNoMemoryUntilFilled() {
        UNet shell = Tensor.withoutValues(() -> new UNet(UNet.Settings.of(4, 4, new int[] {32, 64}, 8),
                new SplittableRandom(1)));
        assertTrue(shell.parameters().size() > 50);
        assertEquals(Shape.of(32, 4, 3, 3), shell.convIn().weight().shape());
    }
}
