package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.DiffusionLossWeighting;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Diffusion training targets and loss weights on the 8x8 {@link ToyImages}, at a fixed
 * training budget and a fixed number of deterministic sampling steps: noise prediction
 * (DDPM's default), velocity prediction (core {@code Diffusion.Prediction.VELOCITY}), each
 * with the uniform weighting of core {@link Diffusion#loss} or with Min-SNR-gamma
 * ({@link DiffusionLossWeighting}). Samples are scored by the mean squared distance per
 * pixel to the nearest real image, a stand-in for FID on a data set small enough to
 * enumerate.
 */
@SuppressWarnings("exports")
public final class DiffusionTargets {
    private static final int STEPS = 100;

    private DiffusionTargets() {
    }

    /**
     * Trains one configuration and scores its samples.
     *
     * @param prediction the training target
     * @param weighting the loss weighting
     * @param iterations training steps of 64 images
     * @param samplingSteps DDIM steps for sampling
     * @param seed the seed
     * @return the mean squared distance per pixel of 128 samples to their nearest real image
     */
    public static double run(Diffusion.Prediction prediction, DiffusionLossWeighting weighting, int iterations,
            int samplingSteps, long seed) {
        MlpDenoiser net = MlpDenoiser.of(ToyImages.PIXELS, 0, 128, 2, seed).timeScale(1f / STEPS);
        Diffusion diffusion = new Diffusion(NoiseSchedule.cosine(STEPS), prediction).clipSample(1f);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int[] none = new int[64];
        Arrays.fill(none, -1);
        for (int i = 0; i < iterations; i++) {
            float[][] x = new float[64][];
            for (int b = 0; b < x.length; b++) {
                x[b] = ToyImages.sample(r.nextInt(ToyImages.CLASSES), r);
            }
            Var loss = weighting.loss(diffusion, net.bind(none), Nets.matrix(x), r);
            loss.backward();
            Nets.clip(net.parameters(), 1f);
            opt.step(net.parameters());
        }
        int[] all = new int[128];
        Arrays.fill(all, -1);
        Tensor noise = Nets.gaussian(all.length, ToyImages.PIXELS, new SplittableRandom(seed + 99));
        float[][] samples = Nets.rows(diffusion.sample(net.bind(all), noise, samplingSteps, 0,
                new SplittableRandom(seed + 2)));
        double d = 0;
        for (float[] s : samples) {
            d += ToyImages.nearest(s)[1] / ToyImages.PIXELS;
        }
        return d / samples.length;
    }
}
