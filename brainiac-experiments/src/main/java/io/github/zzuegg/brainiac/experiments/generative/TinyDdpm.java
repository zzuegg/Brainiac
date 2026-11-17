package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConditionalDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * X29: a tiny denoising diffusion model (DDPM, Ho et al., <a href="https://arxiv.org/abs/2006.11239">2006.11239</a>)
 * on 8x8 shapes, built on core {@link Diffusion}: epsilon prediction over a linear or cosine
 * {@link NoiseSchedule}, sampled with stochastic DDPM steps ({@code eta = 1}, all steps) or
 * deterministic DDIM steps (Song et al., <a href="https://arxiv.org/abs/2010.02502">2010.02502</a>,
 * {@code eta = 0}, few steps). A class-conditional network trained with label dropout adds
 * classifier-free guidance ({@link ConditionalDenoiser#guided}). Sampling clips the implied
 * clean image to [-1, 1] (core {@link Diffusion#clipSample}).
 *
 * <p>Samplers can record the predicted clean images at every step, and
 * {@link #denoisingGif} renders that as an animation of noise crystallizing into shapes.
 */
@SuppressWarnings("exports")   // experimental.nn.diffusion is not exported yet
public final class TinyDdpm {
    private final ConditionalDenoiser net;
    private final Diffusion diffusion;
    private final Optimizer optimizer;
    private final float labelDropout;
    private final SplittableRandom random;

    private TinyDdpm(ConditionalDenoiser net, int steps, boolean cosine, float learningRate, float labelDropout,
            long seed) {
        this.net = net;
        this.diffusion = new Diffusion(cosine ? NoiseSchedule.cosine(steps)
                : NoiseSchedule.linear(steps, 1e-4 * 1000 / steps, 0.02 * 1000 / steps), Diffusion.Prediction.NOISE)
                .clipSample(1f);
        this.optimizer = Optimizer.adam(learningRate);
        this.labelDropout = labelDropout;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates a model.
     *
     * @param net the noise predictor; its time scale should be {@code 1 / steps}
     * @param steps the number of diffusion steps T
     * @param cosine the cosine schedule if true, else the linear one
     * @param learningRate the Adam learning rate
     * @param labelDropout the share of labels replaced by "no label" while training
     * @param seed the seed for noise and step sampling
     * @return the model
     */
    public static TinyDdpm of(ConditionalDenoiser net, int steps, boolean cosine, float learningRate,
            float labelDropout, long seed) {
        return new TinyDdpm(net, steps, cosine, learningRate, labelDropout, seed);
    }

    /** @return the number of diffusion steps */
    public int steps() {
        return diffusion.schedule().steps();
    }

    /**
     * Returns the cumulative signal level.
     *
     * @param t the step, 0..T-1
     * @return {@code alphaBar[t]}
     */
    public double alphaBar(int t) {
        return diffusion.schedule().alphaBar(t);
    }

    /**
     * One optimizer step on the core epsilon-prediction loss.
     *
     * @param x0 clean samples
     * @param labels their classes (or -1)
     * @return the loss before the step
     */
    public float trainStep(float[][] x0, int[] labels) {
        int[] l = ConditionalDenoiser.dropLabels(labels, labelDropout, random);
        Var loss = diffusion.loss(net.bind(l), Nets.matrix(x0), random);
        float value = loss.value().item();
        loss.backward();
        Nets.clip(net.parameters(), 1f);
        optimizer.step(net.parameters());
        return value;
    }

    /** Wraps a noise predictor so every call also records its clean-image estimate. */
    private Denoiser recording(Denoiser model, List<float[][]> trajectory) {
        if (trajectory == null) {
            return model;
        }
        return (noisy, time) -> {
            Var eps = model.apply(noisy, time);
            double ab = diffusion.schedule().alphaBar(Math.round(time.toArray()[0]));
            Tensor x0 = noisy.value().sub(eps.value().mul((float) Math.sqrt(1 - ab))).div((float) Math.sqrt(ab));
            float[][] rows = Nets.rows(x0);
            for (float[] r : rows) {
                for (int j = 0; j < r.length; j++) {
                    r[j] = Math.max(-1, Math.min(1, r[j]));
                }
            }
            trajectory.add(rows);
            return eps;
        };
    }

    /**
     * Samples with the core sampler.
     *
     * @param labels the class of each sample (or -1)
     * @param steps how many denoising steps, at most T
     * @param eta 0 for deterministic DDIM, 1 for DDPM's random steps
     * @param guidance the classifier-free guidance weight; 1 means none
     * @param seed the noise seed
     * @param trajectory receives the predicted clean images after each step, or {@code null}
     * @return the samples
     */
    public float[][] sample(int[] labels, int steps, float eta, float guidance, long seed, List<float[][]> trajectory) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor noise = Nets.gaussian(labels.length, net.dimension(), r);
        Denoiser model = net.guided(labels, guidance);
        return Nets.rows(diffusion.sample(recording(model, trajectory), noise, steps, eta, r));
    }

    /**
     * Ancestral DDPM sampling through all T steps.
     *
     * @param labels the class of each sample (or -1)
     * @param guidance the guidance weight
     * @param seed the noise seed
     * @param trajectory receives the predicted clean images, or {@code null}
     * @return the samples
     */
    public float[][] sampleDdpm(int[] labels, float guidance, long seed, List<float[][]> trajectory) {
        return sample(labels, steps(), 1, guidance, seed, trajectory);
    }

    /**
     * Deterministic DDIM sampling with fewer steps.
     *
     * @param labels the class of each sample (or -1)
     * @param steps how many steps
     * @param guidance the guidance weight
     * @param seed the noise seed
     * @param trajectory receives the predicted clean images, or {@code null}
     * @return the samples
     */
    public float[][] sampleDdim(int[] labels, int steps, float guidance, long seed, List<float[][]> trajectory) {
        return sample(labels, steps, 0, guidance, seed, trajectory);
    }

    /**
     * Renders a sampling trajectory as an animated GIF: each frame tiles the predicted clean
     * images of all samples side by side. The last frame is held a little longer.
     *
     * @param trajectory what a sampler recorded, one {@code [samples][side*side]} array per step
     * @param side the image side length
     * @param scale the pixel upscaling factor
     * @return the GIF file's bytes
     */
    public static byte[] denoisingGif(List<float[][]> trajectory, int side, int scale) {
        int n = trajectory.get(0).length;
        int w = n * (side + 1) - 1;
        List<float[]> frames = new java.util.ArrayList<>();
        for (int f = 0; f < trajectory.size() + 4; f++) {
            float[][] frame = trajectory.get(Math.min(f, trajectory.size() - 1));
            float[] px = new float[w * side];
            for (int s = 0; s < n; s++) {
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        px[y * w + s * (side + 1) + x] = frame[s][y * side + x];
                    }
                }
            }
            frames.add(px);
        }
        return Nets.grayGif(frames, w, side, -1, 1, scale, 12);
    }
}
