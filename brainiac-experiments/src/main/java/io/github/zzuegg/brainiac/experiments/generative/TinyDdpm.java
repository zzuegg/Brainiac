package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.GifWriter;
import io.github.zzuegg.brainiac.experimental.nn.Denoiser;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * X29: a tiny denoising diffusion model (DDPM, Ho et al., <a href="https://arxiv.org/abs/2006.11239">2006.11239</a>).
 * Training corrupts a clean sample {@code x0} to {@code x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps}
 * and teaches the {@link Denoiser} to predict {@code eps} (epsilon prediction, the "simple"
 * loss). Sampling runs the chain backwards, either with the stochastic DDPM step or with
 * the deterministic DDIM step (Song et al., <a href="https://arxiv.org/abs/2010.02502">2010.02502</a>),
 * which can skip most of the steps. Training with a share of labels dropped makes
 * classifier-free guidance possible at sampling time
 * (<a href="https://arxiv.org/abs/2207.12598">2207.12598</a>):
 * {@code eps = eps_uncond + w (eps_cond - eps_uncond)}.
 *
 * <p>Every sampler can record its trajectory of predicted clean images, which
 * {@code GifWriter} turns into an animation of noise crystallizing into shapes.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class TinyDdpm {
    private final Denoiser net;
    private final NoiseSchedule schedule;
    private final Optimizer optimizer;
    private final float labelDropout;
    private final SplittableRandom random;

    private TinyDdpm(Denoiser net, int steps, boolean cosine, float learningRate, float labelDropout, long seed) {
        this.net = net;
        this.schedule = cosine ? NoiseSchedule.cosine(steps) : NoiseSchedule.linear(steps);
        this.optimizer = Optimizer.adam(learningRate);
        this.labelDropout = labelDropout;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates a model.
     *
     * @param net the noise predictor
     * @param steps the number of diffusion steps T
     * @param cosine the cosine schedule if true, else the linear one
     * @param learningRate the Adam learning rate
     * @param labelDropout the share of labels replaced by "no label" while training
     * @param seed the seed for noise and time sampling
     * @return the model
     */
    public static TinyDdpm of(Denoiser net, int steps, boolean cosine, float learningRate, float labelDropout,
            long seed) {
        return new TinyDdpm(net, steps, cosine, learningRate, labelDropout, seed);
    }

    /** @return the number of diffusion steps */
    public int steps() {
        return schedule.steps;
    }

    /**
     * Returns the cumulative signal level {@code abar_t}.
     *
     * @param t the step, 0..T
     * @return {@code prod (1 - beta)}
     */
    public double alphaBar(int t) {
        return schedule.alphaBar[t];
    }

    /**
     * One optimizer step on the epsilon-prediction loss.
     *
     * @param x0 clean samples
     * @param labels their classes (or -1)
     * @return the loss before the step
     */
    public float trainStep(float[][] x0, int[] labels) {
        int n = x0.length;
        int d = net.dimension();
        float[] t = new float[n];
        float[] signal = new float[n];
        float[] noise = new float[n];
        int[] l = labels.clone();
        for (int i = 0; i < n; i++) {
            int step = 1 + random.nextInt(schedule.steps);
            t[i] = (float) step / schedule.steps;
            signal[i] = (float) Math.sqrt(schedule.alphaBar[step]);
            noise[i] = (float) Math.sqrt(1 - schedule.alphaBar[step]);
            if (random.nextDouble() < labelDropout) {
                l[i] = -1;
            }
        }
        Tensor clean = Nets.matrix(x0);
        Tensor eps = Nets.gaussian(n, d, random);
        Tensor xt = clean.mul(Nets.column(signal)).add(eps.mul(Nets.column(noise)));
        Var loss = net.forward(Var.of(xt), t, l).sub(eps).square().mean();
        float value = loss.value().item();
        loss.backward();
        Nets.clip(net.parameters(), 1f);
        optimizer.step(net.parameters());
        return value;
    }

    /**
     * Predicts the noise, with classifier-free guidance when {@code guidance != 1}.
     */
    private float[][] epsilon(float[][] x, int step, int[] labels, float guidance) {
        int n = x.length;
        float[] t = new float[n];
        java.util.Arrays.fill(t, (float) step / schedule.steps);
        Tensor xt = Nets.matrix(x);
        float[][] cond = Nets.rows(net.forward(Var.of(xt), t, labels).value());
        if (guidance == 1f) {
            return cond;
        }
        int[] none = new int[n];
        java.util.Arrays.fill(none, -1);
        float[][] unc = Nets.rows(net.forward(Var.of(xt), t, none).value());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < cond[i].length; j++) {
                cond[i][j] = unc[i][j] + guidance * (cond[i][j] - unc[i][j]);
            }
        }
        return cond;
    }

    private float[][] predictX0(float[][] x, float[][] eps, int step) {
        double ab = schedule.alphaBar[step];
        float[][] x0 = new float[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                double v = (x[i][j] - Math.sqrt(1 - ab) * eps[i][j]) / Math.sqrt(ab);
                x0[i][j] = (float) Math.max(-1, Math.min(1, v));   // data lives in [-1, 1]
            }
        }
        return x0;
    }

    /**
     * Ancestral DDPM sampling through all T steps.
     *
     * @param labels the class of each sample to draw (or -1)
     * @param guidance the classifier-free guidance weight; 1 means none
     * @param seed the noise seed
     * @param trajectory receives the predicted clean samples after each step, or {@code null}
     * @return the samples
     */
    public float[][] sampleDdpm(int[] labels, float guidance, long seed, List<float[][]> trajectory) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = net.dimension();
        float[][] x = Nets.rows(Nets.gaussian(labels.length, d, r));
        for (int step = schedule.steps; step >= 1; step--) {
            float[][] eps = epsilon(x, step, labels, guidance);
            float[][] x0 = predictX0(x, eps, step);
            if (trajectory != null) {
                trajectory.add(x0);
            }
            // posterior mean of q(x_{t-1} | x_t, x0) and its variance
            double ab = schedule.alphaBar[step];
            double abPrev = schedule.alphaBar[step - 1];
            double beta = schedule.beta[step];
            double c0 = Math.sqrt(abPrev) * beta / (1 - ab);
            double ct = Math.sqrt(1 - beta) * (1 - abPrev) / (1 - ab);
            double sigma = Math.sqrt(beta * (1 - abPrev) / (1 - ab));
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < d; j++) {
                    double mean = c0 * x0[i][j] + ct * x[i][j];
                    x[i][j] = (float) (mean + (step > 1 ? sigma * r.nextGaussian() : 0));
                }
            }
        }
        return x;
    }

    /**
     * DDIM sampling with a reduced number of steps.
     *
     * @param labels the class of each sample (or -1)
     * @param steps how many of the T steps to visit, evenly spaced
     * @param eta 0 for the deterministic DDIM, 1 for DDPM-like noise
     * @param guidance the classifier-free guidance weight; 1 means none
     * @param seed the noise seed
     * @param trajectory receives the predicted clean samples after each step, or {@code null}
     * @return the samples
     */
    public float[][] sampleDdim(int[] labels, int steps, float eta, float guidance, long seed,
            List<float[][]> trajectory) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = net.dimension();
        float[][] x = Nets.rows(Nets.gaussian(labels.length, d, r));
        int[] times = new int[steps + 1];
        for (int k = 0; k <= steps; k++) {
            times[k] = Math.round((float) k * schedule.steps / steps);
        }
        for (int k = steps; k >= 1; k--) {
            int step = times[k];
            int prev = times[k - 1];
            float[][] eps = epsilon(x, step, labels, guidance);
            float[][] x0 = predictX0(x, eps, step);
            if (trajectory != null) {
                trajectory.add(x0);
            }
            double ab = schedule.alphaBar[step];
            double abPrev = schedule.alphaBar[prev];
            double sigma = eta * Math.sqrt((1 - abPrev) / (1 - ab)) * Math.sqrt(1 - ab / abPrev);
            double dir = Math.sqrt(Math.max(0, 1 - abPrev - sigma * sigma));
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < d; j++) {
                    // re-derive eps from the clipped x0 so the update stays consistent
                    double e = (x[i][j] - Math.sqrt(ab) * x0[i][j]) / Math.sqrt(1 - ab);
                    x[i][j] = (float) (Math.sqrt(abPrev) * x0[i][j] + dir * e
                            + (prev > 0 ? sigma * r.nextGaussian() : 0));
                }
            }
        }
        return x;
    }

    /**
     * Renders a sampling trajectory as an animated GIF: each frame tiles the predicted clean
     * images of all samples side by side, so the animation shows noise crystallizing into
     * shapes. The last frame is held a little longer.
     *
     * @param trajectory what a sampler recorded, one {@code [samples][side*side]} array per step
     * @param side the image side length
     * @param scale the pixel upscaling factor
     * @return the GIF file's bytes
     */
    public static byte[] denoisingGif(List<float[][]> trajectory, int side, int scale) {
        int n = trajectory.get(0).length;
        int w = n * (side + 1) - 1;
        GifWriter gif = GifWriter.grayscale(w, side).scale(scale).delay(12);
        for (int f = 0; f < trajectory.size() + 4; f++) {
            float[][] frame = trajectory.get(Math.min(f, trajectory.size() - 1));
            float[] px = new float[w * side];
            java.util.Arrays.fill(px, 0);
            for (int s = 0; s < n; s++) {
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        px[y * w + s * (side + 1) + x] = frame[s][y * side + x];
                    }
                }
            }
            gif.frame(px, -1, 1);
        }
        return gif.toBytes();
    }
}
