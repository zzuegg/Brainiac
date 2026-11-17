package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * A consistency model trained from scratch with consistency training (Song et al.,
 * <a href="https://arxiv.org/abs/2303.01469">2303.01469</a>, with the improvements of
 * <a href="https://arxiv.org/abs/2310.14189">2310.14189</a>): a network
 * {@code f(x, sigma) = c_skip(sigma) x + c_out(sigma) F(c_in(sigma) x, sigma)} around any
 * {@link ConditionalDenoiser} {@code F} (it sees the time {@code log(sigma / sigmaMin) / log(sigmaMax / sigmaMin)}) that maps
 * any point on a noising trajectory straight to its clean end point. The boundary
 * condition {@code f(x, sigma_min) = x} holds by construction of {@code c_skip, c_out}.
 *
 * <p>Training needs no pretrained diffusion model: take a data point {@code x}, one noise
 * draw {@code z} and two adjacent noise levels {@code sigma_n < sigma_n+1} on the Karras
 * grid, and pull {@code f(x + sigma_n+1 z)} towards the stop-gradient target
 * {@code f(x + sigma_n z)} with a pseudo-Huber loss. The grid gets finer over training
 * (the discretisation curriculum). Generation is then one network call on pure noise
 * ({@link #sampleOneStep}), optionally refined by a few noise-and-denoise rounds
 * ({@link #sampleMultiStep}).
 */
public final class ConsistencyModel {
    private static final double SIGMA_DATA = 0.5;
    private final ConditionalDenoiser net;
    private final Optimizer optimizer;
    private final double sigmaMin;
    private final double sigmaMax;
    private final SplittableRandom random;
    private int trained;

    private ConsistencyModel(ConditionalDenoiser net, double sigmaMin, double sigmaMax, float learningRate, long seed) {
        this.net = net;
        this.sigmaMin = sigmaMin;
        this.sigmaMax = sigmaMax;
        this.optimizer = Optimizer.adam(learningRate);
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates a model.
     *
     * @param net the network {@code F}
     * @param sigmaMin the smallest noise level, where {@code f} is the identity
     * @param sigmaMax the largest noise level, where sampling starts
     * @param learningRate the Adam learning rate
     * @param seed the seed for noise and levels
     * @return the model
     */
    public static ConsistencyModel of(ConditionalDenoiser net, double sigmaMin, double sigmaMax, float learningRate, long seed) {
        return new ConsistencyModel(net, sigmaMin, sigmaMax, learningRate, seed);
    }

    /** {@code c_skip}: 1 at sigma_min, falling towards 0 for large sigma. */
    static double cSkip(double sigma, double sigmaMin) {
        double d = sigma - sigmaMin;
        return SIGMA_DATA * SIGMA_DATA / (d * d + SIGMA_DATA * SIGMA_DATA);
    }

    /** {@code c_out}: 0 at sigma_min. */
    static double cOut(double sigma, double sigmaMin) {
        return SIGMA_DATA * (sigma - sigmaMin) / Math.sqrt(SIGMA_DATA * SIGMA_DATA + sigma * sigma);
    }

    private float time(double sigma) {
        return (float) (Math.log(sigma / sigmaMin) / Math.log(sigmaMax / sigmaMin));
    }

    /** The Karras et al. noise grid with N levels (rho = 7). */
    double[] grid(int levels) {
        double[] s = new double[levels];
        double a = Math.pow(sigmaMin, 1 / 7.0);
        double b = Math.pow(sigmaMax, 1 / 7.0);
        for (int i = 0; i < levels; i++) {
            s[i] = Math.pow(a + (double) i / (levels - 1) * (b - a), 7);
        }
        return s;
    }

    /**
     * Applies the consistency function.
     *
     * @param x noisy points {@code [n, d]}
     * @param sigma the noise level of each row
     * @param labels labels, or -1
     * @return the predicted clean points, as a graph value
     */
    Var apply(Var x, double[] sigma, int[] labels) {
        int n = sigma.length;
        float[] skip = new float[n];
        float[] out = new float[n];
        float[] in = new float[n];
        float[] t = new float[n];
        for (int i = 0; i < n; i++) {
            skip[i] = (float) cSkip(sigma[i], sigmaMin);
            out[i] = (float) cOut(sigma[i], sigmaMin);
            in[i] = (float) (1 / Math.sqrt(sigma[i] * sigma[i] + SIGMA_DATA * SIGMA_DATA));
            t[i] = time(sigma[i]);
        }
        Var f = net.apply(x.mul(GenBlocks.column(in)), GenBlocks.times(t), labels);
        return x.mul(GenBlocks.column(skip)).add(f.mul(GenBlocks.column(out)));
    }

    /**
     * Number of grid levels at a point of training: doubles from 10 up to 80 over the run.
     */
    int levels(int step, int totalSteps) {
        int doublings = 3;
        int stage = Math.min(doublings, step * (doublings + 1) / Math.max(1, totalSteps));
        return 10 << stage;
    }

    /**
     * One consistency-training step.
     *
     * @param data clean points
     * @param labels labels, or -1
     * @param totalSteps the planned number of training steps, for the curriculum
     * @return the loss before the step
     */
    public float trainStep(float[][] data, int[] labels, int totalSteps) {
        int n = data.length;
        double[] sigmas = grid(levels(trained++, totalSteps));
        double[] lo = new double[n];
        double[] hi = new double[n];
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            int k = random.nextInt(sigmas.length - 1);
            lo[i] = sigmas[k];
            hi[i] = sigmas[k + 1];
            w[i] = (float) (1 / (hi[i] - lo[i]));   // iCT weighting
        }
        Tensor x = GenBlocks.matrix(data);
        Tensor z = GenBlocks.gaussian(n, net.dimension(), random);
        Tensor xHi = x.add(z.mul(GenBlocks.column(toFloat(hi))));
        Tensor xLo = x.add(z.mul(GenBlocks.column(toFloat(lo))));
        Tensor target = apply(Var.of(xLo), lo, labels).value();   // stop-gradient teacher
        Var diff = apply(Var.of(xHi), hi, labels).sub(target);
        // pseudo-Huber: sqrt(|d|^2 + c^2) - c, robust and smooth
        float c = 0.03f * (float) Math.sqrt(net.dimension());
        Var dist = diff.square().sum(1, true).add(c * c).sqrt().sub(c);
        Var loss = dist.mul(GenBlocks.column(w)).mean();
        float value = loss.value().item();
        loss.backward();
        GenBlocks.clip(net.parameters(), 1f);
        optimizer.step(net.parameters());
        return value;
    }

    /**
     * Generates in one network evaluation: {@code f(sigma_max * z, sigma_max)}.
     *
     * @param labels labels, or -1
     * @param seed the noise seed
     * @return the samples
     */
    public float[][] sampleOneStep(int[] labels, long seed) {
        return sampleMultiStep(labels, new double[0], seed);
    }

    /**
     * Generates with extra refinement rounds: after the one-step sample, add noise of each
     * given level and map back to clean again.
     *
     * @param labels labels, or -1
     * @param sigmas the intermediate noise levels, decreasing
     * @param seed the noise seed
     * @return the samples
     */
    public float[][] sampleMultiStep(int[] labels, double[] sigmas, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = labels.length;
        double[] level = new double[n];
        java.util.Arrays.fill(level, sigmaMax);
        Tensor x = GenBlocks.gaussian(n, net.dimension(), r).mul((float) sigmaMax);
        Tensor clean = apply(Var.of(x), level, labels).value();
        for (double s : sigmas) {
            java.util.Arrays.fill(level, s);
            float noise = (float) Math.sqrt(Math.max(0, s * s - sigmaMin * sigmaMin));
            x = clean.add(GenBlocks.gaussian(n, net.dimension(), r).mul(noise));
            clean = apply(Var.of(x), level, labels).value();
        }
        return GenBlocks.rows(clean);
    }

    /**
     * Applies the trained function to points at a noise level, for checks and denoising.
     *
     * @param x the points
     * @param sigma their noise level
     * @param labels labels, or -1
     * @return the predicted clean points
     */
    public float[][] denoise(float[][] x, double sigma, int[] labels) {
        double[] s = new double[x.length];
        java.util.Arrays.fill(s, sigma);
        return GenBlocks.rows(apply(Var.of(GenBlocks.matrix(x)), s, labels).value());
    }

    private static float[] toFloat(double[] d) {
        float[] f = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }
}
