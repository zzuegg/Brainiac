package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * EDM (Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models",
 * <a href="https://arxiv.org/abs/2206.00364">arXiv:2206.00364</a>): a denoiser
 * {@code D(x; sigma)} for data corrupted as {@code x = x0 + sigma n}, wrapped around a raw
 * network {@code F} with the preconditioning
 * {@code D = c_skip x + c_out F(c_in x; c_noise)}, where
 * {@code c_skip = sd^2 / (sigma^2 + sd^2)}, {@code c_out = sigma sd / sqrt(sigma^2 + sd^2)},
 * {@code c_in = 1 / sqrt(sigma^2 + sd^2)} and {@code c_noise = ln(sigma) / 4}. The network's
 * input and target then have unit variance at every noise level, so one network covers
 * {@code sigma} from 0.002 to 80. Training draws {@code ln sigma ~ N(P_mean, P_std)} and uses
 * the loss weight that makes the effective target of {@code F} unit-weighted; sampling uses
 * the Karras {@code rho = 7} noise schedule with Heun's second-order method (2 network
 * evaluations per step, one for the last).
 *
 * <p>It is a core {@link Denoiser} whose "time" is {@code sigma} per row, so it plugs into
 * {@link DpmSolver} and into distillation methods that need a teacher denoiser.
 */
public final class EdmDenoiser implements Denoiser {
    private final Denoiser network;
    private final List<Var> parameters;
    private final float sigmaData;
    private float meanLogSigma = -1.2f;
    private float stdLogSigma = 1.2f;

    private EdmDenoiser(Denoiser network, List<Var> parameters, float sigmaData) {
        this.network = network;
        this.parameters = List.copyOf(parameters);
        this.sigmaData = sigmaData;
    }

    /**
     * Wraps a raw network {@code F(x_in, c_noise)}.
     *
     * @param network the raw network; its "time" input receives {@code c_noise = ln(sigma) / 4}
     * @param parameters its trainable parameters
     * @param sigmaData the data standard deviation, 0.5 in the paper
     * @return the denoiser
     */
    public static EdmDenoiser of(Denoiser network, List<Var> parameters, float sigmaData) {
        return new EdmDenoiser(network, parameters, sigmaData);
    }

    /**
     * Sets the log-normal training distribution of the noise level.
     *
     * @param mean {@code P_mean}, -1.2 in the paper
     * @param std {@code P_std}, 1.2 in the paper
     * @return this denoiser
     */
    public EdmDenoiser trainingNoise(float mean, float std) {
        this.meanLogSigma = mean;
        this.stdLogSigma = std;
        return this;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return parameters;
    }

    /** @return the data standard deviation */
    public float sigmaData() {
        return sigmaData;
    }

    private Tensor[] coefficients(float[] sigma) {
        int n = sigma.length;
        float[] skip = new float[n];
        float[] out = new float[n];
        float[] in = new float[n];
        float[] noise = new float[n];
        double sd2 = (double) sigmaData * sigmaData;
        for (int i = 0; i < n; i++) {
            double s2 = (double) sigma[i] * sigma[i];
            skip[i] = (float) (sd2 / (s2 + sd2));
            out[i] = (float) (sigma[i] * sigmaData / Math.sqrt(s2 + sd2));
            in[i] = (float) (1 / Math.sqrt(s2 + sd2));
            noise[i] = (float) (Math.log(Math.max(sigma[i], 1e-8)) / 4);
        }
        return new Tensor[] {GenerativeOps.column(skip), GenerativeOps.column(out), GenerativeOps.column(in),
            Tensor.of(Shape.of(n), noise)};
    }

    /**
     * The denoised estimate {@code D(x; sigma)}.
     *
     * @param noisy {@code x0 + sigma n}, {@code [batch, dim]}
     * @param sigma the noise level per row {@code [batch]}
     * @return the estimate of {@code x0}
     */
    @Override
    public Var apply(Var noisy, Tensor sigma) {
        Tensor[] c = coefficients(sigma.toArray());
        return noisy.mul(c[0]).add(network.apply(noisy.mul(c[2]), c[3]).mul(c[1]));
    }

    /**
     * The EDM training loss on a batch.
     *
     * @param x0 clean data {@code [batch, dim]}
     * @param random the source of noise levels and noise
     * @return the scalar loss
     */
    public Var loss(Tensor x0, RandomGenerator random) {
        int b = x0.shape().dim(0);
        float[] sigma = new float[b];
        for (int i = 0; i < b; i++) {
            sigma[i] = (float) Math.exp(meanLogSigma + stdLogSigma * random.nextGaussian());
        }
        Tensor[] c = coefficients(sigma);
        Tensor noisy = x0.add(Tensor.normal(x0.shape(), 0, 1, random).mul(GenerativeOps.column(sigma)));
        Tensor target = x0.sub(noisy.mul(c[0])).div(c[1]);
        return network.apply(Var.of(noisy.mul(c[2])), c[3]).sub(Var.of(target)).square().mean();
    }

    /**
     * The Karras noise schedule: {@code n} levels from {@code sigmaMax} down to {@code sigmaMin},
     * evenly spaced in {@code sigma^(1/rho)}, followed by a final 0.
     *
     * @param n the number of nonzero levels, at least 1
     * @param sigmaMin the smallest level
     * @param sigmaMax the largest level
     * @param rho the warping, 7 in the paper
     * @return {@code n + 1} levels
     */
    public static double[] karrasSigmas(int n, double sigmaMin, double sigmaMax, double rho) {
        double[] s = new double[n + 1];
        double a = Math.pow(sigmaMax, 1 / rho);
        double b = Math.pow(sigmaMin, 1 / rho);
        for (int i = 0; i < n; i++) {
            s[i] = Math.pow(a + (n == 1 ? 0 : (double) i / (n - 1)) * (b - a), rho);
        }
        s[n] = 0;
        return s;
    }

    private Tensor denoise(Tensor x, double sigma) {
        int b = x.shape().dim(0);
        return apply(Var.of(x), Tensor.full(Shape.of(b), (float) sigma)).value();
    }

    /**
     * Deterministic sampling of the probability-flow ODE {@code dx/dsigma = (x - D(x; sigma)) / sigma}
     * with Heun's method (or Euler), from {@code x = sigma_0 * noise}.
     *
     * @param noise standard normal noise {@code [batch, dim]}
     * @param sigmas the levels, descending and ending in 0 (see {@link #karrasSigmas})
     * @param heun Heun's second-order correction on every step but the last
     * @return the samples
     */
    public Tensor sample(Tensor noise, double[] sigmas, boolean heun) {
        Tensor x = noise.mul((float) sigmas[0]);
        for (int i = 0; i + 1 < sigmas.length; i++) {
            double s = sigmas[i];
            double next = sigmas[i + 1];
            Tensor d = x.sub(denoise(x, s)).div((float) s);
            Tensor euler = x.add(d.mul((float) (next - s)));
            if (heun && next > 0) {
                Tensor d2 = euler.sub(denoise(euler, next)).div((float) next);
                x = x.add(d.add(d2).mul((float) (0.5 * (next - s))));
            } else {
                x = euler;
            }
        }
        return x;
    }

    /**
     * Network evaluations used by {@link #sample} for a schedule.
     *
     * @param sigmas the levels
     * @param heun whether Heun's correction is used
     * @return the evaluations
     */
    public static int evaluations(double[] sigmas, boolean heun) {
        int steps = sigmas.length - 1;
        return heun ? 2 * steps - 1 : steps;
    }
}
