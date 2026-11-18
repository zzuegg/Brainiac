package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;

/**
 * Flow matching (rectified flow), as Flux, Stable Diffusion 3 and newer image models use:
 * data and noise are joined by straight lines, {@code x(sigma) = (1 - sigma) * data + sigma
 * * noise}, and the model predicts the velocity {@code noise - data} along them. Sampling
 * walks from {@code sigma = 1} (pure noise) to 0 with Euler steps, usually on a schedule
 * shifted towards high noise for large images.
 *
 * <pre>{@code
 * double[] sigmas = FlowSampler.dynamicSigmas(4, FlowSampler.mu(imageTokens));
 * Tensor latents = FlowSampler.sample(FlowSampler.model(dit.denoiser(text, pooled, 3.5f)), noise, sigmas,
 *         SigmaSampler.Listener.NONE);
 *
 * Var loss = FlowSampler.loss(myModel, batch, random);   // training a flow model of your own
 * }</pre>
 */
public final class FlowSampler {

    private FlowSampler() {
    }

    /** A velocity model at one noise level. */
    @FunctionalInterface
    public interface Model {

        /**
         * Predicts the velocity.
         *
         * @param x the current sample
         * @param sigma its noise level in {@code [0, 1]}
         * @param step the sampling step, from 0
         * @return the velocity {@code noise - data}, shaped like {@code x}
         */
        Tensor velocity(Tensor x, double sigma, int step);
    }

    /**
     * A {@link Denoiser} that predicts the velocity, as a sampling model: it runs without
     * gradients, with every example at the same noise level.
     *
     * @param velocity the model
     * @return the sampling model
     */
    public static Model model(Denoiser velocity) {
        return (x, sigma, step) -> Var.noGrad(() -> {
            float[] s = new float[x.shape().dim(0)];
            Arrays.fill(s, (float) sigma);
            return velocity.apply(Var.of(x), Tensor.of(Shape.of(s.length), s, x.device())).value();
        });
    }

    /**
     * {@code steps} evenly spaced noise levels from 1 down, shifted by a constant factor
     * (Stable Diffusion 3 uses 3, Flux schnell 1), then 0.
     *
     * @param steps the number of steps
     * @param shift the shift; above 1 spends more steps at high noise
     * @return {@code steps + 1} noise levels
     */
    public static double[] sigmas(int steps, double shift) {
        double[] s = linear(steps);
        for (int i = 0; i < steps; i++) {
            s[i] = (float) (shift * s[i] / (1 + (shift - 1) * s[i]));
        }
        return s;
    }

    /**
     * {@code steps} evenly spaced noise levels from 1 down, shifted by {@code e^mu}
     * exponentially as Flux dev does for its image size, then 0.
     *
     * @param steps the number of steps
     * @param mu the shift exponent, see {@link #mu(int)}
     * @return {@code steps + 1} noise levels
     */
    public static double[] dynamicSigmas(int steps, double mu) {
        double[] s = linear(steps);
        double e = Math.exp(mu);
        for (int i = 0; i < steps; i++) {
            s[i] = (float) (e / (e + (1 / s[i] - 1)));
        }
        return s;
    }

    private static double[] linear(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }
        double[] s = new double[steps + 1];
        for (int i = 0; i < steps; i++) {
            double l = steps == 1 ? 1 : 1 + (1.0 / steps - 1) * i / (steps - 1);
            s[i] = (float) l;
        }
        return s;
    }

    /**
     * Flux's shift exponent for an image of {@code imageTokens} patches: 0.5 at 256 patches,
     * rising linearly to 1.15 at 4096.
     *
     * @param imageTokens the patches of the image
     * @return the exponent
     */
    public static double mu(int imageTokens) {
        return mu(imageTokens, 256, 4096, 0.5, 1.15);
    }

    /**
     * A shift exponent rising linearly with the image size.
     *
     * @param imageTokens the patches of the image
     * @param baseTokens the size at which the exponent is {@code baseShift}
     * @param maxTokens the size at which it is {@code maxShift}
     * @param baseShift the exponent at {@code baseTokens}
     * @param maxShift the exponent at {@code maxTokens}
     * @return the exponent
     */
    public static double mu(int imageTokens, int baseTokens, int maxTokens, double baseShift, double maxShift) {
        double m = (maxShift - baseShift) / (maxTokens - baseTokens);
        return imageTokens * m + (baseShift - m * baseTokens);
    }

    /**
     * Samples by Euler steps along the predicted velocity.
     *
     * @param model the velocity model
     * @param noise the start, pure noise at {@code sigmas[0]} (not released)
     * @param sigmas the noise levels, falling, usually ending at 0
     * @param listener told after each step with the data estimate {@code x - sigma * v}; it
     *     can stop the sampling
     * @return the sample at the last noise level, in float32
     * @throws CancellationException if the listener stopped the sampling
     */
    public static Tensor sample(Model model, Tensor noise, double[] sigmas, SigmaSampler.Listener listener) {
        int steps = sigmas.length - 1;
        if (steps < 1) {
            throw new IllegalArgumentException("give at least two noise levels");
        }
        Tensor x = noise;
        for (int i = 0; i < steps; i++) {
            double sigma = sigmas[i];
            double next = sigmas[i + 1];
            Tensor xi = x;
            int step = i;
            Tensor v = scoped(() -> model.velocity(xi, sigma, step).to(DType.FLOAT32));
            Tensor nextX = scoped(() -> xi.to(DType.FLOAT32).add(v.mul((float) (next - sigma))));
            Tensor denoised = scoped(() -> xi.to(DType.FLOAT32).sub(v.mul((float) sigma)));
            v.release();
            if (x != noise) {
                x.release();
            }
            x = nextX;
            boolean go = listener.step(i + 1, steps, denoised);
            denoised.release();
            if (!go) {
                x.release();
                throw new CancellationException("sampling stopped at step " + (i + 1));
            }
        }
        return x;
    }

    /**
     * Noise levels as diffusers' UniPC scheduler sets them for flow models (Wan): {@code steps}
     * levels evenly spaced from 1 down to {@code 1 / 1000} (the last left out), shifted by
     * {@code shift}, the first nudged below 1, then 0.
     *
     * @param steps the number of steps
     * @param shift the shift; Wan uses 3 to 12
     * @return {@code steps + 1} noise levels, rounded to float32
     */
    public static double[] uniPcSigmas(int steps, double shift) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }
        double[] s = new double[steps + 1];
        for (int i = 0; i < steps; i++) {
            double l = 1 + (1.0 / 1000 - 1) * i / steps;
            double v = shift * l / (1 + (shift - 1) * l);
            if (i == 0 && Math.abs(v - 1) < 1e-6) {
                v -= 1e-6;
            }
            s[i] = (float) v;
        }
        return s;
    }

    /**
     * Samples with UniPC (Zhao et al., 2023), the multistep predictor-corrector diffusers and
     * Wan use for flow models: second order (B(h) = e^h - 1, data prediction), a corrector after
     * every step but the first, first order at the first and the last step. It reuses each
     * model call twice, so it needs no more calls than Euler steps but far fewer steps.
     *
     * @param model the velocity model
     * @param noise the start, pure noise at {@code sigmas[0]} (not released)
     * @param sigmas the noise levels, falling, ending at 0 (see {@link #uniPcSigmas})
     * @param listener told after each step with the data estimate; it can stop the sampling
     * @return the sample at the last noise level, in float32
     * @throws CancellationException if the listener stopped the sampling
     */
    public static Tensor sampleUniPc(Model model, Tensor noise, double[] sigmas, SigmaSampler.Listener listener) {
        int steps = sigmas.length - 1;
        if (steps < 1) {
            throw new IllegalArgumentException("give at least two noise levels");
        }
        Tensor x = noise;
        Tensor m1 = null;     // the data estimate of the step before
        Tensor m2 = null;     // and of the one before that
        Tensor last = null;   // the sample the last predictor started from
        int lastOrder = 0;
        try {
            for (int i = 0; i < steps; i++) {
                Tensor xi = x;
                int step = i;
                double sigma = sigmas[i];
                Tensor v = scoped(() -> model.velocity(xi, sigma, step).to(DType.FLOAT32));
                Tensor m = scoped(() -> xi.to(DType.FLOAT32).sub(v.mul((float) sigma)));
                v.release();
                if (last != null) {
                    // the corrector: redo the last step knowing this step's estimate
                    Tensor ls = last;
                    Tensor e1 = m1;
                    Tensor e2 = m2;
                    int order = lastOrder;
                    Tensor corrected = scoped(() -> correct(sigmas, step, ls, xi, m, e1, e2, order));
                    if (x != noise) {
                        x.release();
                    }
                    x = corrected;
                }
                // first order at the start and (lower_order_final) at the last step
                int order = Math.min(Math.min(2, steps - i), i + 1);
                if (last != null && last != noise) {
                    last.release();
                }
                last = x;
                Tensor from = x;
                Tensor e1 = m1;
                Tensor next = scoped(() -> predict(sigmas, step, from, m, e1, order));
                if (m2 != null) {
                    m2.release();
                }
                m2 = m1;
                m1 = m;
                lastOrder = order;
                x = next;
                if (!listener.step(i + 1, steps, m)) {
                    throw new CancellationException("sampling stopped at step " + (i + 1));
                }
            }
            Tensor result = x;
            x = null;
            return result;
        } finally {
            for (Tensor t : new Tensor[] {m1, m2, last == x ? null : last, x}) {
                if (t != null && t != noise) {
                    t.release();
                }
            }
        }
    }

    private static double lambda(double sigma) {
        return Math.log(1 - sigma) - Math.log(sigma);
    }

    /**
     * UniP: from {@code x} at level {@code i} to level {@code i + 1} with the data estimate
     * {@code m0} there and, for second order, {@code m1} of the level before.
     */
    private static Tensor predict(double[] s, int i, Tensor x, Tensor m0, Tensor m1, int order) {
        double sigmaT = s[i + 1];
        double sigmaS = s[i];
        double alphaT = 1 - sigmaT;
        double h = lambda(sigmaT) - lambda(sigmaS);
        double hh = -h;
        double phi1 = Math.expm1(hh);
        double bh = Math.expm1(hh);
        // sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        Tensor out = m0.mul((float) (-alphaT * phi1));
        if (sigmaT != 0) {
            out = x.to(DType.FLOAT32).mul((float) (sigmaT / sigmaS)).add(out);
        }
        if (order == 2) {
            double rk = (lambda(s[i - 1]) - lambda(sigmaS)) / h;
            // D1 = (m1 - m0) / rk, weighted by rho = 0.5
            out = out.sub(m1.sub(m0).mul((float) (alphaT * bh * 0.5 / rk)));
        }
        return out;
    }

    /**
     * UniC: corrects the step from {@code last} (level {@code i - 1}) to {@code sample} (level
     * {@code i}) with the new estimate {@code m} at level {@code i}; {@code m1} and {@code m2}
     * are the estimates at levels {@code i - 1} and {@code i - 2}.
     */
    private static Tensor correct(double[] s, int i, Tensor last, Tensor sample, Tensor m, Tensor m1, Tensor m2,
            int order) {
        double sigmaT = s[i];
        double sigmaS = s[i - 1];
        double alphaT = 1 - sigmaT;
        double h = lambda(sigmaT) - lambda(sigmaS);
        double hh = -h;
        double phi1 = Math.expm1(hh);
        double bh = Math.expm1(hh);
        Tensor base = last.to(DType.FLOAT32).mul((float) (sigmaT / sigmaS)).add(m1.mul((float) (-alphaT * phi1)));
        Tensor d1t = m.sub(m1);
        if (order == 1) {
            return base.sub(d1t.mul((float) (alphaT * bh * 0.5)));
        }
        double rk = (lambda(s[i - 2]) - lambda(sigmaS)) / h;
        double k1 = phi1 / hh - 1;
        double b1 = k1 / bh;
        double k2 = k1 / hh - 0.5;
        double b2 = k2 * 2 / bh;
        // solve [[1, 1], [rk, 1]] rho = [b1, b2]
        double r0 = (b1 - b2) / (1 - rk);
        double r1 = b1 - r0;
        Tensor d1 = m2.sub(m1).mul((float) (r0 / rk));
        return base.sub(d1.add(d1t.mul((float) r1)).mul((float) (alphaT * bh)));
    }

    /**
     * The flow-matching loss of a velocity model on a batch: each example gets a noise level
     * drawn uniformly from {@code [0, 1]} and fresh Gaussian noise, and the model is scored
     * by the mean squared error of its velocity.
     *
     * @param model the model, taking the noisy batch and the noise levels {@code [batch]}
     * @param data the batch, {@code [batch, ...]}
     * @param random the source of noise levels and noise
     * @return the loss
     */
    public static Var loss(Denoiser model, Tensor data, RandomGenerator random) {
        int b = data.shape().dim(0);
        float[] s = new float[b];
        for (int i = 0; i < b; i++) {
            s[i] = random.nextFloat();
        }
        Tensor sigmas = Tensor.of(Shape.of(b), s, data.device());
        int[] dims = new int[data.shape().rank()];
        Arrays.fill(dims, 1);
        dims[0] = b;
        Tensor level = sigmas.reshape(dims);
        Tensor noise = Tensor.normal(data.shape(), 0, 1, random).to(data.device()).to(data.dtype());
        Tensor noisy = data.mul(level.neg().add(1f)).add(noise.mul(level));
        Var v = model.apply(Var.of(noisy), sigmas);
        return v.sub(Var.of(noise.sub(data))).square().mean();
    }

    /** Computes {@code body} and frees everything it made except the result. */
    private static Tensor scoped(Supplier<Tensor> body) {
        Tensor[] out = new Tensor[1];
        List<Tensor> made = Tensor.created(() -> out[0] = body.get());
        Tensor.evaluate(List.of(out[0]));
        Tensor.releaseCreated(made, List.of(out[0]));
        return out[0];
    }
}
