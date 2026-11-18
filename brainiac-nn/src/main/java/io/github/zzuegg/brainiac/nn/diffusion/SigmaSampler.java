package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.random.RandomGenerator;

/**
 * The samplers of k-diffusion (Karras et al., 2022; Lu et al., 2022): they walk noisy data
 * {@code x0 + sigma noise} down a list of noise levels {@code sigma}, asking a model for the
 * clean data at each. They work with any {@link Denoiser} trained by {@link Diffusion} (see
 * {@link #model}), and need far fewer steps than {@link Diffusion#sample}.
 *
 * <pre>{@code
 * double[] sigmas = SigmaSampler.karras(schedule, 20);
 * Tensor x = Tensor.normal(shape, 0, 1, random).mul((float) sigmas[0]);
 * Tensor samples = SigmaSampler.sample(SigmaSampler.model(unet, schedule, Diffusion.Prediction.NOISE),
 *         x, sigmas, SigmaSampler.Method.DPM_PP_2M, random, SigmaSampler.Listener.NONE);
 * }</pre>
 */
public final class SigmaSampler {

    private SigmaSampler() {
    }

    /** How each step moves to the next noise level. */
    public enum Method {
        /** Euler's method on the probability-flow ODE; deterministic (DDIM with eta 0). */
        EULER,
        /** Euler with fresh noise each step, as DDPM; images keep changing with more steps. */
        EULER_ANCESTRAL,
        /** DPM-Solver++ (2M): a second-order multistep solver, good in 20 to 30 steps. */
        DPM_PP_2M
    }

    /** Predicts the clean data in noisy data at a noise level. */
    @FunctionalInterface
    public interface Model {

        /**
         * @param x the noisy data {@code x0 + sigma noise}
         * @param sigma the noise level
         * @param step the index of the step in the sampler's list
         * @return the predicted clean data, shaped like {@code x}
         */
        Tensor denoise(Tensor x, double sigma, int step);
    }

    /** Hears each step and can stop the run. */
    @FunctionalInterface
    public interface Listener {

        /** A listener that never stops. */
        Listener NONE = (step, steps, denoised) -> true;

        /**
         * @param step the steps done, from 1
         * @param steps all steps
         * @param denoised the current estimate of the clean data; valid only during the call
         * @return {@code false} to stop
         */
        boolean step(int step, int steps, Tensor denoised);
    }

    /**
     * Noise levels evenly spaced in steps from the noisiest step down to step 0, followed by
     * 0.
     *
     * @param schedule the schedule the model was trained with
     * @param steps how many steps
     * @return {@code steps + 1} noise levels
     */
    public static double[] linear(NoiseSchedule schedule, int steps) {
        checkSteps(steps);
        int last = schedule.steps() - 1;
        double[] s = new double[steps + 1];
        for (int i = 0; i < steps; i++) {
            s[i] = schedule.sigma(steps == 1 ? last : last * (double) (steps - 1 - i) / (steps - 1));
        }
        return s;
    }

    /**
     * Karras et al.'s noise levels: evenly spaced in {@code sigma^(1/7)} between the
     * schedule's largest and smallest, followed by 0. They spend more steps at low noise.
     *
     * @param schedule the schedule the model was trained with
     * @param steps how many steps
     * @return {@code steps + 1} noise levels
     */
    public static double[] karras(NoiseSchedule schedule, int steps) {
        return karras(schedule.sigma(0), schedule.sigma(schedule.steps() - 1), steps);
    }

    /**
     * Karras et al.'s noise levels between two bounds, followed by 0.
     *
     * @param min the smallest noise level
     * @param max the largest
     * @param steps how many steps
     * @return {@code steps + 1} noise levels, from {@code max} down to {@code min}, then 0
     */
    public static double[] karras(double min, double max, int steps) {
        checkSteps(steps);
        double rho = 7;
        double maxInv = Math.pow(max, 1 / rho);
        double minInv = Math.pow(min, 1 / rho);
        double[] s = new double[steps + 1];
        for (int i = 0; i < steps; i++) {
            double ramp = steps == 1 ? 0 : (double) i / (steps - 1);
            s[i] = Math.pow(maxInv + ramp * (minInv - maxInv), rho);
        }
        return s;
    }

    private static void checkSteps(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be at least 1, got " + steps);
        }
    }

    /**
     * Wraps a denoiser trained by {@link Diffusion} on {@code schedule}: scales the noisy
     * data back to the variance it was trained on, gives it the (fractional) step of each
     * noise level, and turns its prediction into clean data. Runs without gradients.
     *
     * @param denoiser the trained model
     * @param schedule its schedule
     * @param prediction what it predicts
     * @return the model for {@link #sample}
     */
    public static Model model(Denoiser denoiser, NoiseSchedule schedule, Diffusion.Prediction prediction) {
        Objects.requireNonNull(denoiser, "denoiser");
        Objects.requireNonNull(prediction, "prediction");
        return (x, sigma, step) -> {
            int batch = x.shape().dim(0);
            float[] t = new float[batch];
            java.util.Arrays.fill(t, (float) schedule.timestep(sigma));
            Tensor time = Tensor.of(Shape.of(batch), t, x.device());
            double scale = 1 / Math.sqrt(sigma * sigma + 1);
            Tensor out = Var.noGrad(() -> denoiser.apply(Var.of(x.mul((float) scale)), time).value());
            return denoised(x, out, sigma, prediction);
        };
    }

    /**
     * Turns a model's output on {@code x / sqrt(sigma^2 + 1)} into clean data.
     *
     * @param x the noisy data {@code x0 + sigma noise}
     * @param out the model's output
     * @param sigma the noise level
     * @param prediction what the model predicts
     * @return the clean data
     */
    public static Tensor denoised(Tensor x, Tensor out, double sigma, Diffusion.Prediction prediction) {
        Tensor o = out.to(x.dtype());
        return switch (prediction) {
            case NOISE -> x.sub(o.mul((float) sigma));
            case SAMPLE -> o;
            case VELOCITY -> x.mul((float) (1 / (sigma * sigma + 1)))
                    .sub(o.mul((float) (sigma / Math.sqrt(sigma * sigma + 1))));
        };
    }

    /**
     * Runs a sampler.
     *
     * @param model predicts clean data
     * @param x the start: noise at the first level, such as unit noise times {@code sigmas[0]}
     * @param sigmas the noise levels, decreasing, ending with 0
     * @param method how to step
     * @param random the source of noise for {@link Method#EULER_ANCESTRAL}
     * @param listener hears each step
     * @return the clean data
     * @throws CancellationException if the listener stopped the run
     */
    public static Tensor sample(Model model, Tensor x, double[] sigmas, Method method, RandomGenerator random,
            Listener listener) {
        int steps = sigmas.length - 1;
        if (steps < 1) {
            throw new IllegalArgumentException("give at least two noise levels");
        }
        Tensor previous = null;   // the last step's denoised data, for DPM-Solver++ (2M)
        Tensor start = x;
        for (int i = 0; i < steps; i++) {
            double sigma = sigmas[i];
            double next = sigmas[i + 1];
            double last = i > 0 ? sigmas[i - 1] : 0;
            Tensor xi = x;
            int step = i;
            Tensor denoised = scoped(() -> model.denoise(xi, sigma, step));
            Tensor prev = previous;
            Tensor nextX = scoped(() -> switch (method) {
                case EULER -> euler(xi, denoised, sigma, next);
                case EULER_ANCESTRAL -> {
                    double up = Math.min(next, Math.sqrt(next * next * (sigma * sigma - next * next)
                            / (sigma * sigma)));
                    double down = Math.sqrt(next * next - up * up);
                    Tensor y = euler(xi, denoised, sigma, down);
                    yield next > 0 ? y.add(gaussian(xi, random).mul((float) up)) : y;
                }
                case DPM_PP_2M -> dpmSolver2m(xi, denoised, prev, sigma, next, last);
            });
            if (previous != null && previous != nextX) {
                previous.release();
            }
            previous = denoised;
            if (x != start) {
                x.release();
            }
            x = nextX;
            if (!listener.step(i + 1, steps, denoised)) {
                if (x != start) {
                    x.release();
                }
                if (denoised != x) {
                    denoised.release();
                }
                throw new CancellationException("sampling stopped at step " + (i + 1));
            }
        }
        // DPM-Solver++ ends on the denoised data itself
        if (previous != null && previous != x) {
            previous.release();
        }
        return x;
    }

    /** Computes {@code body} and frees everything it made except the result. */
    private static Tensor scoped(java.util.function.Supplier<Tensor> body) {
        Tensor[] out = new Tensor[1];
        java.util.List<Tensor> made = Tensor.created(() -> out[0] = body.get());
        Tensor.evaluate(java.util.List.of(out[0]));
        Tensor.releaseCreated(made, java.util.List.of(out[0]));
        return out[0];
    }

    private static Tensor gaussian(Tensor like, RandomGenerator random) {
        float[] v = new float[Math.toIntExact(like.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) random.nextGaussian();
        }
        return Tensor.of(like.shape(), v, like.device()).to(like.dtype());
    }

    private static Tensor euler(Tensor x, Tensor denoised, double sigma, double next) {
        // x + (next - sigma) * (x - denoised) / sigma
        return denoised.add(x.sub(denoised).mul((float) (next / sigma)));
    }

    private static Tensor dpmSolver2m(Tensor x, Tensor denoised, Tensor previous, double sigma, double next,
            double last) {
        if (next == 0) {
            return denoised;
        }
        double t = -Math.log(sigma);
        double tNext = -Math.log(next);
        double h = tNext - t;
        Tensor d = denoised;
        if (previous != null) {
            double hLast = t + Math.log(last);
            double r = hLast / h;
            d = denoised.mul((float) (1 + 1 / (2 * r))).sub(previous.mul((float) (1 / (2 * r))));
        }
        return x.mul((float) (next / sigma)).sub(d.mul((float) Math.expm1(-h)));
    }
}
