package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Denoising diffusion (Ho et al., 2020) over a {@link NoiseSchedule}: {@link #loss} trains
 * a {@link Denoiser} on noisy copies of the data, and {@link #sample} turns noise into data with
 * DDIM steps (Song et al., 2021); {@code eta = 1} gives DDPM's random steps, {@code eta = 0}
 * deterministic ones, and fewer steps than the schedule's skip evenly.
 *
 * <pre>{@code
 * Diffusion d = new Diffusion(NoiseSchedule.cosine(1000), Diffusion.Prediction.NOISE);
 * Var loss = d.loss(model, images, random);          // train on this
 * Tensor samples = d.sample(model, Shape.of(16, 3, 32, 32), 50, 0, random, device);
 * }</pre>
 */
public final class Diffusion {

    /** What the denoiser is trained to predict. */
    public enum Prediction {
        /** The noise that was added (DDPM). */
        NOISE,
        /** The clean data. */
        SAMPLE,
        /** {@code sqrt(alphaBar) noise - sqrt(1 - alphaBar) x0} (Salimans and Ho, 2022). */
        VELOCITY
    }

    private final NoiseSchedule schedule;
    private final Prediction prediction;
    /** While sampling, predicted clean data is clamped to {@code [-clip, clip]}; 0 for none. */
    private final float clip;

    /**
     * @param schedule the noise levels
     * @param prediction what the denoiser predicts
     */
    public Diffusion(NoiseSchedule schedule, Prediction prediction) {
        this(schedule, prediction, 0);
    }

    private Diffusion(NoiseSchedule schedule, Prediction prediction, float clip) {
        this.schedule = Objects.requireNonNull(schedule, "schedule");
        this.prediction = Objects.requireNonNull(prediction, "prediction");
        this.clip = clip;
    }

    /**
     * This diffusion with the clean data predicted at each sampling step clamped to
     * {@code [-limit, limit]}, as DDPM does for data scaled to [-1, 1]: it keeps early,
     * noisy predictions from overshooting.
     *
     * @param limit the bound, positive; 0 turns clamping off
     * @return the diffusion
     */
    public Diffusion clipSample(float limit) {
        if (!(limit >= 0)) {
            throw new IllegalArgumentException("limit must not be negative: " + limit);
        }
        return new Diffusion(schedule, prediction, limit);
    }

    /** @return the noise levels */
    public NoiseSchedule schedule() {
        return schedule;
    }

    /** @return what the denoiser predicts */
    public Prediction prediction() {
        return prediction;
    }

    /**
     * Noisy data at given steps: {@code sqrt(alphaBar[t]) x0 + sqrt(1 - alphaBar[t]) noise}.
     *
     * @param x0 the data, batch first
     * @param noise standard normal noise of the same shape
     * @param steps per example, the step
     * @return the noisy data
     */
    public Tensor noisy(Tensor x0, Tensor noise, int[] steps) {
        double[] a = new double[steps.length];
        double[] b = new double[steps.length];
        for (int i = 0; i < steps.length; i++) {
            a[i] = Math.sqrt(schedule.alphaBar(steps[i]));
            b[i] = Math.sqrt(1 - schedule.alphaBar(steps[i]));
        }
        return x0.mul(perExample(a, x0)).add(noise.mul(perExample(b, x0)));
    }

    /**
     * Noisy sequences with a step per token, as diffusion forcing trains: token {@code j} of
     * example {@code i} gets the noise of step {@code steps[i][j]}.
     *
     * @param x0 the data, {@code [batch, tokens, ...]}
     * @param noise standard normal noise of the same shape
     * @param steps {@code [batch][tokens]} steps
     * @return the noisy data
     */
    public Tensor noisy(Tensor x0, Tensor noise, int[][] steps) {
        double[][] a = new double[steps.length][];
        double[][] b = new double[steps.length][];
        for (int i = 0; i < steps.length; i++) {
            a[i] = new double[steps[i].length];
            b[i] = new double[steps[i].length];
            for (int j = 0; j < steps[i].length; j++) {
                a[i][j] = Math.sqrt(schedule.alphaBar(steps[i][j]));
                b[i][j] = Math.sqrt(1 - schedule.alphaBar(steps[i][j]));
            }
        }
        return x0.mul(perToken(a, x0)).add(noise.mul(perToken(b, x0)));
    }

    /**
     * Independent random steps for every token, as diffusion forcing (Chen et al. 2024)
     * samples them, so one model learns to denoise any mix of clean and noisy tokens.
     *
     * @param batch the examples
     * @param tokens the tokens per example
     * @param random the source of steps
     * @return {@code [batch][tokens]} steps
     */
    public int[][] randomTokenSteps(int batch, int tokens, RandomGenerator random) {
        int[][] steps = new int[batch][tokens];
        for (int[] row : steps) {
            for (int j = 0; j < tokens; j++) {
                row[j] = random.nextInt(schedule.steps());
            }
        }
        return steps;
    }

    /**
     * The training loss with a step per token (diffusion forcing): the denoiser receives the
     * steps as a {@code [batch, tokens]} time tensor and must accept that form.
     *
     * @param model the denoiser
     * @param x0 clean sequences, {@code [batch, tokens, ...]}
     * @param steps {@code [batch][tokens]} steps, such as {@link #randomTokenSteps}
     * @param random the source of noise
     * @return the loss, a scalar
     */
    public Var loss(Denoiser model, Tensor x0, int[][] steps, RandomGenerator random) {
        int batch = x0.shape().dim(0);
        int tokens = x0.shape().dim(1);
        if (steps.length != batch || steps[0].length != tokens) {
            throw new IllegalArgumentException("steps must be [" + batch + "][" + tokens + "]");
        }
        float[] time = new float[batch * tokens];
        double[][] a = new double[batch][tokens];
        double[][] b = new double[batch][tokens];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < tokens; j++) {
                time[i * tokens + j] = steps[i][j];
                a[i][j] = Math.sqrt(schedule.alphaBar(steps[i][j]));
                b[i][j] = Math.sqrt(1 - schedule.alphaBar(steps[i][j]));
            }
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random).to(x0.device()).to(x0.dtype());
        Tensor xt = noisy(x0, noise, steps);
        Tensor target = switch (prediction) {
            case NOISE -> noise;
            case SAMPLE -> x0;
            case VELOCITY -> noise.mul(perToken(a, x0)).sub(x0.mul(perToken(b, x0)));
        };
        Var out = model.apply(Var.of(xt), Tensor.of(Shape.of(batch, tokens), time, x0.device()));
        return out.sub(Var.of(target)).square().mean();
    }

    /**
     * The training loss for a batch: a random step per example, noise added, and the mean
     * squared error of the denoiser's prediction.
     *
     * @param model the denoiser
     * @param x0 clean data, batch first
     * @param random the source of steps and noise
     * @return the loss, a scalar
     */
    public Var loss(Denoiser model, Tensor x0, RandomGenerator random) {
        int batch = x0.shape().dim(0);
        int[] steps = new int[batch];
        float[] time = new float[batch];
        for (int i = 0; i < batch; i++) {
            steps[i] = random.nextInt(schedule.steps());
            time[i] = steps[i];
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random).to(x0.device()).to(x0.dtype());
        Tensor xt = noisy(x0, noise, steps);
        Tensor target = switch (prediction) {
            case NOISE -> noise;
            case SAMPLE -> x0;
            case VELOCITY -> {
                double[] a = new double[batch];
                double[] b = new double[batch];
                for (int i = 0; i < batch; i++) {
                    a[i] = Math.sqrt(schedule.alphaBar(steps[i]));
                    b[i] = Math.sqrt(1 - schedule.alphaBar(steps[i]));
                }
                yield noise.mul(perExample(a, x0)).sub(x0.mul(perExample(b, x0)));
            }
        };
        Var out = model.apply(Var.of(xt), Tensor.of(Shape.of(batch), time, x0.device()));
        return out.sub(Var.of(target)).square().mean();
    }

    /**
     * Turns standard normal noise into data.
     *
     * @param model the trained denoiser
     * @param shape the samples' shape, batch first
     * @param steps how many denoising steps, at most the schedule's
     * @param eta 0 for deterministic DDIM steps, 1 for DDPM's random ones
     * @param random the source of the starting noise and of the random steps
     * @param device where to compute
     * @return the samples
     */
    public Tensor sample(Denoiser model, Shape shape, int steps, float eta, RandomGenerator random, Device device) {
        Tensor x = Tensor.normal(shape, 0, 1, random).to(device);
        return sample(model, x, steps, eta, random);
    }

    /**
     * Denoises given noise: {@link #sample(Denoiser, Shape, int, float, RandomGenerator, Device)}
     * from a chosen start.
     *
     * @param model the trained denoiser
     * @param noise the start, standard normal, batch first
     * @param steps how many denoising steps, at most the schedule's
     * @param eta 0 for deterministic steps, 1 for DDPM's random ones
     * @param random the source of the random steps
     * @return the samples
     */
    public Tensor sample(Denoiser model, Tensor noise, int steps, float eta, RandomGenerator random) {
        int total = schedule.steps();
        if (steps < 1 || steps > total) {
            throw new IllegalArgumentException("steps must be in 1.." + total + ", got " + steps);
        }
        int batch = noise.shape().dim(0);
        Tensor x = noise;
        for (int i = 0; i < steps; i++) {
            int t = (int) Math.round((double) (total - 1) * (steps - 1 - i) / Math.max(1, steps - 1));
            if (steps == 1) {
                t = total - 1;
            }
            int s = i + 1 < steps ? (int) Math.round((double) (total - 1) * (steps - 2 - i) / Math.max(1, steps - 1))
                    : -1;
            double at = schedule.alphaBar(t);
            double as = s < 0 ? 1 : schedule.alphaBar(s);
            float[] time = new float[batch];
            java.util.Arrays.fill(time, t);
            Tensor out = model.apply(Var.of(x), Tensor.of(Shape.of(batch), time, x.device())).value();
            Tensor x0;
            Tensor eps;
            float sa = (float) Math.sqrt(at);
            float sb = (float) Math.sqrt(1 - at);
            switch (prediction) {
                case NOISE -> {
                    eps = out;
                    x0 = x.sub(eps.mul(sb)).div(sa);
                }
                case SAMPLE -> {
                    x0 = out;
                    eps = x.sub(x0.mul(sa)).div(sb);
                }
                default -> {
                    x0 = x.mul(sa).sub(out.mul(sb));
                    eps = x.mul(sb).add(out.mul(sa));
                }
            }
            if (clip > 0) {
                Tensor lo = Tensor.full(Shape.of(), -clip, x0.dtype(), x0.device());
                Tensor hi = Tensor.full(Shape.of(), clip, x0.dtype(), x0.device());
                x0 = x0.maximum(lo).minimum(hi);
                eps = x.sub(x0.mul(sa)).div(sb);   // the noise that goes with the clamped data
            }
            double sigma = eta * Math.sqrt((1 - as) / (1 - at)) * Math.sqrt(Math.max(0, 1 - at / as));
            Tensor next = x0.mul((float) Math.sqrt(as)).add(eps.mul((float) Math.sqrt(Math.max(0, 1 - as - sigma * sigma))));
            if (sigma > 0) {
                next = next.add(Tensor.normal(x.shape(), 0, (float) sigma, random).to(x.device()).to(x.dtype()));
            }
            x = next;
        }
        return x;
    }

    /** {@code [batch][tokens]} values shaped to broadcast over {@code like}'s later axes. */
    static Tensor perToken(double[][] values, Tensor like) {
        int[] dims = new int[like.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = values.length;
        dims[1] = values[0].length;
        float[] f = new float[dims[0] * dims[1]];
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                f[i * dims[1] + j] = (float) values[i][j];
            }
        }
        return Tensor.of(Shape.of(dims), f, like.device()).to(like.dtype());
    }

    /** One value per example, shaped to broadcast over the rest of {@code like}. */
    static Tensor perExample(double[] values, Tensor like) {
        int[] dims = new int[like.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = values.length;
        float[] f = new float[values.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) values[i];
        }
        return Tensor.of(Shape.of(dims), f, like.device()).to(like.dtype());
    }
}
