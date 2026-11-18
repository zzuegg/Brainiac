package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;

import java.util.Map;

/**
 * The noise levels a diffusion model was trained with, and the timesteps and noise levels
 * (sigmas) a sampler visits for a given number of steps. Follows diffusers' discrete
 * schedulers: betas over 1000 training steps, timesteps spaced "leading", "trailing" or
 * "linspace", optionally Karras-spaced sigmas.
 */
public final class NoiseSchedule {

    /** What the model predicts. */
    public enum Prediction {
        /** The noise that was added. */
        EPSILON,
        /** The velocity {@code alpha * noise - sigma * x0}. */
        V_PREDICTION
    }

    /** How inference timesteps are spread over the training steps. */
    public enum Spacing {
        /** {@code 0, r, 2r, ...} plus an offset, from the top: what Stable Diffusion shipped with. */
        LEADING,
        /** Down from the last training step in equal strides. */
        TRAILING,
        /** Evenly from the last training step to the first. */
        LINSPACE
    }

    /**
     * The steps of one sampling run.
     *
     * @param timesteps the model's timestep at each step
     * @param sigmas the noise level at each step, plus a final 0
     * @param initialSigma the scale of the starting noise
     */
    public record Plan(float[] timesteps, double[] sigmas, double initialSigma) {

        /**
         * Returns the number of steps.
         *
         * @return the number of model evaluations of a one-step-per-evaluation sampler
         */
        public int steps() {
            return timesteps.length;
        }
    }

    private final double[] alphasCumprod;
    private final double[] logSigmas;
    private final Prediction prediction;
    private final Spacing spacing;
    private final int stepsOffset;

    /**
     * Creates a schedule.
     *
     * @param trainSteps the training steps (1000)
     * @param betaStart the first beta
     * @param betaEnd the last beta
     * @param scaledLinear whether betas are linear in their square root (Stable Diffusion) or linear
     * @param prediction what the model predicts
     * @param spacing how timesteps are spread
     * @param stepsOffset added to leading timesteps
     */
    public NoiseSchedule(int trainSteps, double betaStart, double betaEnd, boolean scaledLinear,
            Prediction prediction, Spacing spacing, int stepsOffset) {
        alphasCumprod = new double[trainSteps];
        logSigmas = new double[trainSteps];
        double prod = 1;
        for (int i = 0; i < trainSteps; i++) {
            double f = trainSteps == 1 ? 0 : (double) i / (trainSteps - 1);
            double beta = scaledLinear
                    ? Math.pow(Math.sqrt(betaStart) + f * (Math.sqrt(betaEnd) - Math.sqrt(betaStart)), 2)
                    : betaStart + f * (betaEnd - betaStart);
            // diffusers computes the betas in float32
            beta = (float) beta;
            prod *= 1 - beta;
            alphasCumprod[i] = (float) prod;
            logSigmas[i] = Math.log(Math.sqrt((1 - alphasCumprod[i]) / alphasCumprod[i]));
        }
        this.prediction = prediction;
        this.spacing = spacing;
        this.stepsOffset = stepsOffset;
    }

    /**
     * Stable Diffusion's schedule: scaled-linear betas from 0.00085 to 0.012 over 1000 steps,
     * leading spacing with offset 1.
     *
     * @param prediction what the model predicts
     * @return the schedule
     */
    public static NoiseSchedule stableDiffusion(Prediction prediction) {
        return new NoiseSchedule(1000, 0.00085, 0.012, true, prediction, Spacing.LEADING, 1);
    }

    /**
     * Reads diffusers' {@code scheduler/scheduler_config.json}.
     *
     * @param c the parsed config
     * @return the schedule
     */
    public static NoiseSchedule fromJson(Map<String, Object> c) {
        String spacing = Json.string(c, "timestep_spacing", "leading");
        return new NoiseSchedule((int) Json.integer(c, "num_train_timesteps", 1000),
                Json.number(c, "beta_start", 0.00085), Json.number(c, "beta_end", 0.012),
                !"linear".equals(Json.string(c, "beta_schedule", "scaled_linear")),
                "v_prediction".equals(Json.string(c, "prediction_type", "epsilon")) ? Prediction.V_PREDICTION
                        : Prediction.EPSILON,
                switch (spacing) {
                    case "trailing" -> Spacing.TRAILING;
                    case "linspace" -> Spacing.LINSPACE;
                    default -> Spacing.LEADING;
                }, (int) Json.integer(c, "steps_offset", 1));
    }

    /**
     * Returns what the model predicts.
     *
     * @return the prediction type
     */
    public Prediction prediction() {
        return prediction;
    }

    /**
     * Returns the noise level of a training timestep.
     *
     * @param t the timestep, {@code 0..trainSteps-1}
     * @return {@code sqrt((1 - abar) / abar)}
     */
    public double sigma(int t) {
        return Math.exp(logSigmas[t]);
    }

    /**
     * Plans a sampling run.
     *
     * @param steps the number of steps
     * @param karras whether to space sigmas as Karras et al. do ({@code rho = 7}) instead of by timestep
     * @return the timesteps and sigmas
     */
    public Plan plan(int steps, boolean karras) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be at least 1, got " + steps);
        }
        int train = alphasCumprod.length;
        double[] t = new double[steps];
        switch (spacing) {
            case LEADING -> {
                int ratio = train / steps;
                for (int i = 0; i < steps; i++) {
                    t[i] = (double) (steps - 1 - i) * ratio + stepsOffset;
                }
            }
            case TRAILING -> {
                double ratio = (double) train / steps;
                for (int i = 0; i < steps; i++) {
                    t[i] = Math.round(train - i * ratio) - 1;
                }
            }
            case LINSPACE -> {
                for (int i = 0; i < steps; i++) {
                    t[i] = steps == 1 ? 0 : (train - 1) * (double) (steps - 1 - i) / (steps - 1);
                }
            }
        }
        double[] sigmas = new double[steps + 1];
        for (int i = 0; i < steps; i++) {
            sigmas[i] = interpolatedSigma(t[i]);
        }
        if (karras) {
            double max = sigmas[0];
            double min = sigmas[steps - 1];
            double rho = 7;
            double maxInv = Math.pow(max, 1 / rho);
            double minInv = Math.pow(min, 1 / rho);
            for (int i = 0; i < steps; i++) {
                double ramp = steps == 1 ? 0 : (double) i / (steps - 1);
                sigmas[i] = Math.pow(maxInv + ramp * (minInv - maxInv), rho);
                t[i] = timestep(sigmas[i]);
            }
        }
        sigmas[steps] = 0;
        float[] ts = new float[steps];
        for (int i = 0; i < steps; i++) {
            ts[i] = (float) t[i];
        }
        double init = spacing == Spacing.LEADING ? Math.sqrt(sigmas[0] * sigmas[0] + 1) : sigmas[0];
        return new Plan(ts, sigmas, init);
    }

    /** The sigma at a fractional timestep, linear between training steps. */
    private double interpolatedSigma(double t) {
        int lo = (int) Math.floor(t);
        int hi = Math.min(lo + 1, logSigmas.length - 1);
        double f = t - lo;
        return (1 - f) * sigma(lo) + f * sigma(hi);
    }

    /** The fractional timestep of a sigma, by interpolating the log-sigmas (diffusers' {@code _sigma_to_t}). */
    double timestep(double sigma) {
        double logSigma = Math.log(Math.max(sigma, 1e-10));
        int n = logSigmas.length;
        int low = 0;
        for (int i = 0; i < n; i++) {
            if (logSigma - logSigmas[i] >= 0) {
                low = i;
            }
        }
        low = Math.min(low, n - 2);
        int high = low + 1;
        double w = (logSigmas[low] - logSigma) / (logSigmas[low] - logSigmas[high]);
        w = Math.max(0, Math.min(1, w));
        return (1 - w) * low + w * high;
    }
}
