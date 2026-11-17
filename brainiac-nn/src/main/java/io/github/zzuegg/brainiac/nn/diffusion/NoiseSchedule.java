package io.github.zzuegg.brainiac.nn.diffusion;

import java.util.Arrays;

/**
 * How much of the data is left at each step of a diffusion: {@code alphaBar[t]}, the product
 * of {@code 1 - beta} up to step {@code t}. Noisy data at step {@code t} is
 * {@code sqrt(alphaBar[t]) x0 + sqrt(1 - alphaBar[t]) noise}.
 *
 * @param alphaBar the share of the data's variance left after each step, decreasing
 */
public record NoiseSchedule(double[] alphaBar) {

    /** Checks the values. */
    public NoiseSchedule {
        if (alphaBar.length == 0) {
            throw new IllegalArgumentException("a schedule needs at least one step");
        }
        alphaBar = alphaBar.clone();
        for (int t = 0; t < alphaBar.length; t++) {
            if (!(alphaBar[t] > 0 && alphaBar[t] <= 1) || t > 0 && alphaBar[t] > alphaBar[t - 1]) {
                throw new IllegalArgumentException("alphaBar must decrease within (0, 1]: " + Arrays.toString(alphaBar));
            }
        }
    }

    /**
     * Betas rising linearly, as in DDPM (Ho et al., 2020): 1e-4 to 0.02 over 1000 steps.
     *
     * @param steps the number of steps
     * @param first the first beta
     * @param last the last beta
     * @return the schedule
     */
    public static NoiseSchedule linear(int steps, double first, double last) {
        double[] a = new double[steps];
        double prod = 1;
        for (int t = 0; t < steps; t++) {
            double beta = steps == 1 ? first : first + (last - first) * t / (steps - 1);
            prod *= 1 - beta;
            a[t] = prod;
        }
        return new NoiseSchedule(a);
    }

    /**
     * Betas whose square roots rise linearly, as latent diffusion and Stable Diffusion train
     * with (0.00085 to 0.012 over 1000 steps), computed in float32 like diffusers.
     *
     * @param steps the number of steps
     * @param first the first beta
     * @param last the last beta
     * @return the schedule
     */
    public static NoiseSchedule scaledLinear(int steps, double first, double last) {
        double[] a = new double[steps];
        double prod = 1;
        for (int t = 0; t < steps; t++) {
            double f = steps == 1 ? 0 : (double) t / (steps - 1);
            double beta = (float) Math.pow(Math.sqrt(first) + f * (Math.sqrt(last) - Math.sqrt(first)), 2);
            prod *= 1 - beta;
            a[t] = (float) prod;
        }
        return new NoiseSchedule(a);
    }

    /**
     * The cosine schedule of Nichol and Dhariwal (2021), with betas capped at 0.999.
     *
     * @param steps the number of steps
     * @return the schedule
     */
    public static NoiseSchedule cosine(int steps) {
        double s = 0.008;
        double[] a = new double[steps];
        double f0 = Math.pow(Math.cos(s / (1 + s) * Math.PI / 2), 2);
        double prev = 1;
        for (int t = 0; t < steps; t++) {
            double f = Math.pow(Math.cos(((t + 1.0) / steps + s) / (1 + s) * Math.PI / 2), 2) / f0;
            double beta = Math.min(0.999, 1 - f / prev);
            a[t] = (t == 0 ? 1 : a[t - 1]) * (1 - beta);
            prev = f;
        }
        return new NoiseSchedule(a);
    }

    /** @return the number of steps */
    public int steps() {
        return alphaBar.length;
    }

    /**
     * @param t the step
     * @return the share of the data's variance left after step {@code t}
     */
    public double alphaBar(int t) {
        return alphaBar[t];
    }

    /**
     * The noise level of step {@code t} in the variance-exploding view of k-diffusion
     * samplers: {@code sqrt((1 - alphaBar) / alphaBar)}, so noisy data at step {@code t} is
     * {@code (x0 + sigma noise) sqrt(alphaBar)}.
     *
     * @param t the step
     * @return the noise level
     */
    public double sigma(int t) {
        return Math.sqrt((1 - alphaBar[t]) / alphaBar[t]);
    }

    /**
     * The noise level at a fractional step, interpolated linearly between steps.
     *
     * @param t the step, in {@code [0, steps - 1]}
     * @return the noise level
     */
    public double sigma(double t) {
        int lo = (int) Math.floor(t);
        int hi = Math.min(lo + 1, alphaBar.length - 1);
        double f = t - lo;
        return (1 - f) * sigma(lo) + f * sigma(hi);
    }

    /**
     * The fractional step with a noise level, interpolating the log noise levels: the
     * inverse of {@link #sigma(int)}, as k-diffusion samplers tell a denoiser trained on
     * steps where they are.
     *
     * @param sigma the noise level
     * @return the step, in {@code [0, steps - 1]}
     */
    public double timestep(double sigma) {
        int n = alphaBar.length;
        if (n == 1) {
            return 0;
        }
        double logSigma = Math.log(Math.max(sigma, 1e-10));
        int low = 0;
        for (int i = 0; i < n; i++) {
            if (logSigma - Math.log(sigma(i)) >= 0) {
                low = i;
            }
        }
        low = Math.min(low, n - 2);
        int high = low + 1;
        double a = Math.log(sigma(low));
        double b = Math.log(sigma(high));
        double w = Math.max(0, Math.min(1, (a - logSigma) / (a - b)));
        return (1 - w) * low + w * high;
    }

    @Override
    public double[] alphaBar() {
        return alphaBar.clone();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof NoiseSchedule n && Arrays.equals(alphaBar, n.alphaBar);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(alphaBar);
    }

    @Override
    public String toString() {
        return "NoiseSchedule[" + alphaBar.length + " steps, alphaBar " + alphaBar[0] + " to "
                + alphaBar[alphaBar.length - 1] + "]";
    }
}
