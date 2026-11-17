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
