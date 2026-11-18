package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.ml.LogDensity;

import java.util.SplittableRandom;

/**
 * Two-dimensional lattice {@code phi^4} field theory on an {@code L x L} periodic lattice, in the
 * parameterisation of Albergo, Kanwar and Shanahan ("Flow-based generative models for Markov
 * chain Monte Carlo in lattice field theory", PRD 2019,
 * <a href="https://arxiv.org/abs/1904.12072">arXiv:1904.12072</a>):
 * {@code S(phi) = sum_x [ sum_mu phi_x (2 phi_x - phi_(x+mu) - phi_(x-mu)) + m^2 phi_x^2 + lambda phi_x^4 ]}.
 * With {@code m^2 = -4} the quadratic terms cancel and {@code lambda} moves the theory from the
 * symmetric phase (large lambda) through the critical region (their {@code L = 6} ensemble uses
 * {@code lambda = 6.975}) to the broken phase (small lambda).
 *
 * <p>It gives the action and its gradient, the magnetisation and susceptibility, a local
 * Metropolis sampler (the physics baseline, which slows down near criticality), the integrated
 * autocorrelation time with Sokal's automatic window, and the target as a {@link LogDensity}
 * for Hamiltonian Monte Carlo.
 */
public final class LatticePhi4 {

    private final int l;
    private final double m2;
    private final double lambda;

    /**
     * @param l the lattice side
     * @param m2 the bare mass squared
     * @param lambda the quartic coupling
     */
    public LatticePhi4(int l, double m2, double lambda) {
        this.l = l;
        this.m2 = m2;
        this.lambda = lambda;
    }

    /** @return the lattice side */
    public int side() {
        return l;
    }

    /** @return the number of sites */
    public int volume() {
        return l * l;
    }

    /** @return {@code m^2} */
    public double massSquared() {
        return m2;
    }

    /** @return {@code lambda} */
    public double lambda() {
        return lambda;
    }

    private int site(int x, int y) {
        return ((y + l) % l) * l + (x + l) % l;
    }

    /**
     * The action.
     *
     * @param phi the field, {@code [L * L]} row-major
     * @return {@code S(phi)}
     */
    public double action(double[] phi) {
        double s = 0;
        for (int y = 0; y < l; y++) {
            for (int x = 0; x < l; x++) {
                double p = phi[site(x, y)];
                double p2 = p * p;
                s += (4 + m2) * p2 + lambda * p2 * p2 - 2 * p * (phi[site(x + 1, y)] + phi[site(x, y + 1)]);
            }
        }
        return s;
    }

    /**
     * The action's gradient.
     *
     * @param phi the field
     * @param grad receives {@code dS/dphi}
     */
    public void gradient(double[] phi, double[] grad) {
        for (int y = 0; y < l; y++) {
            for (int x = 0; x < l; x++) {
                double p = phi[site(x, y)];
                double nb = phi[site(x + 1, y)] + phi[site(x - 1, y)] + phi[site(x, y + 1)] + phi[site(x, y - 1)];
                grad[site(x, y)] = 2 * (4 + m2) * p + 4 * lambda * p * p * p - 2 * nb;
            }
        }
    }

    /**
     * The magnetisation per site.
     *
     * @param phi the field
     * @return {@code M / V}
     */
    public static double magnetization(double[] phi) {
        double s = 0;
        for (double p : phi) {
            s += p;
        }
        return s / phi.length;
    }

    /**
     * The two-point susceptibility {@code V (<M^2> - <|M|>^2)} from per-site magnetisations.
     *
     * @param m magnetisations per site, one per sample
     * @param volume the number of sites
     * @return the susceptibility
     */
    public static double susceptibility(double[] m, int volume) {
        double a = 0;
        double b = 0;
        for (double v : m) {
            a += v * v;
            b += Math.abs(v);
        }
        a /= m.length;
        b /= m.length;
        return volume * (a - b * b);
    }

    /**
     * The target density {@code exp(-S)} for HMC.
     *
     * @return the log density and its gradient
     */
    public LogDensity logDensity() {
        return (x, g) -> {
            if (g != null) {
                gradient(x, g);
                for (int i = 0; i < g.length; i++) {
                    g[i] = -g[i];
                }
            }
            return -action(x);
        };
    }

    /**
     * Local Metropolis sweeps: each site in turn proposes {@code phi_x + U(-delta, delta)}.
     *
     * @param phi the field (updated)
     * @param sweeps sweeps to run
     * @param delta the proposal half-width
     * @param random the random source
     * @return the acceptance rate
     */
    public double metropolis(double[] phi, int sweeps, double delta, SplittableRandom random) {
        long acc = 0;
        for (int s = 0; s < sweeps; s++) {
            for (int y = 0; y < l; y++) {
                for (int x = 0; x < l; x++) {
                    int i = site(x, y);
                    double p = phi[i];
                    double q = p + delta * (2 * random.nextDouble() - 1);
                    double nb = phi[site(x + 1, y)] + phi[site(x - 1, y)] + phi[site(x, y + 1)] + phi[site(x, y - 1)];
                    double ds = (4 + m2) * (q * q - p * p) + lambda * (q * q * q * q - p * p * p * p) - 2 * (q - p) * nb;
                    if (ds <= 0 || random.nextDouble() < Math.exp(-ds)) {
                        phi[i] = q;
                        acc++;
                    }
                }
            }
        }
        return (double) acc / ((long) sweeps * l * l);
    }

    /**
     * The integrated autocorrelation time {@code tau = 1/2 + sum_t rho(t)} with Sokal's
     * automatic window (the smallest {@code W >= 5 tau(W)}).
     *
     * @param series the chain of an observable
     * @return {@code tau_int} in units of the chain's steps
     */
    public static double integratedAutocorrelation(double[] series) {
        int n = series.length;
        double mean = 0;
        for (double v : series) {
            mean += v;
        }
        mean /= n;
        double c0 = 0;
        for (double v : series) {
            c0 += (v - mean) * (v - mean);
        }
        c0 /= n;
        if (c0 == 0) {
            return 0.5;
        }
        double tau = 0.5;
        for (int t = 1; t < n / 2; t++) {
            double c = 0;
            for (int i = 0; i + t < n; i++) {
                c += (series[i] - mean) * (series[i + t] - mean);
            }
            c /= (n - t) * c0;
            tau += c;
            if (t >= 5 * tau) {
                break;
            }
        }
        return Math.max(0.5, tau);
    }
}
