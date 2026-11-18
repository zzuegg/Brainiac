package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Simulated bifurcation (Goto, Tatsumura and Dixon, Science Advances 2019; Goto et al.,
 * Science Advances 2021), a quantum-inspired Ising solver: each spin is a continuous position
 * {@code x_i} with momentum {@code y_i}, integrated by symplectic Euler while a pump
 * amplitude ramps up, so every {@code x_i} bifurcates to one of two branches whose signs
 * minimise the Ising energy {@code E(s) = -Σ_(i<j) J_ij s_i s_j - Σ_i h_i s_i}. All spins move
 * in parallel.
 *
 * <ul>
 *   <li>{@link Variant#ADIABATIC} (aSB, 2019): Kerr oscillators,
 *       {@code ẏ = -(x² - p(t) + Δ) x + ξ0 (J x + h)}, {@code ẋ = (Δ + p(t)) y},
 *       {@code p} ramped from 0 to {@code 1 + Δ};</li>
 *   <li>{@link Variant#BALLISTIC} (bSB, 2021): no Kerr term but inelastic walls at
 *       {@code |x| = 1}, {@code ẏ = -(a0 - a(t)) x + c0 (J x + h)}, {@code ẋ = a0 y};</li>
 *   <li>{@link Variant#DISCRETE} (dSB, 2021): as bSB with the couplings acting on
 *       {@code sign(x)}, which escapes the analog errors of bSB.</li>
 * </ul>
 * Couplings are held as sparse neighbour lists. {@link #anneal} is a plain single-spin
 * Metropolis simulated-annealing reference on the same instance.
 */
public final class SimulatedBifurcation {

    /** The simulated-bifurcation equations. */
    public enum Variant {
        /** adiabatic SB */
        ADIABATIC,
        /** ballistic SB */
        BALLISTIC,
        /** discrete SB */
        DISCRETE
    }

    /**
     * A solver's answer.
     *
     * @param spins the best state found, {@code ±1}
     * @param energy its Ising energy
     * @param nanos the wall time
     */
    public record Solution(int[] spins, double energy, long nanos) {
    }

    private final int n;
    private final int[][] neighbours;
    private final double[][] couplings;
    private final double[] field;
    private final double scale;

    private SimulatedBifurcation(double[][] j, double[] h) {
        this.n = h.length;
        this.field = h.clone();
        this.neighbours = new int[n][];
        this.couplings = new double[n][];
        double squares = 0;
        for (int i = 0; i < n; i++) {
            List<Integer> idx = new ArrayList<>();
            for (int k = 0; k < n; k++) {
                if (k != i && j[i][k] != 0) {
                    if (j[i][k] != j[k][i]) {
                        throw new IllegalArgumentException("couplings must be symmetric");
                    }
                    idx.add(k);
                    squares += j[i][k] * j[i][k];
                }
            }
            neighbours[i] = idx.stream().mapToInt(Integer::intValue).toArray();
            couplings[i] = new double[neighbours[i].length];
            for (int m = 0; m < neighbours[i].length; m++) {
                couplings[i][m] = j[i][neighbours[i][m]];
            }
        }
        double sigma = Math.sqrt(squares / Math.max(1.0, (double) n * (n - 1)));
        double unit = Math.max(sigma * Math.sqrt(n), spectralRadius() / 2);
        this.scale = unit > 0 ? 1 / unit : 1;
    }

    /**
     * The largest absolute eigenvalue of {@code J} by power iteration. Goto et al. scale the
     * coupling by {@code 1 / (σ √N)}, right for zero-mean couplings; couplings with a large
     * mean (dense MaxCut graphs) have one eigenvalue far above that, which would make the
     * synchronous update oscillate, so the scale is capped by it.
     */
    private double spectralRadius() {
        double[] v = new double[n];
        java.util.Arrays.fill(v, 1 / Math.sqrt(Math.max(1, n)));
        double[] w = new double[n];
        double lambda = 0;
        for (int it = 0; it < 30; it++) {
            double norm = 0;
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int m = 0; m < neighbours[i].length; m++) {
                    s += couplings[i][m] * v[neighbours[i][m]];
                }
                w[i] = s;
                norm += s * s;
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                return 0;
            }
            for (int i = 0; i < n; i++) {
                v[i] = w[i] / norm;
            }
            lambda = norm;
        }
        return lambda;
    }

    /**
     * Builds a solver.
     *
     * @param j symmetric couplings with a zero diagonal
     * @param h the fields
     * @return the solver
     */
    public static SimulatedBifurcation of(double[][] j, double[] h) {
        return new SimulatedBifurcation(j, h);
    }

    /** @return the number of spins */
    public int size() {
        return n;
    }

    /**
     * The Ising energy.
     *
     * @param s spins {@code ±1}
     * @return {@code E(s)}
     */
    public double energy(int[] s) {
        double e = 0;
        for (int i = 0; i < n; i++) {
            e -= field[i] * s[i];
            for (int m = 0; m < neighbours[i].length; m++) {
                if (neighbours[i][m] > i) {
                    e -= couplings[i][m] * s[i] * s[neighbours[i][m]];
                }
            }
        }
        return e;
    }

    private void force(double[] x, boolean discrete, double[] out) {
        for (int i = 0; i < n; i++) {
            double f = field[i];
            int[] nb = neighbours[i];
            double[] c = couplings[i];
            for (int m = 0; m < nb.length; m++) {
                double v = x[nb[m]];
                f += c[m] * (discrete ? Math.signum(v) : v);
            }
            out[i] = f;
        }
    }

    /**
     * Runs one simulated-bifurcation trajectory from small random positions.
     *
     * @param variant the equations
     * @param steps integration steps
     * @param dt the time step (about 0.5 for aSB, 1 for bSB and dSB)
     * @param seed the seed
     * @return the signs of the final positions (the best state seen at the checkpoints)
     */
    public Solution solve(Variant variant, int steps, double dt, long seed) {
        long start = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = r.nextDouble(-0.1, 0.1);
            y[i] = r.nextDouble(-0.1, 0.1);
        }
        double[] f = new double[n];
        int[] best = null;
        double bestEnergy = Double.POSITIVE_INFINITY;
        double delta = 1;
        double a0 = 1;
        double xi = 0.7 * delta * scale;
        double c0 = 0.5 * scale;
        int checkEvery = Math.max(1, steps / 10);
        for (int t = 0; t < steps; t++) {
            double progress = t / (double) steps;
            if (variant == Variant.ADIABATIC) {
                double p = progress * (1 + delta);
                force(x, false, f);
                for (int i = 0; i < n; i++) {
                    y[i] += (-(x[i] * x[i] - p + delta) * x[i] + xi * f[i]) * dt;
                    x[i] += (delta + p) * y[i] * dt;
                }
            } else {
                double a = progress * a0;
                force(x, variant == Variant.DISCRETE, f);
                for (int i = 0; i < n; i++) {
                    y[i] += (-(a0 - a) * x[i] + c0 * f[i]) * dt;
                    x[i] += a0 * y[i] * dt;
                    if (Math.abs(x[i]) > 1) {
                        x[i] = Math.signum(x[i]);
                        y[i] = 0;
                    }
                }
            }
            if ((t + 1) % checkEvery == 0 || t == steps - 1) {
                int[] s = signs(x);
                double e = energy(s);
                if (e < bestEnergy) {
                    bestEnergy = e;
                    best = s;
                }
            }
        }
        return new Solution(best, bestEnergy, System.nanoTime() - start);
    }

    private static int[] signs(double[] x) {
        int[] s = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            s[i] = x[i] >= 0 ? 1 : -1;
        }
        return s;
    }

    /**
     * Simulated annealing: single-spin Metropolis sweeps with a geometric inverse-temperature
     * schedule, keeping the best state.
     *
     * @param sweeps sweeps over all spins
     * @param betaStart the initial inverse temperature
     * @param betaEnd the final inverse temperature
     * @param seed the seed
     * @return the best state
     */
    public Solution anneal(int sweeps, double betaStart, double betaEnd, long seed) {
        long start = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        int[] s = new int[n];
        for (int i = 0; i < n; i++) {
            s[i] = r.nextBoolean() ? 1 : -1;
        }
        double[] local = new double[n];
        for (int i = 0; i < n; i++) {
            double f = field[i];
            for (int m = 0; m < neighbours[i].length; m++) {
                f += couplings[i][m] * s[neighbours[i][m]];
            }
            local[i] = f;
        }
        double e = energy(s);
        double bestEnergy = e;
        int[] best = s.clone();
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double beta = betaStart * Math.pow(betaEnd / betaStart, sweep / (double) Math.max(1, sweeps - 1));
            for (int i = 0; i < n; i++) {
                double change = 2 * s[i] * local[i];
                if (change <= 0 || r.nextDouble() < Math.exp(-beta * change)) {
                    s[i] = -s[i];
                    e += change;
                    for (int m = 0; m < neighbours[i].length; m++) {
                        local[neighbours[i][m]] += 2 * couplings[i][m] * s[i];
                    }
                }
            }
            if (e < bestEnergy) {
                bestEnergy = e;
                best = s.clone();
            }
        }
        return new Solution(best, bestEnergy, System.nanoTime() - start);
    }
}
