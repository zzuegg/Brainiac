package io.github.zzuegg.brainiac.experimental.quantum;

import java.util.SplittableRandom;
import java.util.function.DoubleFunction;

/**
 * A "foundation" neural quantum state: one network {@code psi_theta(sigma; h)} for a whole
 * family of Hamiltonians {@code H(h)}, conditioned on the coupling {@code h} (Rende et al.,
 * "Foundation neural-network quantum states", 2025,
 * <a href="https://arxiv.org/abs/2502.09488">arXiv:2502.09488</a>, there with a ViT on large
 * lattices; here an RBM with FiLM-style conditioning at toy scale). With {@code u = h - h0}:
 * <pre>
 *   log psi = sum_i (a_i + c_i u) s_i + sum_j log cosh((1 + g_j u)(b_j + sum_i W_ji s_i) + e_j u)
 * </pre>
 * It is trained by <em>ensemble</em> stochastic reconfiguration: each step draws several
 * couplings, computes each coupling's force and geometric tensor (exact sums here) and solves
 * with their averages. Because {@code psi} depends on {@code h} explicitly, the fidelity
 * susceptibility {@code chi_F(h) = Var_(|psi|^2)(d log psi / dh)} (the curvature of the overlap
 * of neighbouring ground states, which peaks at a phase transition without any order
 * parameter) comes from the network's derivative in {@code h}.
 */
public final class ConditionalQuantumState {

    private final int n;
    private final int m;
    private final double h0;
    private final double[] theta;

    /**
     * An untrained state.
     *
     * @param n spins
     * @param alpha hidden units per spin
     * @param centre the coupling {@code h0} the conditioning is centred on
     * @param seed the seed
     */
    public ConditionalQuantumState(int n, int alpha, double centre, long seed) {
        this.n = n;
        this.m = alpha * n;
        this.h0 = centre;
        this.theta = new double[2 * n + m + m * n + 2 * m];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < theta.length; i++) {
            theta[i] = 0.01 * r.nextGaussian();
        }
    }

    /** @return the number of parameters */
    public int parameterCount() {
        return theta.length;
    }

    // layout: a[n], c[n], b[m], W[m n], g[m], e[m]
    private int aOff() {
        return 0;
    }

    private int cOff() {
        return n;
    }

    private int bOff() {
        return 2 * n;
    }

    private int wOff() {
        return 2 * n + m;
    }

    private int gOff() {
        return 2 * n + m + m * n;
    }

    private int eOff() {
        return 2 * n + 2 * m + m * n;
    }

    private static double spin(int basis, int i) {
        return ((basis >> i) & 1) == 0 ? 1 : -1;
    }

    private double pre(int j, int basis) {
        double t = theta[bOff() + j];
        for (int i = 0; i < n; i++) {
            t += theta[wOff() + j * n + i] * spin(basis, i);
        }
        return t;
    }

    /**
     * {@code log psi(sigma; h)}.
     *
     * @param basis the configuration
     * @param h the coupling
     * @return the log-amplitude
     */
    public double logPsi(int basis, double h) {
        double u = h - h0;
        double v = 0;
        for (int i = 0; i < n; i++) {
            v += (theta[aOff() + i] + theta[cOff() + i] * u) * spin(basis, i);
        }
        for (int j = 0; j < m; j++) {
            double t = (1 + theta[gOff() + j] * u) * pre(j, basis) + theta[eOff() + j] * u;
            double a = Math.abs(t);
            v += a + Math.log1p(Math.exp(-2 * a));
        }
        return v;
    }

    /**
     * {@code d log psi / d h}.
     *
     * @param basis the configuration
     * @param h the coupling
     * @return the derivative
     */
    public double dLogPsiDh(int basis, double h) {
        double u = h - h0;
        double v = 0;
        for (int i = 0; i < n; i++) {
            v += theta[cOff() + i] * spin(basis, i);
        }
        for (int j = 0; j < m; j++) {
            double p = pre(j, basis);
            double t = (1 + theta[gOff() + j] * u) * p + theta[eOff() + j] * u;
            v += Math.tanh(t) * (theta[gOff() + j] * p + theta[eOff() + j]);
        }
        return v;
    }

    /**
     * The state at one coupling, as a {@link WaveFunction} sharing this network's parameters.
     *
     * @param h the coupling
     * @return the view
     */
    public WaveFunction at(double h) {
        double u = h - h0;
        return new WaveFunction() {
            @Override
            public int spins() {
                return n;
            }

            @Override
            public int parameterCount() {
                return theta.length;
            }

            @Override
            public double[] parameters() {
                return theta;
            }

            @Override
            public double logPsi(int basis) {
                return ConditionalQuantumState.this.logPsi(basis, h);
            }

            @Override
            public double sign(int basis) {
                return 1;
            }

            @Override
            public void logDerivatives(int basis, double[] out) {
                for (int i = 0; i < n; i++) {
                    out[aOff() + i] = spin(basis, i);
                    out[cOff() + i] = u * spin(basis, i);
                }
                for (int j = 0; j < m; j++) {
                    double p = pre(j, basis);
                    double gj = 1 + theta[gOff() + j] * u;
                    double th = Math.tanh(gj * p + theta[eOff() + j] * u);
                    out[bOff() + j] = th * gj;
                    for (int i = 0; i < n; i++) {
                        out[wOff() + j * n + i] = th * gj * spin(basis, i);
                    }
                    out[gOff() + j] = th * u * p;
                    out[eOff() + j] = th * u;
                }
            }
        };
    }

    /**
     * Trains by ensemble stochastic reconfiguration with exact sums.
     *
     * @param family the Hamiltonian at each coupling
     * @param low the smallest training coupling
     * @param high the largest training coupling
     * @param steps SR steps
     * @param perStep couplings drawn per step
     * @param learningRate the SR step
     * @param shift the diagonal shift
     * @param seed the seed
     * @return the number of (coupling, step) energy evaluations used
     */
    public long train(DoubleFunction<PauliHamiltonian> family, double low, double high, int steps, int perStep,
            double learningRate, double shift, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int p = theta.length;
        long cost = 0;
        for (int s = 0; s < steps; s++) {
            double[] f = new double[p];
            double[][] sm = new double[p][p];
            for (int k = 0; k < perStep; k++) {
                // stratified draw over the training interval
                double h = low + (high - low) * (k + r.nextDouble()) / perStep;
                VariationalMonteCarlo.Estimate est = VariationalMonteCarlo.estimate(family.apply(h), at(h), null, true);
                cost++;
                for (int a = 0; a < p; a++) {
                    f[a] += est.force()[a] / perStep;
                    for (int b = 0; b < p; b++) {
                        sm[a][b] += est.s()[a][b] / perStep;
                    }
                }
            }
            double[] d = VariationalMonteCarlo.solveShifted(sm, f, shift);
            for (int a = 0; a < p; a++) {
                theta[a] -= learningRate * d[a];
            }
        }
        return cost;
    }

    /**
     * The fidelity susceptibility {@code Var(d log psi / dh)} under {@code |psi(h)|^2}, by exact
     * sum.
     *
     * @param h the coupling
     * @return {@code chi_F(h)}
     */
    public double fidelitySusceptibility(double h) {
        int dim = 1 << n;
        double[] lp = new double[dim];
        double max = Double.NEGATIVE_INFINITY;
        for (int b = 0; b < dim; b++) {
            lp[b] = 2 * logPsi(b, h);
            max = Math.max(max, lp[b]);
        }
        double z = 0;
        double d1 = 0;
        double d2 = 0;
        for (int b = 0; b < dim; b++) {
            double w = Math.exp(lp[b] - max);
            double d = dLogPsiDh(b, h);
            z += w;
            d1 += w * d;
            d2 += w * d * d;
        }
        d1 /= z;
        return d2 / z - d1 * d1;
    }

    /**
     * The exact fidelity susceptibility from neighbouring Lanczos ground states,
     * {@code 2 (1 - |<psi(h - d)|psi(h + d)>|) / (2d)^2}.
     *
     * @param family the Hamiltonian at each coupling
     * @param h the coupling
     * @param delta the half step {@code d}
     * @return {@code chi_F(h)}
     */
    public static double exactFidelitySusceptibility(DoubleFunction<PauliHamiltonian> family, double h, double delta) {
        PauliHamiltonian a = family.apply(h - delta);
        PauliHamiltonian b = family.apply(h + delta);
        StateVector sa = a.groundState(100, 1).state();
        StateVector sb = b.groundState(100, 1).state();
        double ov = Math.sqrt(sa.fidelity(sb));
        return 2 * (1 - ov) / (4 * delta * delta);
    }
}
