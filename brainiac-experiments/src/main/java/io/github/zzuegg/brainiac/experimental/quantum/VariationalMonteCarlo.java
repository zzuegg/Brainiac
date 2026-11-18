package io.github.zzuegg.brainiac.experimental.quantum;

import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Variational Monte Carlo for a {@link WaveFunction}: estimate the energy as the mean local
 * energy under {@code |psi|^2} and follow its gradient
 * {@code g_k = 2 (<E_loc O_k> - <E_loc><O_k>)}.
 *
 * <p>{@link Method#STOCHASTIC_RECONFIGURATION} (Sorella, PRL 1998) preconditions the gradient by
 * the quantum geometric tensor {@code S_kl = <O_k O_l> - <O_k><O_l>}: it solves
 * {@code (S + lambda I) delta = f} by core {@link Matrices#cholesky} and steps
 * {@code theta -= eta delta}, which is imaginary-time evolution projected onto the ansatz (the
 * natural gradient of the state) and far more stable than plain descent. {@link Method#SGD} and
 * {@link Method#ADAM} use the core optimizers on the same gradient.
 *
 * <p>With {@code samples = 0} the expectations are exact sums over all {@code 2^n}
 * configurations, which separates the ansatz error from the sampling error; otherwise they are
 * Metropolis estimates from {@link NeuralQuantumState#sample}.
 */
public final class VariationalMonteCarlo {

    /** The update rule. */
    public enum Method {
        /** Core SGD on the energy gradient. */
        SGD,
        /** Core Adam on the energy gradient. */
        ADAM,
        /** Stochastic reconfiguration (natural gradient). */
        STOCHASTIC_RECONFIGURATION
    }

    /**
     * Expectations under {@code |psi|^2}.
     *
     * @param energy the mean local energy
     * @param variance the variance of the local energy (zero for an eigenstate)
     * @param force {@code f_k = <E O_k> - <E><O_k>}
     * @param s the geometric tensor, or null when not asked for
     */
    public record Estimate(double energy, double variance, double[] force, double[][] s) {
    }

    /**
     * A run.
     *
     * @param energy the exact energy at the end (exact sum)
     * @param variance its local-energy variance
     * @param trace the estimated energy at every step
     */
    public record Result(double energy, double variance, double[] trace) {
    }

    private VariationalMonteCarlo() {
    }

    /**
     * Estimates energy, force and optionally {@code S}.
     *
     * @param h the Hamiltonian
     * @param psi the wave function
     * @param samples sampled configurations, or null for the exact sum over all
     * @param withS whether to build {@code S}
     * @return the estimate
     */
    public static Estimate estimate(PauliHamiltonian h, WaveFunction psi, int[] samples, boolean withS) {
        int p = psi.parameterCount();
        int count;
        int[] configs;
        double[] weight;
        if (samples == null) {
            count = 1 << psi.spins();
            configs = new int[count];
            weight = new double[count];
            double max = Double.NEGATIVE_INFINITY;
            for (int b = 0; b < count; b++) {
                configs[b] = b;
                weight[b] = 2 * psi.logPsi(b);
                max = Math.max(max, weight[b]);
            }
            double z = 0;
            for (int b = 0; b < count; b++) {
                weight[b] = Math.exp(weight[b] - max);
                z += weight[b];
            }
            for (int b = 0; b < count; b++) {
                weight[b] /= z;
            }
        } else {
            count = samples.length;
            configs = samples;
            weight = new double[count];
            java.util.Arrays.fill(weight, 1.0 / count);
        }
        double[] o = new double[p];
        double[] meanO = new double[p];
        double[] meanEO = new double[p];
        double[][] oo = withS ? new double[p][p] : null;
        double e = 0;
        double e2 = 0;
        for (int c = 0; c < count; c++) {
            double w = weight[c];
            if (w == 0) {
                continue;
            }
            double el = psi.localEnergy(h, configs[c]);
            psi.logDerivatives(configs[c], o);
            e += w * el;
            e2 += w * el * el;
            for (int k = 0; k < p; k++) {
                meanO[k] += w * o[k];
                meanEO[k] += w * el * o[k];
                if (withS) {
                    double wo = w * o[k];
                    double[] row = oo[k];
                    for (int l = k; l < p; l++) {
                        row[l] += wo * o[l];
                    }
                }
            }
        }
        double[] f = new double[p];
        for (int k = 0; k < p; k++) {
            f[k] = meanEO[k] - e * meanO[k];
        }
        if (withS) {
            for (int k = 0; k < p; k++) {
                for (int l = k; l < p; l++) {
                    oo[k][l] -= meanO[k] * meanO[l];
                    oo[l][k] = oo[k][l];
                }
            }
        }
        return new Estimate(e, Math.max(0, e2 - e * e), f, oo);
    }

    /**
     * Solves {@code (S + shift I) x = f} by Cholesky.
     *
     * @param s the geometric tensor (not modified)
     * @param f the force
     * @param shift the diagonal shift
     * @return the solution
     */
    public static double[] solveShifted(double[][] s, double[] f, double shift) {
        int p = f.length;
        double[][] a = new double[p][];
        for (int k = 0; k < p; k++) {
            a[k] = s[k].clone();
            a[k][k] += shift;
        }
        double[][] l = Matrices.cholesky(a);
        double[] y = new double[p];
        for (int i = 0; i < p; i++) {
            double v = f[i];
            for (int j = 0; j < i; j++) {
                v -= l[i][j] * y[j];
            }
            y[i] = v / l[i][i];
        }
        double[] x = new double[p];
        for (int i = p - 1; i >= 0; i--) {
            double v = y[i];
            for (int j = i + 1; j < p; j++) {
                v -= l[j][i] * x[j];
            }
            x[i] = v / l[i][i];
        }
        return x;
    }

    /**
     * Optimises a neural quantum state.
     *
     * @param h the Hamiltonian
     * @param psi the state (its parameters are updated)
     * @param method the update rule
     * @param steps the steps
     * @param learningRate the step size
     * @param shift SR's diagonal shift
     * @param samples Metropolis samples per step, or 0 for exact sums
     * @param exchange whether the sampler conserves magnetisation
     * @param seed the sampling seed
     * @return the run
     */
    public static Result run(PauliHamiltonian h, NeuralQuantumState psi, Method method, int steps,
            double learningRate, double shift, int samples, boolean exchange, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] trace = new double[steps];
        double[] theta = psi.parameters();
        if (method == Method.STOCHASTIC_RECONFIGURATION) {
            for (int s = 0; s < steps; s++) {
                int[] cfg = samples == 0 ? null : psi.sample(samples, 1, s == 0 ? 50 : 5, exchange, r);
                Estimate est = estimate(h, psi, cfg, true);
                trace[s] = est.energy();
                double[] d = solveShifted(est.s(), est.force(), shift);
                for (int k = 0; k < theta.length; k++) {
                    theta[k] -= learningRate * d[k];
                }
            }
        } else {
            int[] step = {0};
            ParameterObjective f = (x, g) -> {
                System.arraycopy(x, 0, theta, 0, x.length);
                int[] cfg = samples == 0 ? null : psi.sample(samples, 1, 20, exchange, r);
                Estimate est = estimate(h, psi, cfg, false);
                for (int k = 0; k < g.length; k++) {
                    g[k] = 2 * est.force()[k];
                }
                if (step[0] < trace.length) {
                    trace[step[0]++] = est.energy();
                }
                return est.energy();
            };
            Optimizer opt = method == Method.ADAM ? Optimizer.adam((float) learningRate)
                    : Optimizer.sgd((float) learningRate);
            double[] end = f.minimize(opt, theta.clone(), steps, null);
            System.arraycopy(end, 0, theta, 0, end.length);
        }
        Estimate fin = estimate(h, psi, null, false);
        return new Result(fin.energy(), fin.variance(), trace);
    }
}
