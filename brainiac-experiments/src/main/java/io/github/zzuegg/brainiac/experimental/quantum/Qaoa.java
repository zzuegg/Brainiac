package io.github.zzuegg.brainiac.experimental.quantum;

import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The quantum approximate optimisation algorithm for MaxCut (Farhi, Goldstone and Gutmann,
 * <a href="https://arxiv.org/abs/1411.4028">arXiv:1411.4028</a>): from {@code |+>^n}, apply
 * {@code p} layers of the cost unitary {@code exp(-i gamma_l C)} (diagonal, built from the cut
 * value of every basis state) and the mixer {@code exp(-i beta_l sum X)}, and maximise
 * {@code <C>} over the {@code 2p} angles with core {@link Optimize#nelderMead}.
 *
 * <p>Initialisations (Zhou et al., "Quantum approximate optimization algorithm: performance,
 * mechanism and implementation on near-term devices", PRX 2020):
 * <ul>
 *   <li>{@link Init#RANDOM}: uniform random angles at every depth.</li>
 *   <li>{@link Init#INTERP}: depth {@code p + 1} starts from the linear interpolation of the
 *       optimal depth-{@code p} schedule, {@code x'_i = (i-1)/p x_(i-1) + (p-i+1)/p x_i}.</li>
 *   <li>{@link Init#FOURIER}: optimise the discrete sine/cosine amplitudes {@code (u, v)} of the
 *       schedule, padding with a zero amplitude when the depth grows.</li>
 * </ul>
 * Transfer of angles between graphs is {@link #expectation} at another graph's angles.
 */
public final class Qaoa {

    /** How each depth's angles start. */
    public enum Init {
        /** Uniform random angles. */
        RANDOM,
        /** Interpolate the previous depth's optimum. */
        INTERP,
        /** Fourier amplitudes, warm-started across depths. */
        FOURIER
    }

    /**
     * The optimum found at one depth.
     *
     * @param p the depth
     * @param expectation {@code <C>}
     * @param ratio {@code <C> / C_max}
     * @param optimalProbability the probability of measuring a maximum cut
     * @param gammas the cost angles
     * @param betas the mixer angles
     * @param evaluations circuit evaluations used at this depth
     */
    public record Result(int p, double expectation, double ratio, double optimalProbability, double[] gammas,
            double[] betas, long evaluations) {
    }

    private final int n;
    private final double[] cost;
    private final double best;
    private long evaluations;

    /**
     * @param w symmetric MaxCut weights
     */
    public Qaoa(double[][] w) {
        this.n = w.length;
        this.cost = SpinModels.maxCutDiagonal(w);
        double m = 0;
        for (double c : cost) {
            m = Math.max(m, c);
        }
        this.best = m;
    }

    /** @return the maximum cut, by enumeration */
    public double maxCut() {
        return best;
    }

    /**
     * The QAOA state.
     *
     * @param gammas cost angles
     * @param betas mixer angles
     * @return the state
     */
    public StateVector state(double[] gammas, double[] betas) {
        StateVector s = new StateVector(n);
        for (int q = 0; q < n; q++) {
            s.h(q);
        }
        for (int l = 0; l < gammas.length; l++) {
            s.diagonalPhase(cost, gammas[l]);
            for (int q = 0; q < n; q++) {
                s.pauliRotation(1 << q, 0, betas[l]);
            }
        }
        return s;
    }

    /**
     * {@code <C>} at the given angles.
     *
     * @param gammas cost angles
     * @param betas mixer angles
     * @return the expected cut
     */
    public double expectation(double[] gammas, double[] betas) {
        evaluations++;
        double[] pr = state(gammas, betas).probabilities();
        double e = 0;
        for (int b = 0; b < pr.length; b++) {
            e += pr[b] * cost[b];
        }
        return e;
    }

    private double optimalProbability(double[] gammas, double[] betas) {
        double[] pr = state(gammas, betas).probabilities();
        double s = 0;
        for (int b = 0; b < pr.length; b++) {
            if (cost[b] >= best - 1e-9) {
                s += pr[b];
            }
        }
        return s;
    }

    /**
     * Samples cuts from the QAOA state and returns the best.
     *
     * @param gammas cost angles
     * @param betas mixer angles
     * @param shots samples
     * @param random the random source
     * @return the largest sampled cut
     */
    public double bestSampledCut(double[] gammas, double[] betas, int shots, SplittableRandom random) {
        int[] s = StateVector.sample(state(gammas, betas).probabilities(), shots, random);
        double m = 0;
        for (int b : s) {
            m = Math.max(m, cost[b]);
        }
        return m;
    }

    /**
     * Optimises the {@code 2p} angles by Nelder–Mead from a start.
     *
     * @param gammas starting cost angles
     * @param betas starting mixer angles
     * @param maxIterations the Nelder–Mead budget
     * @return the optimum
     */
    public Result optimize(double[] gammas, double[] betas, int maxIterations) {
        int p = gammas.length;
        long before = evaluations;
        double[] x0 = new double[2 * p];
        System.arraycopy(gammas, 0, x0, 0, p);
        System.arraycopy(betas, 0, x0, p, p);
        Optimize.Minimum m = Optimize.nelderMead(x -> -expectation(java.util.Arrays.copyOf(x, p),
                java.util.Arrays.copyOfRange(x, p, 2 * p)), x0, 0.1, maxIterations, 1e-8);
        double[] g = java.util.Arrays.copyOf(m.x(), p);
        double[] b = java.util.Arrays.copyOfRange(m.x(), p, 2 * p);
        double e = -m.value();
        return new Result(p, e, e / best, optimalProbability(g, b), g, b, evaluations - before);
    }

    private Result optimizeFourier(double[] u, double[] v, int maxIterations) {
        int p = u.length;
        long before = evaluations;
        double[] x0 = new double[2 * p];
        System.arraycopy(u, 0, x0, 0, p);
        System.arraycopy(v, 0, x0, p, p);
        Optimize.Minimum m = Optimize.nelderMead(x -> {
            double[][] gb = fourierAngles(java.util.Arrays.copyOf(x, p), java.util.Arrays.copyOfRange(x, p, 2 * p));
            return -expectation(gb[0], gb[1]);
        }, x0, 0.1, maxIterations, 1e-8);
        double[][] gb = fourierAngles(java.util.Arrays.copyOf(m.x(), p), java.util.Arrays.copyOfRange(m.x(), p, 2 * p));
        double e = -m.value();
        Result r = new Result(p, e, e / best, optimalProbability(gb[0], gb[1]), gb[0], gb[1], evaluations - before);
        lastFourier = m.x();
        return r;
    }

    private double[] lastFourier;

    /**
     * The angles of a Fourier schedule: {@code gamma_i = sum_k u_k sin((k - 1/2)(i - 1/2) pi / p)}
     * and {@code beta_i = sum_k v_k cos((k - 1/2)(i - 1/2) pi / p)}.
     *
     * @param u sine amplitudes (length p)
     * @param v cosine amplitudes (length p)
     * @return {gammas, betas}
     */
    public static double[][] fourierAngles(double[] u, double[] v) {
        int p = u.length;
        double[] g = new double[p];
        double[] b = new double[p];
        for (int i = 1; i <= p; i++) {
            for (int k = 1; k <= p; k++) {
                double a = (k - 0.5) * (i - 0.5) * Math.PI / p;
                g[i - 1] += u[k - 1] * Math.sin(a);
                b[i - 1] += v[k - 1] * Math.cos(a);
            }
        }
        return new double[][] {g, b};
    }

    /**
     * Zhou et al.'s INTERP rule from depth {@code p} to {@code p + 1}.
     *
     * @param x the depth-p schedule
     * @return the depth-(p+1) schedule
     */
    public static double[] interpolate(double[] x) {
        int p = x.length;
        double[] y = new double[p + 1];
        for (int i = 1; i <= p + 1; i++) {
            double prev = i - 2 >= 0 ? x[i - 2] : 0;
            double cur = i - 1 < p ? x[i - 1] : 0;
            y[i - 1] = (double) (i - 1) / p * prev + (double) (p - i + 1) / p * cur;
        }
        return y;
    }

    /**
     * Optimises every depth {@code 1..maxP} with an initialisation strategy.
     *
     * @param init the strategy
     * @param maxP the largest depth
     * @param maxIterations Nelder–Mead budget per depth
     * @param seed the seed (for random starts)
     * @return one result per depth
     */
    public List<Result> sweep(Init init, int maxP, int maxIterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Result> out = new ArrayList<>();
        double[] g = {0.4};
        double[] b = {0.4};
        double[] u = {0.4};
        double[] v = {0.4};
        for (int p = 1; p <= maxP; p++) {
            Result res;
            switch (init) {
                case RANDOM -> {
                    g = new double[p];
                    b = new double[p];
                    for (int i = 0; i < p; i++) {
                        g[i] = r.nextDouble() * Math.PI;
                        b[i] = r.nextDouble() * Math.PI / 2;
                    }
                    res = optimize(g, b, maxIterations);
                }
                case INTERP -> {
                    if (p > 1) {
                        Result prev = out.get(p - 2);
                        g = interpolate(prev.gammas());
                        b = interpolate(prev.betas());
                    }
                    res = optimize(g, b, maxIterations);
                }
                case FOURIER -> {
                    if (p > 1) {
                        u = java.util.Arrays.copyOf(java.util.Arrays.copyOf(lastFourier, p - 1), p);
                        v = java.util.Arrays.copyOf(java.util.Arrays.copyOfRange(lastFourier, p - 1, 2 * p - 2), p);
                    }
                    res = optimizeFourier(u, v, maxIterations);
                }
                default -> throw new IllegalStateException();
            }
            out.add(res);
        }
        return out;
    }
}
