package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Particle swarm optimisation: every particle keeps a velocity, its own best position and
 * sees the best position of its neighbourhood; each step
 * {@code v <- w v + c1 r1 (personal best - x) + c2 r2 (neighbourhood best - x)} with fresh
 * uniform {@code r1, r2} per coordinate, then {@code x <- x + v}. Defaults are Clerc's
 * constriction coefficients ({@code w = 0.7298}, {@code c1 = c2 = 1.49618}), velocities are
 * clamped to a fifth of the box, and the neighbourhood is either the whole swarm (global best,
 * fast) or a ring of the two adjacent particles (local best, slower but less prone to
 * premature convergence on multimodal costs). Costs are <b>minimised</b>.
 *
 * <p>Kennedy and Eberhart 1995, "Particle Swarm Optimization", Proc. ICNN; Clerc and Kennedy
 * 2002, "The Particle Swarm: Explosion, Stability, and Convergence", IEEE TEC 6(1).
 */
public final class ParticleSwarm {

    private final double[] low;
    private final double[] high;
    private final int n;
    private int size = 40;
    private double inertia = 0.7298;
    private double pull = 1.49618;
    private boolean ring;
    private long seed = 1;
    private double[] bestX;
    private double bestCost = Double.POSITIVE_INFINITY;
    private int evaluations;

    private ParticleSwarm(double[] low, double[] high) {
        this.low = low.clone();
        this.high = high.clone();
        this.n = low.length;
    }

    /**
     * @param low the lower bounds of the search box
     * @param high the upper bounds
     * @return the search, to configure
     */
    public static ParticleSwarm of(double[] low, double[] high) {
        Objects.requireNonNull(low, "low");
        if (low.length == 0 || low.length != high.length) {
            throw new IllegalArgumentException("need matching, non-empty bounds");
        }
        return new ParticleSwarm(low, high);
    }

    /**
     * @param particles the swarm size (default 40)
     * @return this
     */
    public ParticleSwarm swarm(int particles) {
        if (particles < 3) {
            throw new IllegalArgumentException("need at least 3 particles");
        }
        this.size = particles;
        return this;
    }

    /**
     * @param w the inertia weight
     * @param c the personal and social pull
     * @return this
     */
    public ParticleSwarm coefficients(double w, double c) {
        this.inertia = w;
        this.pull = c;
        return this;
    }

    /**
     * @param on use a ring neighbourhood (local best) instead of the global best
     * @return this
     */
    public ParticleSwarm ring(boolean on) {
        this.ring = on;
        return this;
    }

    /**
     * @param s the seed
     * @return this
     */
    public ParticleSwarm seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the best point found */
    public double[] best() {
        return bestX == null ? null : bestX.clone();
    }

    /** @return its cost */
    public double bestCost() {
        return bestCost;
    }

    /** @return cost evaluations used */
    public int evaluations() {
        return evaluations;
    }

    /**
     * Minimises within the box.
     *
     * @param cost the cost to minimise
     * @param maxEvaluations the evaluation budget
     * @return the best point found
     */
    public double[] minimize(ToDoubleFunction<double[]> cost, int maxEvaluations) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[size][n];
        double[][] v = new double[size][n];
        double[][] pBest = new double[size][];
        double[] pCost = new double[size];
        double[] vmax = new double[n];
        for (int j = 0; j < n; j++) {
            vmax[j] = 0.2 * (high[j] - low[j]);
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < n; j++) {
                x[i][j] = r.nextDouble(low[j], high[j]);
                v[i][j] = r.nextDouble(-vmax[j], vmax[j]) * 0.5;
            }
            pBest[i] = x[i].clone();
            pCost[i] = Double.POSITIVE_INFINITY;
        }
        while (evaluations < maxEvaluations) {
            for (int i = 0; i < size && evaluations < maxEvaluations; i++) {
                double c = cost.applyAsDouble(x[i]);
                evaluations++;
                if (c < pCost[i]) {
                    pCost[i] = c;
                    pBest[i] = x[i].clone();
                }
                if (c < bestCost) {
                    bestCost = c;
                    bestX = x[i].clone();
                }
            }
            for (int i = 0; i < size; i++) {
                double[] social = bestX;
                if (ring) {
                    int left = (i + size - 1) % size;
                    int right = (i + 1) % size;
                    int k = pCost[left] < pCost[i] ? left : i;
                    k = pCost[right] < pCost[k] ? right : k;
                    social = pBest[k];
                }
                for (int j = 0; j < n; j++) {
                    double vj = inertia * v[i][j] + pull * r.nextDouble() * (pBest[i][j] - x[i][j])
                            + pull * r.nextDouble() * (social[j] - x[i][j]);
                    vj = Math.max(-vmax[j], Math.min(vmax[j], vj));
                    double xj = x[i][j] + vj;
                    if (xj < low[j] || xj > high[j]) {
                        xj = Math.max(low[j], Math.min(high[j], xj));
                        vj = 0;
                    }
                    v[i][j] = vj;
                    x[i][j] = xj;
                }
            }
        }
        return best();
    }
}
