package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Differential evolution, DE/rand/1/bin: for every member (the target) of the population,
 * a mutant {@code a + F (b - c)} is formed from three other random members, mixed with the
 * target by binomial crossover (each coordinate from the mutant with probability {@code CR},
 * at least one always), and the trial replaces the target if it is no worse. The step sizes
 * come from the population's own spread, so the search scales itself to the landscape, and it
 * needs no covariance matrix. Coordinates leaving the box are reset between the target and
 * the bound. Costs are <b>minimised</b>, as in {@link CmaEs}.
 *
 * <p>Storn and Price 1997, "Differential Evolution: A Simple and Efficient Heuristic for
 * Global Optimization over Continuous Spaces", Journal of Global Optimization 11.
 */
public final class DifferentialEvolution {

    private final double[] low;
    private final double[] high;
    private final int n;
    private int size;
    private double f = 0.5;
    private double cr = 0.9;
    private long seed = 1;
    private double[] bestX;
    private double bestCost = Double.POSITIVE_INFINITY;
    private int evaluations;

    private DifferentialEvolution(double[] low, double[] high) {
        this.low = low.clone();
        this.high = high.clone();
        this.n = low.length;
        this.size = Math.max(20, 5 * n);
    }

    /**
     * @param low the lower bounds of the search box
     * @param high the upper bounds
     * @return the search, to configure
     */
    public static DifferentialEvolution of(double[] low, double[] high) {
        Objects.requireNonNull(low, "low");
        if (low.length == 0 || low.length != high.length) {
            throw new IllegalArgumentException("need matching, non-empty bounds");
        }
        return new DifferentialEvolution(low, high);
    }

    /**
     * @param members the population size (at least 4; default {@code max(20, 5 n)})
     * @return this
     */
    public DifferentialEvolution population(int members) {
        if (members < 4) {
            throw new IllegalArgumentException("DE needs at least 4 members");
        }
        this.size = members;
        return this;
    }

    /**
     * @param weight the differential weight {@code F} (default 0.5)
     * @param crossover the crossover rate {@code CR} (default 0.9)
     * @return this
     */
    public DifferentialEvolution rates(double weight, double crossover) {
        this.f = weight;
        this.cr = crossover;
        return this;
    }

    /**
     * @param s the seed
     * @return this
     */
    public DifferentialEvolution seed(long s) {
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

    private double eval(ToDoubleFunction<double[]> cost, double[] x) {
        double c = cost.applyAsDouble(x);
        evaluations++;
        if (c < bestCost) {
            bestCost = c;
            bestX = x.clone();
        }
        return c;
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
        double[][] pop = new double[size][n];
        double[] fit = new double[size];
        for (int i = 0; i < size && evaluations < maxEvaluations; i++) {
            for (int j = 0; j < n; j++) {
                pop[i][j] = r.nextDouble(low[j], high[j]);
            }
            fit[i] = eval(cost, pop[i]);
        }
        while (evaluations < maxEvaluations) {
            for (int i = 0; i < size && evaluations < maxEvaluations; i++) {
                int a;
                int b;
                int c;
                do {
                    a = r.nextInt(size);
                } while (a == i);
                do {
                    b = r.nextInt(size);
                } while (b == i || b == a);
                do {
                    c = r.nextInt(size);
                } while (c == i || c == a || c == b);
                double[] trial = pop[i].clone();
                int forced = r.nextInt(n);
                for (int j = 0; j < n; j++) {
                    if (j == forced || r.nextDouble() < cr) {
                        double v = pop[a][j] + f * (pop[b][j] - pop[c][j]);
                        if (v < low[j]) {
                            v = low[j] + r.nextDouble() * (pop[i][j] - low[j]);
                        } else if (v > high[j]) {
                            v = high[j] - r.nextDouble() * (high[j] - pop[i][j]);
                        }
                        trial[j] = v;
                    }
                }
                double ft = eval(cost, trial);
                if (ft <= fit[i]) {
                    pop[i] = trial;
                    fit[i] = ft;
                }
            }
        }
        return best();
    }
}
