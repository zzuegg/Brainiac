package io.github.zzuegg.brainiac.experimental.design;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.DesignProblem;
import io.github.zzuegg.brainiac.experimental.train.ModelBasedSearch;
import io.github.zzuegg.brainiac.tensor.Optimize;
import io.github.zzuegg.brainiac.train.HyperparameterSearch;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Static design optimisation over a box: {@link #random} search, a space-filling {@link #grid},
 * {@link #cmaEs} (core {@link Optimize#cmaEs}) on a noisy EIG estimate, Gaussian-process
 * {@link #bayesOpt} with expected improvement (wave 12's {@link ModelBasedSearch}), and
 * {@link #pceAscent}: stochastic gradient ascent on the PCE bound with central finite
 * differences under common random numbers (the same prior draws and noise on both sides of
 * each difference), with Adam-style normalised steps (Foster et al. 2020).
 */
public final class DesignOptimiser {

    private DesignOptimiser() {
    }

    /**
     * An optimised design.
     *
     * @param design the design
     * @param value the objective at the design, as estimated during the search
     * @param evaluations objective evaluations used
     */
    public record Result(double[] design, double value, int evaluations) {
    }

    private static double[] clamp(double[] x, double[] lo, double[] hi) {
        double[] c = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            c[i] = Math.min(hi[i], Math.max(lo[i], x[i]));
        }
        return c;
    }

    /**
     * The best of uniformly random designs.
     *
     * @param eig the objective (higher is better)
     * @param lo lower corner
     * @param hi upper corner
     * @param count designs to try
     * @param seed the seed
     * @return the best design
     */
    public static Result random(ToDoubleFunction<double[]> eig, double[] lo, double[] hi, int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] best = null;
        double bv = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            double[] x = new double[lo.length];
            for (int d = 0; d < x.length; d++) {
                x[d] = lo[d] + r.nextDouble() * (hi[d] - lo[d]);
            }
            double v = eig.applyAsDouble(x);
            if (v > bv) {
                bv = v;
                best = x;
            }
        }
        return new Result(best, bv, count);
    }

    /**
     * The best design on a regular grid.
     *
     * @param eig the objective
     * @param lo lower corner
     * @param hi upper corner
     * @param perDim grid points per dimension
     * @return the best grid design
     */
    public static Result grid(ToDoubleFunction<double[]> eig, double[] lo, double[] hi, int perDim) {
        int d = lo.length;
        int total = (int) Math.pow(perDim, d);
        double[] best = null;
        double bv = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < total; i++) {
            double[] x = new double[d];
            int rest = i;
            for (int c = 0; c < d; c++) {
                int k = rest % perDim;
                rest /= perDim;
                x[c] = perDim == 1 ? 0.5 * (lo[c] + hi[c]) : lo[c] + k * (hi[c] - lo[c]) / (perDim - 1);
            }
            double v = eig.applyAsDouble(x);
            if (v > bv) {
                bv = v;
                best = x;
            }
        }
        return new Result(best, bv, total);
    }

    /**
     * CMA-ES on the objective, clamped to the box.
     *
     * @param eig the objective
     * @param lo lower corner
     * @param hi upper corner
     * @param generations the most generations
     * @param seed the seed
     * @return the best design
     */
    public static Result cmaEs(ToDoubleFunction<double[]> eig, double[] lo, double[] hi, int generations, long seed) {
        int[] count = new int[1];
        double[] x0 = new double[lo.length];
        double width = 0;
        for (int i = 0; i < lo.length; i++) {
            x0[i] = 0.5 * (lo[i] + hi[i]);
            width = Math.max(width, hi[i] - lo[i]);
        }
        Optimize.Minimum m = Optimize.cmaEs(x -> {
            count[0]++;
            return -eig.applyAsDouble(clamp(x, lo, hi));
        }, x0, width / 3, generations, 1e-6, seed);
        double[] x = clamp(m.x(), lo, hi);
        return new Result(x, -m.value(), count[0]);
    }

    /**
     * Gaussian-process Bayesian optimisation with expected improvement.
     *
     * @param eig the objective
     * @param lo lower corner
     * @param hi upper corner
     * @param trials total evaluations
     * @param seed the seed
     * @return the best design
     */
    public static Result bayesOpt(ToDoubleFunction<double[]> eig, double[] lo, double[] hi, int trials, long seed) {
        Map<String, HyperparameterSearch.Range> space = new LinkedHashMap<>();
        for (int i = 0; i < lo.length; i++) {
            space.put("x" + i, new HyperparameterSearch.Uniform(lo[i], hi[i]));
        }
        ModelBasedSearch search = new ModelBasedSearch(space, seed);
        HyperparameterSearch.Result res = search.bayesOpt((config, budget) -> {
            double[] x = new double[lo.length];
            for (int i = 0; i < x.length; i++) {
                x[i] = ((Number) config.get("x" + i)).doubleValue();
            }
            return -eig.applyAsDouble(x);
        }, trials, 1, Math.max(2, trials / 4), ModelBasedSearch.Acquisition.EXPECTED_IMPROVEMENT);
        double[] x = new double[lo.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((Number) res.best().config().get("x" + i)).doubleValue();
        }
        return new Result(x, -res.best().score(), trials);
    }

    /**
     * Gradient ascent on the PCE bound.
     *
     * @param problem the problem
     * @param start the starting design
     * @param steps ascent steps
     * @param learningRate the step size in design units
     * @param outer outer samples per gradient estimate
     * @param contrasts contrastive samples per outer sample
     * @param seed the seed
     * @return the final design with its PCE estimate during the last step
     */
    public static Result pceAscent(DesignProblem problem, double[] start, int steps, double learningRate, int outer,
            int contrasts, long seed) {
        double[] lo = problem.designLow();
        double[] hi = problem.designHigh();
        int d = start.length;
        double[] x = start.clone();
        double[] m1 = new double[d];
        double[] m2 = new double[d];
        SplittableRandom seeds = new SplittableRandom(seed);
        int evals = 0;
        double value = Double.NaN;
        for (int s = 1; s <= steps; s++) {
            long crn = seeds.nextLong();
            double[] g = new double[d];
            for (int c = 0; c < d; c++) {
                double h = 1e-2 * (hi[c] - lo[c]);
                double[] a = x.clone();
                double[] b = x.clone();
                a[c] = Math.min(hi[c], x[c] + h);
                b[c] = Math.max(lo[c], x[c] - h);
                double fa = InformationGain.pce(problem, a, outer, contrasts, new SplittableRandom(crn));
                double fb = InformationGain.pce(problem, b, outer, contrasts, new SplittableRandom(crn));
                g[c] = (fa - fb) / Math.max(a[c] - b[c], 1e-12);
                value = 0.5 * (fa + fb);
                evals += 2;
            }
            for (int c = 0; c < d; c++) {
                m1[c] = 0.9 * m1[c] + 0.1 * g[c];
                m2[c] = 0.999 * m2[c] + 0.001 * g[c] * g[c];
                double mh = m1[c] / (1 - Math.pow(0.9, s));
                double vh = m2[c] / (1 - Math.pow(0.999, s));
                x[c] += learningRate * (hi[c] - lo[c]) * mh / (Math.sqrt(vh) + 1e-8);
            }
            x = clamp(x, lo, hi);
        }
        return new Result(x, value, evals);
    }
}
