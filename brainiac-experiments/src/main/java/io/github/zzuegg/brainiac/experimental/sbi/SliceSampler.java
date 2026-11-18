package io.github.zzuegg.brainiac.experimental.sbi;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Coordinate-wise slice sampling with stepping out and shrinkage (Neal, "Slice sampling",
 * Annals of Statistics 2003, <a href="https://arxiv.org/abs/physics/0009028">physics/0009028</a>):
 * a gradient-free MCMC kernel with no step size to tune, the sampler that neural likelihood
 * and ratio estimators use on their learned (non-differentiable to the caller) log densities.
 * Several chains from different starts cover separated modes; {@link #sample} picks the starts
 * as the best of a set of candidate points.
 */
public final class SliceSampler {

    private SliceSampler() {
    }

    /**
     * One chain.
     *
     * @param logDensity the unnormalised log density, negative infinity outside the support
     * @param start a point of finite density
     * @param warmup sweeps discarded
     * @param samples sweeps kept
     * @param width the initial bracket width per coordinate
     * @param random the random source
     * @return {@code [samples][d]}
     */
    public static double[][] chain(ToDoubleFunction<double[]> logDensity, double[] start, int warmup, int samples,
            double width, SplittableRandom random) {
        double[] x = start.clone();
        double lx = logDensity.applyAsDouble(x);
        if (!Double.isFinite(lx)) {
            throw new IllegalArgumentException("the start has zero density");
        }
        int d = x.length;
        double[] w = new double[d];
        Arrays.fill(w, width);
        double[][] out = new double[samples][];
        for (int it = 0; it < warmup + samples; it++) {
            for (int j = 0; j < d; j++) {
                double level = lx + Math.log(random.nextDouble());
                double x0 = x[j];
                double lo = x0 - w[j] * random.nextDouble();
                double hi = lo + w[j];
                for (int s = 0; s < 20; s++) {
                    x[j] = lo;
                    if (logDensity.applyAsDouble(x) <= level) {
                        break;
                    }
                    lo -= w[j];
                }
                for (int s = 0; s < 20; s++) {
                    x[j] = hi;
                    if (logDensity.applyAsDouble(x) <= level) {
                        break;
                    }
                    hi += w[j];
                }
                while (true) {
                    double cand = lo + (hi - lo) * random.nextDouble();
                    x[j] = cand;
                    double lc = logDensity.applyAsDouble(x);
                    if (lc > level) {
                        lx = lc;
                        if (it < warmup) {
                            w[j] = 0.9 * w[j] + 0.1 * Math.max(1e-6, 2 * (hi - lo) / 3);
                        }
                        break;
                    }
                    if (cand < x0) {
                        lo = cand;
                    } else {
                        hi = cand;
                    }
                    if (hi - lo < 1e-12) {
                        x[j] = x0;
                        break;
                    }
                }
            }
            if (it >= warmup) {
                out[it - warmup] = x.clone();
            }
        }
        return out;
    }

    /**
     * Several chains started at the highest-density points among {@code candidates}, pooled.
     *
     * @param logDensity the unnormalised log density
     * @param candidates candidate start points (for example prior draws)
     * @param chains the number of chains
     * @param warmup warm-up sweeps per chain
     * @param n the total number of samples returned
     * @param width the initial bracket width
     * @param random the random source
     * @return {@code [n][d]}
     */
    public static double[][] sample(ToDoubleFunction<double[]> logDensity, double[][] candidates, int chains,
            int warmup, int n, double width, SplittableRandom random) {
        double[] score = new double[candidates.length];
        Integer[] order = new Integer[candidates.length];
        for (int i = 0; i < candidates.length; i++) {
            score[i] = logDensity.applyAsDouble(candidates[i]);
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        double[][] out = new double[n][];
        int per = (n + chains - 1) / chains;
        int k = 0;
        for (int c = 0; c < chains && k < n; c++) {
            double[] start = candidates[order[Math.min(c * Math.max(1, candidates.length / (4 * chains)),
                    candidates.length - 1)]];
            if (!Double.isFinite(logDensity.applyAsDouble(start))) {
                start = candidates[order[0]];
            }
            double[][] ch = chain(logDensity, start, warmup, per, width, random);
            for (int i = 0; i < per && k < n; i++) {
                out[k++] = ch[i];
            }
        }
        return out;
    }
}
