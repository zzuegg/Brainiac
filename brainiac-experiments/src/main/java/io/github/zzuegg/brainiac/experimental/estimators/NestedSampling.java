package io.github.zzuegg.brainiac.experimental.estimators;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

/**
 * Nested sampling for the Bayesian evidence {@code Z = int L(theta) pi(theta) dtheta} (Skilling,
 * Bayesian Analysis 2006).
 *
 * <p>The prior is given as a transform of the unit hypercube ({@code theta = T(u)},
 * {@code u ~ U[0,1]^d}), so every sampler works in {@code u}. Starting from {@code N} live points
 * drawn from the prior, each iteration removes the live point with the lowest likelihood
 * {@code L*}, credits it with the prior-volume shell {@code X_{i-1} - X_i},
 * {@code X_i = exp(-i / N)}, and replaces it with a prior draw constrained to
 * {@code L > L*}. The constrained draw is the hard part:
 * <ul>
 *   <li>{@link Replacement#RANDOM_WALK}: a Metropolis walk from a random live point with a
 *       Gaussian step scaled to the spread of its nearest live points and adapted to about 50%
 *       acceptance;</li>
 *   <li>{@link Replacement#SLICE}: hit-and-run slice sampling along random directions, with
 *       stepping out and shrinkage (Neal 2003; Handley et al., PolyChord, 2015);</li>
 *   <li>{@link Replacement#MULTI_ELLIPSOID}: rejection sampling from a union of enlarged
 *       ellipsoids that bound the live points, split recursively with 2-means when that shrinks
 *       the total volume (Feroz, Hobson and Bridges, MultiNest, 2009).</li>
 * </ul>
 * The estimate carries the information {@code H} and the standard error {@code sqrt(H / N)},
 * and the dead points with importance weights give posterior samples.
 */
public final class NestedSampling {

    /** How a replacement point is drawn under the likelihood constraint. */
    public enum Replacement {
        /** Metropolis random walk from a live point. */
        RANDOM_WALK,
        /** Hit-and-run slice sampling. */
        SLICE,
        /** Rejection sampling from bounding ellipsoids. */
        MULTI_ELLIPSOID
    }

    /**
     * The evidence estimate.
     *
     * @param logZ the estimate of {@code log Z}
     * @param error the standard error {@code sqrt(H / N)}
     * @param information the information {@code H} (KL from prior to posterior), in nats
     * @param iterations the iterations run
     * @param likelihoodEvaluations the likelihood evaluations spent
     * @param samples the dead and final live points, in parameter space
     * @param logWeights their normalised log importance weights
     */
    public record Result(double logZ, double error, double information, int iterations, long likelihoodEvaluations,
            double[][] samples, double[] logWeights) {

        /**
         * Equal-weight posterior samples by systematic resampling.
         *
         * @param count the number of samples
         * @param random the random source
         * @return {@code count} points
         */
        public double[][] resample(int count, SplittableRandom random) {
            double[] cum = new double[logWeights.length];
            double c = 0;
            for (int i = 0; i < cum.length; i++) {
                c += Math.exp(logWeights[i]);
                cum[i] = c;
            }
            double[][] out = new double[count][];
            double u = random.nextDouble() / count;
            int j = 0;
            for (int i = 0; i < count; i++) {
                double t = (u + (double) i / count) * c;
                while (j < cum.length - 1 && cum[j] < t) {
                    j++;
                }
                out[i] = samples[j].clone();
            }
            return out;
        }
    }

    private final Replacement replacement;
    private final int livePoints;
    private final int walkSteps;
    private final double tolerance;
    private final int maxIterations;

    private NestedSampling(Replacement replacement, int livePoints, int walkSteps, double tolerance,
            int maxIterations) {
        this.replacement = Objects.requireNonNull(replacement, "replacement");
        this.livePoints = livePoints;
        this.walkSteps = walkSteps;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    /**
     * A nested sampler.
     *
     * @param replacement the constrained-prior sampler
     * @param livePoints the number of live points {@code N}
     * @param walkSteps proposals per replacement (walk), slices per dimension and replacement
     *     (slice), unused for ellipsoids
     * @param tolerance stop when the live points can add at most this to {@code log Z}, for
     *     example 0.01
     * @param maxIterations a hard cap on iterations
     * @return the sampler
     */
    public static NestedSampling of(Replacement replacement, int livePoints, int walkSteps, double tolerance,
            int maxIterations) {
        if (livePoints < 2) {
            throw new IllegalArgumentException("need at least 2 live points");
        }
        return new NestedSampling(replacement, livePoints, walkSteps, tolerance, maxIterations);
    }

    /**
     * Runs nested sampling.
     *
     * @param dimension the parameter dimension
     * @param priorTransform maps {@code u in [0,1]^d} to {@code theta}
     * @param logLikelihood {@code log L(theta)}
     * @param random the random source
     * @return the estimate
     */
    public Result run(int dimension, UnaryOperator<double[]> priorTransform, ToDoubleFunction<double[]> logLikelihood,
            SplittableRandom random) {
        int n = livePoints;
        long[] evaluations = new long[1];
        ToDoubleFunction<double[]> logL = u -> {
            evaluations[0]++;
            return logLikelihood.applyAsDouble(priorTransform.apply(u));
        };
        double[][] live = new double[n][];
        double[] liveL = new double[n];
        for (int i = 0; i < n; i++) {
            live[i] = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                live[i][d] = random.nextDouble();
            }
            liveL[i] = logL.applyAsDouble(live[i]);
        }
        List<double[]> dead = new ArrayList<>();
        List<Double> deadLogW = new ArrayList<>();
        List<Double> deadL = new ArrayList<>();
        double logZ = Double.NEGATIVE_INFINITY;
        double h = 0;
        double logWidth = Math.log(1 - Math.exp(-1.0 / n));
        double[] scale = {0.5};
        Ellipsoid[] bounds = null;
        int it = 0;
        for (; it < maxIterations; it++) {
            int worst = 0;
            double maxL = liveL[0];
            for (int i = 1; i < n; i++) {
                if (liveL[i] < liveL[worst]) {
                    worst = i;
                }
                maxL = Math.max(maxL, liveL[i]);
            }
            double lStar = liveL[worst];
            double logWt = logWidth + lStar;
            double newZ = logAdd(logZ, logWt);
            h = Math.exp(logWt - newZ) * lStar
                    + (Double.isFinite(logZ) ? Math.exp(logZ - newZ) * (h + logZ) : 0) - newZ;
            logZ = newZ;
            dead.add(live[worst].clone());
            deadLogW.add(logWt);
            deadL.add(lStar);
            logWidth -= 1.0 / n;
            // remaining evidence bound: L_max X_i
            if (maxL - (it + 1.0) / n < logZ + Math.log(tolerance)) {
                it++;
                break;
            }
            double[] fresh;
            double freshL;
            if (replacement == Replacement.MULTI_ELLIPSOID) {
                if (bounds == null || it % Math.max(1, n / 10) == 0) {
                    bounds = Ellipsoid.bound(live, worst, Math.exp(-(it + 1.0) / n), random);
                }
                double[] r = ellipsoidDraw(bounds, lStar, logL, random);
                fresh = Arrays.copyOf(r, dimension);
                freshL = r[dimension];
            } else {
                int start;
                do {
                    start = random.nextInt(n);
                } while (start == worst);
                double[] r = replacement == Replacement.SLICE ? slice(live[start], liveL[start], lStar, logL, live,
                        random) : walk(live[start], liveL[start], lStar, logL, live, scale, random);
                fresh = Arrays.copyOf(r, dimension);
                freshL = r[dimension];
            }
            live[worst] = fresh;
            liveL[worst] = freshL;
        }
        // the final live points share the remaining volume X = exp(-it / N)
        double logX = -(double) it / n - Math.log(n);
        for (int i = 0; i < n; i++) {
            double logWt = logX + liveL[i];
            double newZ = logAdd(logZ, logWt);
            h = Math.exp(logWt - newZ) * liveL[i] + Math.exp(logZ - newZ) * (h + logZ) - newZ;
            logZ = newZ;
            dead.add(live[i].clone());
            deadLogW.add(logWt);
            deadL.add(liveL[i]);
        }
        double[][] samples = new double[dead.size()][];
        double[] w = new double[dead.size()];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = priorTransform.apply(dead.get(i));
            w[i] = deadLogW.get(i) - logZ;
        }
        h = Math.max(h, 0);
        return new Result(logZ, Math.sqrt(h / n), h, it, evaluations[0], samples, w);
    }

    private static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        }
        if (b == Double.NEGATIVE_INFINITY) {
            return a;
        }
        double m = Math.max(a, b);
        return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    }

    private static boolean inCube(double[] u) {
        for (double v : u) {
            if (v < 0 || v > 1) {
                return false;
            }
        }
        return true;
    }

    /** Per-coordinate spread of the live points nearest to {@code centre}, so steps follow the local mode. */
    private static double[] spread(double[][] live, double[] centre) {
        int d = centre.length;
        int k = Math.min(live.length, Math.max(4 * d, 20));
        double[] dist = new double[live.length];
        for (int i = 0; i < live.length; i++) {
            dist[i] = MlMatrices.squaredDistance(live[i], centre);
        }
        double[] sorted = dist.clone();
        Arrays.sort(sorted);
        double cut = sorted[k - 1];
        double[] m = new double[d];
        int c = 0;
        for (int i = 0; i < live.length; i++) {
            if (dist[i] <= cut) {
                for (int j = 0; j < d; j++) {
                    m[j] += live[i][j];
                }
                c++;
            }
        }
        double[] s = new double[d];
        for (int j = 0; j < d; j++) {
            m[j] /= c;
        }
        for (int i = 0; i < live.length; i++) {
            if (dist[i] <= cut) {
                for (int j = 0; j < d; j++) {
                    s[j] += (live[i][j] - m[j]) * (live[i][j] - m[j]) / c;
                }
            }
        }
        for (int j = 0; j < d; j++) {
            s[j] = Math.sqrt(s[j]) + 1e-12;
        }
        return s;
    }

    private double[] walk(double[] start, double startL, double lStar, ToDoubleFunction<double[]> logL,
            double[][] live, double[] scale, SplittableRandom random) {
        int d = start.length;
        double[] sd = spread(live, start);
        double[] cur = start.clone();
        double curL = startL;
        int accepted = 0;
        // a fixed kernel within the walk keeps the uniform constrained prior invariant; the step
        // scale adapts between walks towards 50% acceptance
        for (int t = 0; t < walkSteps; t++) {
            double[] prop = new double[d];
            for (int j = 0; j < d; j++) {
                prop[j] = cur[j] + scale[0] * sd[j] * gaussian(random);
            }
            if (inCube(prop)) {
                double l = logL.applyAsDouble(prop);
                if (l > lStar) {
                    cur = prop;
                    curL = l;
                    accepted++;
                }
            }
        }
        scale[0] *= Math.exp(2 * ((double) accepted / walkSteps - 0.5));
        double[] out = Arrays.copyOf(cur, d + 1);
        out[d] = curL;
        return out;
    }

    private double[] slice(double[] start, double startL, double lStar, ToDoubleFunction<double[]> logL,
            double[][] live, SplittableRandom random) {
        int d = start.length;
        double[] sd = spread(live, start);
        double[] cur = start.clone();
        double curL = startL;
        for (int s = 0; s < walkSteps * d; s++) {
            double[] dir = new double[d];
            double norm = 0;
            for (int j = 0; j < d; j++) {
                dir[j] = gaussian(random);
                norm += dir[j] * dir[j];
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < d; j++) {
                dir[j] = dir[j] / norm * sd[j] * Math.sqrt(d);
            }
            double width = 1;
            double lo = -random.nextDouble() * width;
            double hi = lo + width;
            for (int k = 0; k < 20 && ok(cur, dir, lo, lStar, logL); k++) {
                lo -= width;
            }
            for (int k = 0; k < 20 && ok(cur, dir, hi, lStar, logL); k++) {
                hi += width;
            }
            for (int k = 0; k < 100; k++) {
                double t = lo + random.nextDouble() * (hi - lo);
                double[] p = at(cur, dir, t);
                double l = inCube(p) ? logL.applyAsDouble(p) : Double.NEGATIVE_INFINITY;
                if (l > lStar) {
                    cur = p;
                    curL = l;
                    break;
                }
                if (t < 0) {
                    lo = t;
                } else {
                    hi = t;
                }
            }
        }
        double[] out = Arrays.copyOf(cur, d + 1);
        out[d] = curL;
        return out;
    }

    private static double[] at(double[] x, double[] dir, double t) {
        double[] p = new double[x.length];
        for (int j = 0; j < p.length; j++) {
            p[j] = x[j] + t * dir[j];
        }
        return p;
    }

    private static boolean ok(double[] x, double[] dir, double t, double lStar, ToDoubleFunction<double[]> logL) {
        double[] p = at(x, dir, t);
        return inCube(p) && logL.applyAsDouble(p) > lStar;
    }

    private static double[] ellipsoidDraw(Ellipsoid[] bounds, double lStar, ToDoubleFunction<double[]> logL,
            SplittableRandom random) {
        double[] vol = new double[bounds.length];
        double total = 0;
        for (int e = 0; e < bounds.length; e++) {
            vol[e] = Math.exp(bounds[e].logVolume - bounds[0].logVolume);
            total += vol[e];
        }
        int d = bounds[0].centre.length;
        for (int attempt = 0; attempt < 1_000_000; attempt++) {
            double r = random.nextDouble() * total;
            int e = 0;
            while (e < bounds.length - 1 && r > vol[e]) {
                r -= vol[e];
                e++;
            }
            double[] p = bounds[e].sample(random);
            if (!inCube(p)) {
                continue;
            }
            int count = 0;
            for (Ellipsoid b : bounds) {
                if (b.contains(p)) {
                    count++;
                }
            }
            if (count > 1 && random.nextDouble() > 1.0 / count) {
                continue;
            }
            double l = logL.applyAsDouble(p);
            if (l > lStar) {
                double[] out = Arrays.copyOf(p, d + 1);
                out[d] = l;
                return out;
            }
        }
        throw new IllegalStateException("ellipsoid sampling failed to find a point above the likelihood constraint");
    }

    private static double gaussian(SplittableRandom random) {
        double u = random.nextDouble();
        double v = random.nextDouble();
        return Math.sqrt(-2 * Math.log(1 - u)) * Math.cos(2 * Math.PI * v);
    }

    /** A bounding ellipsoid {@code (x - c)^T A^-1 (x - c) <= 1} with {@code A = L L^T}. */
    private static final class Ellipsoid {
        static final double ENLARGE = 1.25;

        final double[] centre;
        final double[][] chol;
        final double[][] inverse;
        final double logVolume;

        private Ellipsoid(double[] centre, double[][] cov, double scale) {
            int d = centre.length;
            this.centre = centre;
            double[][] a = new double[d][d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    a[i][j] = cov[i][j] * scale + (i == j ? 1e-10 : 0);
                }
            }
            this.chol = MlMatrices.cholesky(a);
            this.inverse = MlMatrices.choleskyInverse(chol);
            this.logVolume = MlMatrices.logDet(chol) / 2;
        }

        boolean contains(double[] x) {
            int d = x.length;
            double s = 0;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    s += (x[i] - centre[i]) * inverse[i][j] * (x[j] - centre[j]);
                }
            }
            return s <= 1;
        }

        double[] sample(SplittableRandom random) {
            int d = centre.length;
            double[] z = new double[d];
            double norm = 0;
            for (int j = 0; j < d; j++) {
                z[j] = gaussian(random);
                norm += z[j] * z[j];
            }
            double r = Math.pow(random.nextDouble(), 1.0 / d) / Math.sqrt(norm);
            double[] p = centre.clone();
            for (int i = 0; i < d; i++) {
                for (int j = 0; j <= i; j++) {
                    p[i] += chol[i][j] * z[j] * r;
                }
            }
            return p;
        }

        /** The smallest covariance-shaped ellipsoid holding the points, enlarged in volume. */
        static Ellipsoid of(List<double[]> points, double targetLogVolume) {
            int d = points.get(0).length;
            double[] c = new double[d];
            for (double[] p : points) {
                for (int j = 0; j < d; j++) {
                    c[j] += p[j] / points.size();
                }
            }
            double[][] cov = new double[d][d];
            for (double[] p : points) {
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        cov[i][j] += (p[i] - c[i]) * (p[j] - c[j]) / points.size();
                    }
                }
            }
            for (int i = 0; i < d; i++) {
                cov[i][i] += 1e-12;
            }
            Ellipsoid unit = new Ellipsoid(c, cov, 1);
            double maxM = 0;
            for (double[] p : points) {
                double s = 0;
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        s += (p[i] - c[i]) * unit.inverse[i][j] * (p[j] - c[j]);
                    }
                }
                maxM = Math.max(maxM, s);
            }
            double scale = maxM * Math.pow(ENLARGE, 2.0 / d);
            // never smaller than the expected prior volume share (MultiNest's volume floor)
            double logVol = unit.logVolume + d * Math.log(scale) / 2;
            if (logVol < targetLogVolume) {
                scale *= Math.exp(2 * (targetLogVolume - logVol) / d);
            }
            return new Ellipsoid(c, cov, scale);
        }

        static Ellipsoid[] bound(double[][] live, int exclude, double priorVolume, SplittableRandom random) {
            List<double[]> pts = new ArrayList<>();
            for (int i = 0; i < live.length; i++) {
                if (i != exclude) {
                    pts.add(live[i]);
                }
            }
            int d = live[0].length;
            // log volume of a unit d-ball, so ellipsoid log volumes compare to the prior volume
            double logBall = d / 2.0 * Math.log(Math.PI) - ProbabilisticProgram.logGamma(d / 2.0 + 1);
            List<Ellipsoid> out = new ArrayList<>();
            split(pts, Math.log(priorVolume) - logBall, out, random, 0);
            return out.toArray(new Ellipsoid[0]);
        }

        private static void split(List<double[]> pts, double targetLogVolume, List<Ellipsoid> out,
                SplittableRandom random, int depth) {
            int d = pts.get(0).length;
            Ellipsoid whole = of(pts, targetLogVolume);
            if (pts.size() < 2 * (d + 1) || depth > 6) {
                out.add(whole);
                return;
            }
            List<List<double[]>> parts = twoMeans(pts, random);
            if (parts.get(0).size() <= d || parts.get(1).size() <= d) {
                out.add(whole);
                return;
            }
            double frac0 = (double) parts.get(0).size() / pts.size();
            Ellipsoid a = of(parts.get(0), targetLogVolume + Math.log(frac0));
            Ellipsoid b = of(parts.get(1), targetLogVolume + Math.log(1 - frac0));
            double splitVol = Math.log(Math.exp(a.logVolume - whole.logVolume) + Math.exp(b.logVolume
                    - whole.logVolume));
            if (splitVol < Math.log(0.5)) {
                split(parts.get(0), targetLogVolume + Math.log(frac0), out, random, depth + 1);
                split(parts.get(1), targetLogVolume + Math.log(1 - frac0), out, random, depth + 1);
            } else {
                out.add(whole);
            }
        }

        private static List<List<double[]>> twoMeans(List<double[]> pts, SplittableRandom random) {
            int d = pts.get(0).length;
            double[] c0 = pts.get(random.nextInt(pts.size())).clone();
            double[] c1 = c0;
            double far = -1;
            for (double[] p : pts) {
                double s = MlMatrices.squaredDistance(p, c0);
                if (s > far) {
                    far = s;
                    c1 = p.clone();
                }
            }
            int[] assign = new int[pts.size()];
            for (int iter = 0; iter < 20; iter++) {
                double[][] sum = new double[2][d];
                int[] count = new int[2];
                for (int i = 0; i < pts.size(); i++) {
                    double[] p = pts.get(i);
                    assign[i] = MlMatrices.squaredDistance(p, c0) <= MlMatrices.squaredDistance(p, c1) ? 0 : 1;
                    count[assign[i]]++;
                    for (int j = 0; j < d; j++) {
                        sum[assign[i]][j] += p[j];
                    }
                }
                if (count[0] == 0 || count[1] == 0) {
                    break;
                }
                for (int j = 0; j < d; j++) {
                    c0[j] = sum[0][j] / count[0];
                    c1[j] = sum[1][j] / count[1];
                }
            }
            List<List<double[]>> parts = List.of(new ArrayList<>(), new ArrayList<>());
            for (int i = 0; i < pts.size(); i++) {
                parts.get(assign[i]).add(pts.get(i));
            }
            return parts;
        }
    }
}
