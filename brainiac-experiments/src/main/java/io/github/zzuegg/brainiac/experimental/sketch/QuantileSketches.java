package io.github.zzuegg.brainiac.experimental.sketch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Streaming quantile (rank) sketches, all mergeable:
 *
 * <ul>
 *   <li>{@link #greenwaldKhanna}: tuples {@code (v, g, Δ)} whose rank bounds are kept within
 *       {@code 2εn}, compressed every {@code 1/(2ε)} inserts; deterministic {@code ε}-rank
 *       error (Greenwald and Khanna, SIGMOD 2001).</li>
 *   <li>{@link #kll}: a hierarchy of compactors whose capacity shrinks by {@code 2/3} per level
 *       below the top; a full compactor sorts its items and promotes every other one (random
 *       offset) with doubled weight, which gives the optimal space for a randomised sketch
 *       (Karnin, Lang and Liberty, FOCS 2016, <a href="https://arxiv.org/abs/1603.05346">arXiv:1603.05346</a>).</li>
 *   <li>{@link #tDigest}: a merging t-digest whose centroids are bounded by the {@code k1} scale
 *       function {@code k(q) = δ/(2π) asin(2q - 1)}, so clusters are tiny near the tails and
 *       extreme quantiles are accurate (Dunning and Ertl,
 *       <a href="https://arxiv.org/abs/1902.04023">arXiv:1902.04023</a>).</li>
 * </ul>
 */
public final class QuantileSketches {

    private QuantileSketches() {
    }

    /** A streaming summary that answers rank and quantile queries. */
    public interface QuantileSketch {

        /**
         * Adds a value.
         *
         * @param v the value
         */
        void add(double v);

        /**
         * @param q a probability in {@code [0, 1]}
         * @return an estimate of the {@code q}-quantile
         */
        double quantile(double q);

        /** @return the number of values added */
        long count();

        /** @return the number of stored numbers (values, weights, bounds) */
        int stored();

        /**
         * Folds another sketch of the same kind into this one.
         *
         * @param other the other sketch
         */
        void merge(QuantileSketch other);
    }

    /**
     * A Greenwald–Khanna sketch.
     *
     * @param epsilon the rank error
     * @return the sketch
     */
    public static QuantileSketch greenwaldKhanna(double epsilon) {
        return new Gk(epsilon);
    }

    /**
     * A KLL sketch.
     *
     * @param k the top compactor's capacity (space is about {@code 3k})
     * @param seed the seed of the compaction coins
     * @return the sketch
     */
    public static QuantileSketch kll(int k, long seed) {
        return new Kll(k, seed);
    }

    /**
     * A merging t-digest.
     *
     * @param compression {@code δ}, about the number of centroids kept
     * @return the sketch
     */
    public static QuantileSketch tDigest(double compression) {
        return new TDigest(compression);
    }

    /** Greenwald–Khanna. */
    private static final class Gk implements QuantileSketch {
        private final double eps;
        private final List<double[]> tuples = new ArrayList<>();
        private long n;

        Gk(double eps) {
            this.eps = eps;
        }

        @Override
        public void add(double v) {
            int i = 0;
            int lo = 0;
            int hi = tuples.size();
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (tuples.get(mid)[0] <= v) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            i = lo;
            double delta = i == 0 || i == tuples.size() ? 0 : Math.max(0, Math.floor(2 * eps * n) - 1);
            tuples.add(i, new double[] {v, 1, delta});
            n++;
            if (n % Math.max(1, (long) (1 / (2 * eps))) == 0) {
                compress();
            }
        }

        private void compress() {
            double bound = Math.floor(2 * eps * n);
            for (int i = tuples.size() - 2; i >= 1; i--) {
                double[] a = tuples.get(i);
                double[] b = tuples.get(i + 1);
                if (a[1] + b[1] + b[2] <= bound) {
                    b[1] += a[1];
                    tuples.remove(i);
                }
            }
        }

        @Override
        public double quantile(double q) {
            if (tuples.isEmpty()) {
                return Double.NaN;
            }
            // the tuple whose rank interval [rmin, rmax] is closest to r on both sides; after a
            // merge the guarantee is the sum of the inputs' rank errors, not eps
            double r = Math.ceil(q * n);
            double rmin = 0;
            double bestGap = Double.POSITIVE_INFINITY;
            double best = tuples.get(0)[0];
            for (double[] t : tuples) {
                rmin += t[1];
                double gap = Math.max(r - rmin, rmin + t[2] - r);
                if (gap < bestGap) {
                    bestGap = gap;
                    best = t[0];
                }
            }
            return best;
        }

        @Override
        public long count() {
            return n;
        }

        @Override
        public int stored() {
            return 3 * tuples.size();
        }

        @Override
        public void merge(QuantileSketch other) {
            Gk o = (Gk) other;
            // the rank-bound combine: a tuple of one summary at value v gets
            // rmin = its rmin + the other summary's rmin of its last tuple <= v, and
            // rmax = its rmax + the other's rmax of its first tuple > v, minus one (or the
            // other's count when there is none); then back to (v, g, delta) and recompress
            double[][] a = ranks(tuples);
            double[][] b = ranks(o.tuples);
            List<double[]> all = new ArrayList<>();
            combine(a, b, o.n, false, all);
            combine(b, a, n, true, all);
            all.sort((x, y) -> Double.compare(x[0], y[0]));
            tuples.clear();
            double prevMin = 0;
            for (double[] t : all) {
                double g = Math.max(0, t[1] - prevMin);
                tuples.add(new double[] {t[0], g, Math.max(0, t[2] - t[1])});
                prevMin = Math.max(prevMin, t[1]);
            }
            n += o.n;
            compress();
        }

        /** {value, rmin, rmax} per tuple. */
        private static double[][] ranks(List<double[]> list) {
            double[][] r = new double[list.size()][];
            double rmin = 0;
            for (int i = 0; i < r.length; i++) {
                double[] t = list.get(i);
                rmin += t[1];
                r[i] = new double[] {t[0], rmin, rmin + t[2]};
            }
            return r;
        }

        private static void combine(double[][] self, double[][] other, long otherCount, boolean strict,
                List<double[]> out) {
            int j = 0;
            for (double[] t : self) {
                // other tuples before t: value <= v (or < v for the second summary, so ties order)
                while (j < other.length && (strict ? other[j][0] < t[0] : other[j][0] <= t[0])) {
                    j++;
                }
                double predMin = j > 0 ? other[j - 1][1] : 0;
                double succMax = j < other.length ? other[j][2] - 1 : otherCount;
                out.add(new double[] {t[0], t[1] + predMin, t[2] + Math.max(predMin, succMax)});
            }
        }
    }

    /** KLL. */
    private static final class Kll implements QuantileSketch {
        private final int k;
        private final SplittableRandom random;
        private final List<double[]> levels = new ArrayList<>();
        private final List<Integer> sizes = new ArrayList<>();
        private long n;

        Kll(int k, long seed) {
            this.k = k;
            this.random = new SplittableRandom(seed);
            grow();
        }

        private void grow() {
            levels.add(new double[8]);
            sizes.add(0);
        }

        private int capacity(int level) {
            int depth = levels.size() - 1 - level;
            return Math.max(2, (int) Math.ceil(k * Math.pow(2.0 / 3, depth)));
        }

        private void push(int level, double v) {
            double[] buf = levels.get(level);
            int s = sizes.get(level);
            if (s == buf.length) {
                buf = Arrays.copyOf(buf, s * 2);
                levels.set(level, buf);
            }
            buf[s] = v;
            sizes.set(level, s + 1);
        }

        @Override
        public void add(double v) {
            push(0, v);
            n++;
            compact();
        }

        private void compact() {
            for (int h = 0; h < levels.size(); h++) {
                if (sizes.get(h) >= capacity(h)) {
                    if (h + 1 == levels.size()) {
                        grow();
                    }
                    double[] buf = levels.get(h);
                    int s = sizes.get(h);
                    Arrays.sort(buf, 0, s);
                    int offset = random.nextInt(2);
                    int keep = s % 2 == 1 ? 1 : 0;
                    // an odd item stays behind at this level
                    double leftover = buf[s - 1];
                    for (int i = offset; i < s - keep; i += 2) {
                        push(h + 1, buf[i]);
                    }
                    sizes.set(h, 0);
                    if (keep == 1) {
                        push(h, leftover);
                    }
                }
            }
        }

        @Override
        public double quantile(double q) {
            int total = 0;
            for (int s : sizes) {
                total += s;
            }
            if (total == 0) {
                return Double.NaN;
            }
            double[][] items = new double[total][2];
            int c = 0;
            for (int h = 0; h < levels.size(); h++) {
                double[] buf = levels.get(h);
                for (int i = 0; i < sizes.get(h); i++) {
                    items[c][0] = buf[i];
                    items[c++][1] = 1L << h;
                }
            }
            Arrays.sort(items, (a, b) -> Double.compare(a[0], b[0]));
            double weight = 0;
            for (double[] it : items) {
                weight += it[1];
            }
            double target = q * weight;
            double acc = 0;
            for (double[] it : items) {
                acc += it[1];
                if (acc >= target) {
                    return it[0];
                }
            }
            return items[items.length - 1][0];
        }

        @Override
        public long count() {
            return n;
        }

        @Override
        public int stored() {
            int s = 0;
            for (int v : sizes) {
                s += v;
            }
            return s;
        }

        @Override
        public void merge(QuantileSketch other) {
            Kll o = (Kll) other;
            while (levels.size() < o.levels.size()) {
                grow();
            }
            for (int h = 0; h < o.levels.size(); h++) {
                double[] buf = o.levels.get(h);
                for (int i = 0; i < o.sizes.get(h); i++) {
                    push(h, buf[i]);
                }
            }
            n += o.n;
            compact();
        }
    }

    /** Merging t-digest with the k1 scale function. */
    private static final class TDigest implements QuantileSketch {
        private final double delta;
        private double[] mean = new double[0];
        private double[] weight = new double[0];
        private final double[] buffer;
        private int buffered;
        private long n;
        private double min = Double.POSITIVE_INFINITY;
        private double max = Double.NEGATIVE_INFINITY;

        TDigest(double delta) {
            this.delta = delta;
            this.buffer = new double[(int) Math.ceil(5 * delta)];
        }

        @Override
        public void add(double v) {
            buffer[buffered++] = v;
            n++;
            min = Math.min(min, v);
            max = Math.max(max, v);
            if (buffered == buffer.length) {
                flush();
            }
        }

        private double k(double q) {
            return delta / (2 * Math.PI) * Math.asin(2 * Math.max(0, Math.min(1, q)) - 1);
        }

        private void flush() {
            if (buffered == 0) {
                return;
            }
            int total = mean.length + buffered;
            double[][] all = new double[total][2];
            for (int i = 0; i < mean.length; i++) {
                all[i][0] = mean[i];
                all[i][1] = weight[i];
            }
            for (int i = 0; i < buffered; i++) {
                all[mean.length + i][0] = buffer[i];
                all[mean.length + i][1] = 1;
            }
            buffered = 0;
            mergeSorted(all);
        }

        private void mergeSorted(double[][] all) {
            Arrays.sort(all, (a, b) -> Double.compare(a[0], b[0]));
            double totalWeight = 0;
            for (double[] c : all) {
                totalWeight += c[1];
            }
            List<double[]> out = new ArrayList<>();
            double soFar = 0;
            double[] cur = all[0].clone();
            double qLeft = 0;
            double limit = k(qLeft) + 1;
            for (int i = 1; i < all.length; i++) {
                double q = (soFar + cur[1] + all[i][1]) / totalWeight;
                if (k(q) <= limit) {
                    double w = cur[1] + all[i][1];
                    cur[0] += (all[i][0] - cur[0]) * all[i][1] / w;
                    cur[1] = w;
                } else {
                    out.add(cur);
                    soFar += cur[1];
                    qLeft = soFar / totalWeight;
                    limit = k(qLeft) + 1;
                    cur = all[i].clone();
                }
            }
            out.add(cur);
            mean = new double[out.size()];
            weight = new double[out.size()];
            for (int i = 0; i < out.size(); i++) {
                mean[i] = out.get(i)[0];
                weight[i] = out.get(i)[1];
            }
        }

        @Override
        public double quantile(double q) {
            flush();
            if (mean.length == 0) {
                return Double.NaN;
            }
            if (mean.length == 1) {
                return mean[0];
            }
            double total = n;
            double target = q * total;
            // interpolate between centroid centres, which sit at cumulative weight w_<i + w_i/2
            double left = weight[0] / 2;
            if (target <= left) {
                return min + (mean[0] - min) * (left > 0 ? target / left : 0);
            }
            double acc = left;
            for (int i = 0; i < mean.length - 1; i++) {
                double step = (weight[i] + weight[i + 1]) / 2;
                if (target <= acc + step) {
                    double f = (target - acc) / step;
                    return mean[i] + f * (mean[i + 1] - mean[i]);
                }
                acc += step;
            }
            double right = weight[mean.length - 1] / 2;
            double f = right > 0 ? Math.min(1, (target - acc) / right) : 1;
            return mean[mean.length - 1] + f * (max - mean[mean.length - 1]);
        }

        @Override
        public long count() {
            return n;
        }

        @Override
        public int stored() {
            return 2 * mean.length + buffered;
        }

        @Override
        public void merge(QuantileSketch other) {
            TDigest o = (TDigest) other;
            flush();
            o.flush();
            double[][] all = new double[mean.length + o.mean.length][2];
            for (int i = 0; i < mean.length; i++) {
                all[i][0] = mean[i];
                all[i][1] = weight[i];
            }
            for (int i = 0; i < o.mean.length; i++) {
                all[mean.length + i][0] = o.mean[i];
                all[mean.length + i][1] = o.weight[i];
            }
            n += o.n;
            min = Math.min(min, o.min);
            max = Math.max(max, o.max);
            if (all.length > 0) {
                mergeSorted(all);
            }
        }
    }
}
