package io.github.zzuegg.brainiac.experimental.serving;

import java.util.SplittableRandom;

/**
 * MagicPIG: attention estimated by sampling keys with locality-sensitive hashing (Chen et al.,
 * ICLR 2025, <a href="https://arxiv.org/abs/2410.16179">arXiv:2410.16179</a>). Top-k attention
 * drops the long tail of the softmax and is biased when attention is not sparse. MagicPIG
 * instead hashes every key with SimHash into {@code L} tables of {@code K} sign bits; a key is
 * sampled when it collides with the query in at least {@code minCollisions} tables. The
 * probability of that is known in closed form from the angle {@code theta} between query and
 * key: one table collides with {@code p = (1 - theta / pi)^K}, and the number of colliding
 * tables is binomial. Dividing each sampled key's softmax weight by its sampling probability
 * gives a self-normalised importance-sampling estimate of the attention output, so dense heads
 * are estimated without the top-k bias. The first {@code sinks} and last {@code local} tokens
 * are computed exactly, as in the paper; the hash tables need only key signs, so they can live
 * in CPU memory.
 *
 * <p>Build the tables once per KV set with {@link #index}, then call {@link Index#attend} per
 * query.
 */
public final class LshSampledAttention {

    private final int bits;
    private final int tables;
    private final int minCollisions;
    private final int sinks;
    private final int local;
    private final float[][] planes;   // [tables * bits][dim]

    private LshSampledAttention(int dim, int bits, int tables, int minCollisions, int sinks, int local, long seed) {
        if (bits < 1 || tables < 1 || minCollisions < 1 || minCollisions > tables || sinks < 0 || local < 0) {
            throw new IllegalArgumentException("need bits, tables >= 1 and 1 <= minCollisions <= tables");
        }
        this.bits = bits;
        this.tables = tables;
        this.minCollisions = minCollisions;
        this.sinks = sinks;
        this.local = local;
        SplittableRandom r = new SplittableRandom(seed);
        planes = new float[tables * bits][dim];
        for (float[] p : planes) {
            for (int c = 0; c < dim; c++) {
                p[c] = (float) r.nextGaussian();
            }
        }
    }

    /**
     * @param dim the head dimension
     * @param bits sign bits per table (K)
     * @param tables hash tables (L)
     * @param minCollisions tables a key must share with the query to be sampled
     * @param sinks leading tokens attended exactly
     * @param local trailing tokens attended exactly
     * @param seed the hyperplane seed
     * @return the sampler
     */
    public static LshSampledAttention of(int dim, int bits, int tables, int minCollisions, int sinks, int local,
            long seed) {
        return new LshSampledAttention(dim, bits, tables, minCollisions, sinks, local, seed);
    }

    /**
     * The probability that a key at angle {@code theta} from the query is sampled.
     *
     * @param theta the angle in radians
     * @return the probability
     */
    public double samplingProbability(double theta) {
        double p = Math.pow(1 - theta / Math.PI, bits);
        double tail = 0;
        for (int m = minCollisions; m <= tables; m++) {
            tail += binomial(tables, m) * Math.pow(p, m) * Math.pow(1 - p, tables - m);
        }
        return tail;
    }

    private static double binomial(int n, int k) {
        double c = 1;
        for (int i = 1; i <= k; i++) {
            c = c * (n - k + i) / i;
        }
        return c;
    }

    private long[] codes(float[] x) {
        long[] code = new long[tables];
        for (int t = 0; t < tables; t++) {
            long c = 0;
            for (int b = 0; b < bits; b++) {
                float[] p = planes[t * bits + b];
                double d = 0;
                for (int i = 0; i < x.length; i++) {
                    d += p[i] * x[i];
                }
                c = (c << 1) | (d >= 0 ? 1 : 0);
            }
            code[t] = c;
        }
        return code;
    }

    /**
     * Hashes a KV set.
     *
     * @param keys the keys {@code [n][dim]}
     * @param values the values {@code [n][dv]}
     * @return the index
     */
    public Index index(float[][] keys, float[][] values) {
        long[][] codes = new long[keys.length][];
        for (int i = 0; i < keys.length; i++) {
            codes[i] = codes(keys[i]);
        }
        return new Index(keys, values, codes);
    }

    /** The hashed keys of one KV set. */
    public final class Index {
        private final float[][] keys;
        private final float[][] values;
        private final long[][] codes;
        private long touched;
        private long queries;

        private Index(float[][] keys, float[][] values, long[][] codes) {
            this.keys = keys;
            this.values = values;
            this.codes = codes;
        }

        /**
         * Estimates {@code softmax(scale q K^T) V}.
         *
         * @param q the query
         * @param scale the logit scale
         * @return the estimated output
         */
        public float[] attend(float[] q, float scale) {
            long[] qc = codes(q);
            int n = keys.length;
            int dv = values[0].length;
            double qn = norm(q);
            double[] acc = new double[dv];
            double z = 0;
            double max = Double.NEGATIVE_INFINITY;
            double[] logit = new double[n];
            double[] weight = new double[n];
            for (int i = 0; i < n; i++) {
                boolean exact = i < sinks || i >= n - local;
                double u = 1;
                if (!exact) {
                    int hits = 0;
                    for (int t = 0; t < tables; t++) {
                        hits += codes[i][t] == qc[t] ? 1 : 0;
                    }
                    if (hits < minCollisions) {
                        continue;
                    }
                    double cos = dot(q, keys[i]) / (qn * norm(keys[i]) + 1e-12);
                    u = Math.max(1e-12, samplingProbability(Math.acos(Math.max(-1, Math.min(1, cos)))));
                }
                touched++;
                logit[i] = scale * dot(q, keys[i]);
                weight[i] = 1 / u;
                max = Math.max(max, logit[i]);
            }
            queries++;
            for (int i = 0; i < n; i++) {
                if (weight[i] == 0) {
                    continue;
                }
                double w = Math.exp(logit[i] - max) * weight[i];
                z += w;
                for (int c = 0; c < dv; c++) {
                    acc[c] += w * values[i][c];
                }
            }
            float[] out = new float[dv];
            if (z > 0) {
                for (int c = 0; c < dv; c++) {
                    out[c] = (float) (acc[c] / z);
                }
            }
            return out;
        }

        /** @return the mean keys whose logits were computed per query */
        public double keysTouchedPerQuery() {
            return queries == 0 ? 0 : touched / (double) queries;
        }
    }

    private static double dot(float[] a, float[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
        }
        return d;
    }

    private static double norm(float[] a) {
        return Math.sqrt(dot(a, a));
    }

    @Override
    public String toString() {
        return "MagicPIG(K " + bits + ", L " + tables + ", collisions >= " + minCollisions + ")";
    }
}
