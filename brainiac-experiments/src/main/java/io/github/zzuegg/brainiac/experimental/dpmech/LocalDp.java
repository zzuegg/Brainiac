package io.github.zzuegg.brainiac.experimental.dpmech;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Frequency estimation under local differential privacy: every user randomises their own
 * value in {@code [0, k)} before sending it, and the server decodes unbiased frequency
 * estimates (Wang et al., USENIX Security 2017,
 * <a href="https://arxiv.org/abs/1705.04421">arXiv:1705.04421</a>):
 *
 * <ul>
 *   <li>{@link Protocol#GRR}: k-ary randomized response reports the true value with
 *       probability {@code p = e^ε/(e^ε + k - 1)}, otherwise one of the others uniformly
 *       (Warner 1965).</li>
 *   <li>{@link Protocol#RAPPOR}: basic one-time RAPPOR, a one-hot vector whose bits are kept
 *       with probability {@code e^{ε/2}/(e^{ε/2} + 1)} (symmetric unary encoding; Erlingsson,
 *       Pihur and Korolova, CCS 2014, <a href="https://arxiv.org/abs/1407.6981">arXiv:1407.6981</a>).</li>
 *   <li>{@link Protocol#OUE}: optimised unary encoding, the 1 kept with probability 1/2 and
 *       each 0 flipped with probability {@code 1/(e^ε + 1)}.</li>
 *   <li>{@link Protocol#OLH}: optimised local hashing hashes the value into
 *       {@code g = round(e^ε + 1)} buckets with the user's own hash seed, then applies GRR
 *       over the buckets; the server counts the users whose report supports each value.</li>
 * </ul>
 *
 * <p>Every decoder gives {@code f̂_v = (C_v / n - q*) / (p* - q*)}; {@link #projectSimplex} and
 * {@link #normSub} post-process the estimate into a distribution, which removes the negative
 * entries and lowers the error. {@link #centralLaplace} is the trusted-curator baseline.
 */
public final class LocalDp {

    /** A local randomiser. */
    public enum Protocol {
        /** generalised randomized response */
        GRR,
        /** basic RAPPOR (symmetric unary encoding) */
        RAPPOR,
        /** optimised unary encoding */
        OUE,
        /** optimised local hashing */
        OLH
    }

    private LocalDp() {
    }

    /**
     * Simulates every user's report and decodes the frequencies.
     *
     * @param protocol the randomiser
     * @param values one value per user in {@code [0, k)}
     * @param k the domain size
     * @param epsilon the per-user privacy budget
     * @param seed the seed
     * @return the unbiased estimate of the frequency vector (may have negative entries)
     */
    public static double[] estimate(Protocol protocol, int[] values, int k, double epsilon, long seed) {
        Objects.requireNonNull(protocol);
        SplittableRandom r = new SplittableRandom(seed);
        int n = values.length;
        double e = Math.exp(epsilon);
        double[] count = new double[k];
        double p;
        double q;
        switch (protocol) {
            case GRR -> {
                p = e / (e + k - 1);
                q = 1 / (e + k - 1);
                for (int v : values) {
                    count[grr(v, k, p, r)]++;
                }
            }
            case RAPPOR, OUE -> {
                double half = Math.exp(epsilon / 2);
                p = protocol == Protocol.RAPPOR ? half / (half + 1) : 0.5;
                q = protocol == Protocol.RAPPOR ? 1 / (half + 1) : 1 / (e + 1);
                for (int v : values) {
                    for (int j = 0; j < k; j++) {
                        if (r.nextDouble() < (j == v ? p : q)) {
                            count[j]++;
                        }
                    }
                }
            }
            case OLH -> {
                int g = Math.max(2, (int) Math.round(e + 1));
                p = e / (e + g - 1);
                q = 1.0 / g;
                for (int v : values) {
                    long hashSeed = r.nextLong();
                    int report = grr(hash(hashSeed, v, g), g, p, r);
                    for (int j = 0; j < k; j++) {
                        if (hash(hashSeed, j, g) == report) {
                            count[j]++;
                        }
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
        double[] f = new double[k];
        for (int j = 0; j < k; j++) {
            f[j] = (count[j] / n - q) / (p - q);
        }
        return f;
    }

    /**
     * The analytical variance of one estimated frequency (for a value of small frequency),
     * from Wang et al.'s Table 1.
     *
     * @param protocol the randomiser
     * @param n users
     * @param k domain size
     * @param epsilon the budget
     * @return {@code Var[f̂_v]}
     */
    public static double variance(Protocol protocol, int n, int k, double epsilon) {
        double e = Math.exp(epsilon);
        return switch (protocol) {
            case GRR -> (e + k - 2) / (n * (e - 1) * (e - 1));
            case RAPPOR -> {
                double h = Math.exp(epsilon / 2);
                yield h / (n * (h - 1) * (h - 1));
            }
            case OUE, OLH -> 4 * e / (n * (e - 1) * (e - 1));
        };
    }

    /**
     * Report bits per user.
     *
     * @param protocol the randomiser
     * @param k domain size
     * @param epsilon the budget
     * @return the communication cost
     */
    public static double bits(Protocol protocol, int k, double epsilon) {
        return switch (protocol) {
            case GRR -> Math.ceil(Math.log(k) / Math.log(2));
            case RAPPOR, OUE -> k;
            case OLH -> 64 + Math.ceil(Math.log(Math.max(2, Math.round(Math.exp(epsilon) + 1))) / Math.log(2));
        };
    }

    private static int grr(int v, int k, double p, SplittableRandom r) {
        if (r.nextDouble() < p) {
            return v;
        }
        int o = r.nextInt(k - 1);
        return o >= v ? o + 1 : o;
    }

    /**
     * A seeded universal-style hash into {@code g} buckets (SplitMix64 finaliser).
     *
     * @param seed the user's hash seed
     * @param v the value
     * @param g buckets
     * @return the bucket
     */
    public static int hash(long seed, int v, int g) {
        long z = seed + 0x9E3779B97F4A7C15L * (v + 1);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return Math.floorMod(z, g);
    }

    /**
     * Euclidean projection onto the probability simplex.
     *
     * @param f the estimate
     * @return the closest distribution
     */
    public static double[] projectSimplex(double[] f) {
        double[] s = f.clone();
        Arrays.sort(s);
        double cum = 0;
        double theta = 0;
        for (int i = s.length - 1; i >= 0; i--) {
            cum += s[i];
            double t = (cum - 1) / (s.length - i);
            if (i == 0 || s[i - 1] <= t) {
                theta = t;
                break;
            }
        }
        double[] out = new double[f.length];
        for (int j = 0; j < f.length; j++) {
            out[j] = Math.max(0, f[j] - theta);
        }
        return out;
    }

    /**
     * Norm-Sub: zero the negative entries and shift the positive ones by a common amount so
     * the result sums to one (Wang et al., NDSS 2020).
     *
     * @param f the estimate
     * @return the post-processed distribution
     */
    public static double[] normSub(double[] f) {
        double lo = -1;
        double hi = 1;
        for (double v : f) {
            hi = Math.max(hi, v);
        }
        for (int it = 0; it < 100; it++) {
            double mid = 0.5 * (lo + hi);
            double s = 0;
            for (double v : f) {
                s += Math.max(0, v - mid);
            }
            if (s > 1) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        double d = 0.5 * (lo + hi);
        double[] out = new double[f.length];
        for (int j = 0; j < f.length; j++) {
            out[j] = Math.max(0, f[j] - d);
        }
        return out;
    }

    /**
     * The central-model baseline: a histogram with Laplace noise of scale {@code 1/ε} per
     * count (add/remove-one neighbours), normalised by {@code n}.
     *
     * @param values the users' values
     * @param k domain size
     * @param epsilon the budget
     * @param seed the seed
     * @return the noisy frequencies
     */
    public static double[] centralLaplace(int[] values, int k, double epsilon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] f = new double[k];
        for (int v : values) {
            f[v]++;
        }
        for (int j = 0; j < k; j++) {
            f[j] = (f[j] + laplace(1 / epsilon, r)) / values.length;
        }
        return f;
    }

    /**
     * A Laplace sample.
     *
     * @param scale the scale {@code b}
     * @param r the random source
     * @return a draw from {@code Lap(0, b)}
     */
    public static double laplace(double scale, SplittableRandom r) {
        double u = r.nextDouble() - 0.5;
        return -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
    }

    /**
     * Mean squared error.
     *
     * @param estimate the estimate
     * @param truth the true frequencies
     * @return the MSE over the domain
     */
    public static double mse(double[] estimate, double[] truth) {
        double s = 0;
        for (int j = 0; j < truth.length; j++) {
            s += (estimate[j] - truth[j]) * (estimate[j] - truth[j]) / truth.length;
        }
        return s;
    }

    /**
     * Recall of the true top-{@code m} values among the estimated top-{@code m}.
     *
     * @param estimate the estimate
     * @param truth the true frequencies
     * @param m the list length
     * @return the recall
     */
    public static double topRecall(double[] estimate, double[] truth, int m) {
        boolean[] a = top(estimate, m);
        boolean[] b = top(truth, m);
        int hit = 0;
        for (int j = 0; j < a.length; j++) {
            hit += a[j] && b[j] ? 1 : 0;
        }
        return hit / (double) m;
    }

    private static boolean[] top(double[] v, int m) {
        Integer[] order = new Integer[v.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(v[b], v[a]));
        boolean[] out = new boolean[v.length];
        for (int i = 0; i < Math.min(m, v.length); i++) {
            out[order[i]] = true;
        }
        return out;
    }
}
