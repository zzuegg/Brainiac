package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;

/**
 * Affinity propagation (Frey and Dueck, Science 2007): exemplar-based clustering by passing
 * responsibilities {@code r(i, k) = s(i, k) - max_{k' != k} (a(i, k') + s(i, k'))} and
 * availabilities {@code a(i, k) = min(0, r(k, k) + sum_{i' not in {i, k}} max(0, r(i', k)))}
 * ({@code a(k, k) = sum_{i' != k} max(0, r(i', k))}) between points, with damping. The
 * similarity is the negative squared Euclidean distance and the self-similarity (preference)
 * sets how many exemplars emerge; the median similarity is the customary default.
 */
public final class AffinityPropagation {

    /**
     * The clustering.
     *
     * @param labels cluster labels {@code 0 ..}
     * @param exemplars the exemplar point of each cluster
     * @param iterations the iterations run
     */
    public record Result(int[] labels, int[] exemplars, int iterations) {
    }

    private AffinityPropagation() {
    }

    /**
     * Clusters points.
     *
     * @param x the points {@code [n][d]}
     * @param preference the self-similarity, or {@code NaN} for the median similarity
     * @param damping the message damping in {@code [0.5, 1)}
     * @param maxIterations the iteration cap
     * @param convergence stop when the exemplar set is unchanged this many iterations
     * @return the clustering
     */
    public static Result cluster(double[][] x, double preference, double damping, int maxIterations,
            int convergence) {
        int n = x.length;
        double[][] s = MlMatrices.squaredDistances(x);
        double[] off = new double[n * (n - 1)];
        int c = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                s[i][k] = -s[i][k];
                if (i != k) {
                    off[c++] = s[i][k];
                }
            }
        }
        double pref = preference;
        if (Double.isNaN(pref)) {
            Arrays.sort(off);
            pref = off[off.length / 2];
        }
        // tiny deterministic jitter breaks ties between equally good exemplars
        for (int i = 0; i < n; i++) {
            s[i][i] = pref;
            for (int k = 0; k < n; k++) {
                s[i][k] += 1e-12 * Math.abs(s[i][k]) * (((i * 31 + k * 17) % 97) / 97.0);
            }
        }
        double[][] r = new double[n][n];
        double[][] a = new double[n][n];
        boolean[] last = new boolean[n];
        int stable = 0;
        int it = 0;
        for (; it < maxIterations; it++) {
            for (int i = 0; i < n; i++) {
                double first = Double.NEGATIVE_INFINITY;
                double second = Double.NEGATIVE_INFINITY;
                int arg = -1;
                for (int k = 0; k < n; k++) {
                    double v = a[i][k] + s[i][k];
                    if (v > first) {
                        second = first;
                        first = v;
                        arg = k;
                    } else if (v > second) {
                        second = v;
                    }
                }
                for (int k = 0; k < n; k++) {
                    double nr = s[i][k] - (k == arg ? second : first);
                    r[i][k] = damping * r[i][k] + (1 - damping) * nr;
                }
            }
            for (int k = 0; k < n; k++) {
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    if (i != k) {
                        sum += Math.max(0, r[i][k]);
                    }
                }
                for (int i = 0; i < n; i++) {
                    double na = i == k ? sum : Math.min(0, r[k][k] + sum - Math.max(0, r[i][k]));
                    a[i][k] = damping * a[i][k] + (1 - damping) * na;
                }
            }
            boolean[] ex = new boolean[n];
            boolean same = true;
            for (int k = 0; k < n; k++) {
                ex[k] = r[k][k] + a[k][k] > 0;
                same &= ex[k] == last[k];
            }
            last = ex;
            stable = same ? stable + 1 : 0;
            if (stable >= convergence && it > convergence) {
                it++;
                break;
            }
        }
        int count = 0;
        for (boolean e : last) {
            count += e ? 1 : 0;
        }
        int[] exemplars = new int[Math.max(count, 1)];
        if (count == 0) {
            // no exemplar emerged: take the point with the largest r + a
            int arg = 0;
            for (int k = 1; k < n; k++) {
                if (r[k][k] + a[k][k] > r[arg][arg] + a[arg][arg]) {
                    arg = k;
                }
            }
            exemplars[0] = arg;
        } else {
            int e = 0;
            for (int k = 0; k < n; k++) {
                if (last[k]) {
                    exemplars[e++] = k;
                }
            }
        }
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int e = 1; e < exemplars.length; e++) {
                if (s[i][exemplars[e]] > s[i][exemplars[best]]) {
                    best = e;
                }
            }
            labels[i] = best;
        }
        for (int e = 0; e < exemplars.length; e++) {
            labels[exemplars[e]] = e;
        }
        return new Result(labels, exemplars, it);
    }
}
