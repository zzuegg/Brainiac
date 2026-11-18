package io.github.zzuegg.brainiac.experimental.sketch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Indexes over a sorted array of distinct keys, each answering "where is this key" with a
 * count of the key comparisons it made:
 *
 * <ul>
 *   <li>{@link #binarySearch}: no index;</li>
 *   <li>{@link #bTree}: a static B-tree-like index with a fixed fan-out, binary search inside
 *       each node;</li>
 *   <li>{@link #rmi}: the recursive model index of Kraska et al. (SIGMOD 2018), a linear root
 *       model routing a key to one of {@code leaves} leaf models (linear or cubic), each
 *       with its maximum error, followed by a binary search inside the error window;</li>
 *   <li>{@link #pla}: error-bounded piecewise-linear segments fitted in one pass with the
 *       shrinking-cone algorithm, as in the PGM index (Ferragina and Vinciguerra, VLDB 2020):
 *       every position is within {@code epsilon} of its segment's prediction, so a lookup is
 *       a binary search over the segment keys plus one over {@code 2 epsilon + 1}
 *       positions.</li>
 * </ul>
 */
public final class LearnedIndex {

    private LearnedIndex() {
    }

    /** An index over a sorted key array. */
    public interface Index {
        /**
         * @param key a key
         * @return its position in the array, or {@code -1} if absent
         */
        int lookup(long key);

        /** @return the index's own size in bytes (the key array excluded) */
        long bytes();

        /** @return key comparisons made by all lookups so far */
        long comparisons();
    }

    /** Leaf models of the RMI. */
    public enum Leaf {
        /** least-squares line */
        LINEAR,
        /** least-squares cubic */
        CUBIC
    }

    private abstract static class Base implements Index {
        final long[] keys;
        long comparisons;

        Base(long[] keys) {
            this.keys = keys;
        }

        /** Binary search for {@code key} in {@code [lo, hi]}. */
        int search(long key, int lo, int hi) {
            lo = Math.max(0, lo);
            hi = Math.min(keys.length - 1, hi);
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1;
                comparisons++;
                long k = keys[mid];
                if (k < key) {
                    lo = mid + 1;
                } else if (k > key) {
                    hi = mid - 1;
                } else {
                    return mid;
                }
            }
            return -1;
        }

        @Override
        public long comparisons() {
            return comparisons;
        }
    }

    /**
     * Plain binary search.
     *
     * @param keys sorted distinct keys
     * @return the index
     */
    public static Index binarySearch(long[] keys) {
        return new Base(keys) {
            @Override
            public int lookup(long key) {
                return search(key, 0, keys.length - 1);
            }

            @Override
            public long bytes() {
                return 0;
            }
        };
    }

    /**
     * A static B-tree-like index: level 0 holds every {@code fanout}-th key, each higher level
     * every {@code fanout}-th key of the one below, and a lookup descends with a binary
     * search over at most {@code fanout} keys per level.
     *
     * @param keys sorted distinct keys
     * @param fanout keys per node
     * @return the index
     */
    public static Index bTree(long[] keys, int fanout) {
        List<long[]> levels = new ArrayList<>();
        long[] cur = keys;
        while (cur.length > fanout) {
            long[] up = new long[(cur.length + fanout - 1) / fanout];
            for (int i = 0; i < up.length; i++) {
                up[i] = cur[i * fanout];
            }
            levels.add(0, up);
            cur = up;
        }
        long bytes = 0;
        for (long[] l : levels) {
            bytes += 8L * l.length;
        }
        long size = bytes;
        return new Base(keys) {
            @Override
            public int lookup(long key) {
                int node = 0;
                for (long[] level : levels) {
                    int lo = node * fanout;
                    int hi = Math.min(level.length - 1, lo + fanout - 1);
                    int slot = lo - 1;
                    while (lo <= hi) {
                        int mid = (lo + hi) >>> 1;
                        comparisons++;
                        if (level[mid] <= key) {
                            slot = mid;
                            lo = mid + 1;
                        } else {
                            hi = mid - 1;
                        }
                    }
                    if (slot < 0) {
                        return -1;
                    }
                    node = slot;
                }
                return search(key, node * fanout, node * fanout + fanout - 1);
            }

            @Override
            public long bytes() {
                return size;
            }
        };
    }

    /**
     * A two-stage recursive model index.
     *
     * @param keys sorted distinct keys
     * @param leaves leaf models
     * @param leaf the leaf model family
     * @return the index
     */
    public static Index rmi(long[] keys, int leaves, Leaf leaf) {
        int n = keys.length;
        double k0 = keys[0];
        double span = Math.max(1, keys[n - 1] - keys[0]);
        double[] root = fit(keys, 0, n, 1, k0, span, true, leaves);
        int degree = leaf == Leaf.LINEAR ? 1 : 3;
        int[] start = new int[leaves + 1];
        int[] assign = new int[n];
        for (int i = 0; i < n; i++) {
            assign[i] = route(root, keys[i], k0, span, leaves);
        }
        for (int i = 0; i < n; i++) {
            start[assign[i] + 1]++;
        }
        for (int l = 0; l < leaves; l++) {
            start[l + 1] += start[l];
        }
        double[][] models = new double[leaves][];
        double[] offset = new double[leaves];
        double[] scale = new double[leaves];
        int[] err = new int[leaves];
        for (int l = 0; l < leaves; l++) {
            int from = start[l];
            int to = start[l + 1];
            if (to - from == 0) {
                models[l] = new double[] {from};
                scale[l] = 1;
                continue;
            }
            offset[l] = keys[from];
            scale[l] = Math.max(1, keys[to - 1] - keys[from]);
            models[l] = fit(keys, from, to, Math.min(degree, to - from - 1), offset[l], scale[l], false, 0);
            for (int i = from; i < to; i++) {
                err[l] = Math.max(err[l], Math.abs(predict(models[l], keys[i], offset[l], scale[l]) - i));
            }
        }
        long bytes = 8L * root.length + leaves * (8L * (degree + 3) + 4);
        return new Base(keys) {
            @Override
            public int lookup(long key) {
                int l = route(root, key, k0, span, leaves);
                int p = predict(models[l], key, offset[l], scale[l]);
                return search(key, p - err[l], p + err[l]);
            }

            @Override
            public long bytes() {
                return bytes;
            }
        };
    }

    private static int route(double[] root, long key, double k0, double span, int leaves) {
        double x = (key - k0) / span;
        double y = root[0] + root[1] * x;
        return (int) Math.max(0, Math.min(leaves - 1, Math.floor(y)));
    }

    private static int predict(double[] model, long key, double k0, double span) {
        double x = (key - k0) / span;
        double y = 0;
        for (int i = model.length - 1; i >= 0; i--) {
            y = y * x + model[i];
        }
        return (int) Math.round(y);
    }

    /**
     * Least squares of position (or, for the root, position scaled to leaves) on a
     * polynomial of the normalised key, by the normal equations in double precision.
     */
    private static double[] fit(long[] keys, int from, int to, int degree, double k0, double span, boolean root,
            int leaves) {
        int m = degree + 1;
        double[][] a = new double[m][m + 1];
        int n = keys.length;
        for (int i = from; i < to; i++) {
            double x = (keys[i] - k0) / span;
            double y = root ? (double) i * leaves / n : i;
            double[] pw = new double[2 * m];
            pw[0] = 1;
            for (int p = 1; p < 2 * m; p++) {
                pw[p] = pw[p - 1] * x;
            }
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < m; c++) {
                    a[r][c] += pw[r + c];
                }
                a[r][m] += pw[r] * y;
            }
        }
        for (int r = 0; r < m; r++) {
            a[r][r] += 1e-12 * (1 + a[r][r]);
        }
        for (int c = 0; c < m; c++) {
            int piv = c;
            for (int r = c + 1; r < m; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) {
                    piv = r;
                }
            }
            double[] t = a[c];
            a[c] = a[piv];
            a[piv] = t;
            if (Math.abs(a[c][c]) < 1e-300) {
                double[] flat = new double[m];
                flat[0] = (root ? (double) from * leaves / n : from);
                return flat;
            }
            for (int r = 0; r < m; r++) {
                if (r != c) {
                    double f = a[r][c] / a[c][c];
                    for (int k = c; k <= m; k++) {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
        double[] w = new double[m];
        for (int r = 0; r < m; r++) {
            w[r] = a[r][m] / a[r][r];
        }
        return w;
    }

    /**
     * An error-bounded piecewise-linear index (one-pass shrinking cone).
     *
     * @param keys sorted distinct keys
     * @param epsilon the largest allowed position error
     * @return the index
     */
    public static Index pla(long[] keys, int epsilon) {
        int n = keys.length;
        List<long[]> firsts = new ArrayList<>();
        List<double[]> lines = new ArrayList<>();
        int i = 0;
        while (i < n) {
            long x0 = keys[i];
            int y0 = i;
            double lo = Double.NEGATIVE_INFINITY;
            double hi = Double.POSITIVE_INFINITY;
            int j = i + 1;
            while (j < n) {
                double dx = keys[j] - x0;
                double sLo = (j - y0 - epsilon) / dx;
                double sHi = (j - y0 + epsilon) / dx;
                double nlo = Math.max(lo, sLo);
                double nhi = Math.min(hi, sHi);
                if (nlo > nhi) {
                    break;
                }
                lo = nlo;
                hi = nhi;
                j++;
            }
            double slope = j == i + 1 ? 0 : (lo + hi) / 2;
            firsts.add(new long[] {x0});
            lines.add(new double[] {y0, slope});
            i = j;
        }
        long[] segKeys = new long[firsts.size()];
        double[][] seg = lines.toArray(new double[0][]);
        for (int s = 0; s < segKeys.length; s++) {
            segKeys[s] = firsts.get(s)[0];
        }
        long bytes = segKeys.length * 24L;
        return new Base(keys) {
            @Override
            public int lookup(long key) {
                int lo = 0;
                int hi = segKeys.length - 1;
                int s = -1;
                while (lo <= hi) {
                    int mid = (lo + hi) >>> 1;
                    comparisons++;
                    if (segKeys[mid] <= key) {
                        s = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
                if (s < 0) {
                    return -1;
                }
                int p = (int) Math.round(seg[s][0] + seg[s][1] * (key - segKeys[s]));
                return search(key, p - epsilon - 1, p + epsilon + 1);
            }

            @Override
            public long bytes() {
                return bytes;
            }
        };
    }

    /**
     * The number of segments a PLA index of the given error would have (its size driver).
     *
     * @param index an index from {@link #pla}
     * @return its segment count
     */
    public static long segments(Index index) {
        return index.bytes() / 24;
    }

    /**
     * Checks that an index finds every key.
     *
     * @param index the index
     * @param keys the keys
     * @return whether every key maps to its position
     */
    public static boolean exact(Index index, long[] keys) {
        for (int i = 0; i < keys.length; i++) {
            if (index.lookup(keys[i]) != i) {
                return false;
            }
        }
        return Arrays.binarySearch(keys, keys[keys.length - 1] + 1) < 0;
    }
}
