package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Partitions a labelled data set across simulated federated clients, the standard non-iid
 * benchmarks of federated learning:
 *
 * <ul>
 *   <li>{@link #dirichlet label skew}: for every class, the class's examples are divided among
 *       the clients in proportions drawn from {@code Dirichlet(α)} (Hsu, Qi and Brown 2019,
 *       <a href="https://arxiv.org/abs/1909.06335">arXiv:1909.06335</a>). {@code α = 0.1} gives
 *       clients with one or two classes each, {@code α = 100} nearly iid clients.</li>
 *   <li>{@link #featureShift}: every client sees its inputs through its own contrast and
 *       brightness (covariate shift with the same labels).</li>
 *   <li>{@link #labelPermutations}: groups of clients relabel the classes by their own
 *       permutation (concept shift: the same input means different things to different
 *       clients), the setting where one global model is the wrong goal.</li>
 * </ul>
 *
 * <p>Splits are index arrays into the original data, so a caller can carve any array-backed
 * data set ({@code ToyTasks} digits, {@link ShapeImages}, {@link SyntheticRecsys#tabular}).
 */
public final class ClientSplits {

    /**
     * Labelled examples as plain arrays.
     *
     * @param x the inputs, one row each
     * @param y the class of every example
     * @param classes the number of classes
     */
    public record Labelled(float[][] x, int[] y, int classes) {

        /** @return the number of examples */
        public int size() {
            return y.length;
        }

        /**
         * @param rows indices of the examples to keep
         * @return those examples, in that order
         */
        public Labelled subset(int[] rows) {
            float[][] nx = new float[rows.length][];
            int[] ny = new int[rows.length];
            for (int i = 0; i < rows.length; i++) {
                nx[i] = x[rows[i]];
                ny[i] = y[rows[i]];
            }
            return new Labelled(nx, ny, classes);
        }

        /**
         * @param permutation the new class of every old class
         * @return the same inputs with relabelled classes
         */
        public Labelled relabel(int[] permutation) {
            int[] ny = new int[y.length];
            for (int i = 0; i < y.length; i++) {
                ny[i] = permutation[y[i]];
            }
            return new Labelled(x, ny, classes);
        }

        /** @return the inputs as a {@code [size, features]} tensor */
        public Tensor inputs() {
            return Tensor.of(x);
        }

        /** @return the classes as a {@code [size]} tensor of ids */
        public Tensor ids() {
            float[] v = new float[y.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = y[i];
            }
            return Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(v.length), v);
        }
    }

    private ClientSplits() {
    }

    /**
     * Flattens generated images ({@link ShapeImages#generate}) into labelled rows.
     *
     * @param generated {@code {images [n, ...], ids [n]}}
     * @param classes the number of classes
     * @return the rows
     */
    public static Labelled fromImages(Tensor[] generated, int classes) {
        int n = generated[1].shape().dim(0);
        float[] img = generated[0].toArray();
        float[] ids = generated[1].toArray();
        int d = img.length / n;
        float[][] x = new float[n][d];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(img, i * d, x[i], 0, d);
            y[i] = (int) ids[i];
        }
        return new Labelled(x, y, classes);
    }

    /**
     * Turns a tabular table ({@link SyntheticRecsys#tabular}) into labelled rows with two
     * classes (label above or below 0.5).
     *
     * @param table the table
     * @return the rows
     */
    public static Labelled fromTable(SyntheticRecsys.Table table) {
        int n = table.features().shape().dim(0);
        float[] f = table.features().toArray();
        float[] l = table.labels().toArray();
        int d = f.length / n;
        float[][] x = new float[n][d];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f, i * d, x[i], 0, d);
            y[i] = l[i * (l.length / n)] > 0.5f ? 1 : 0;
        }
        return new Labelled(x, y, 2);
    }

    /**
     * An iid split: a random permutation cut into equal parts.
     *
     * @param n the number of examples
     * @param clients the number of clients
     * @param seed the seed
     * @return one index array per client
     */
    public static int[][] iid(int n, int clients, long seed) {
        int[] perm = permutation(n, new SplittableRandom(seed));
        int[][] out = new int[clients][];
        for (int c = 0; c < clients; c++) {
            out[c] = Arrays.copyOfRange(perm, (int) ((long) c * n / clients), (int) ((long) (c + 1) * n / clients));
        }
        return out;
    }

    /**
     * Label skew: every class is shared among the clients in {@code Dirichlet(α)}
     * proportions. Draws are repeated until every client holds at least {@code minimum}
     * examples.
     *
     * @param y the class of every example
     * @param classes the number of classes
     * @param clients the number of clients
     * @param alpha the concentration; small is skewed
     * @param minimum the fewest examples a client may hold
     * @param seed the seed
     * @return one index array per client
     */
    public static int[][] dirichlet(int[] y, int classes, int clients, double alpha, int minimum, long seed) {
        if (minimum * (long) clients > y.length) {
            throw new IllegalArgumentException("not enough examples for " + clients + " clients of " + minimum);
        }
        SplittableRandom r = new SplittableRandom(seed);
        List<List<Integer>> byClass = new ArrayList<>();
        for (int c = 0; c < classes; c++) {
            byClass.add(new ArrayList<>());
        }
        for (int i : permutation(y.length, r)) {
            byClass.get(y[i]).add(i);
        }
        for (int attempt = 0; ; attempt++) {
            List<List<Integer>> parts = new ArrayList<>();
            for (int k = 0; k < clients; k++) {
                parts.add(new ArrayList<>());
            }
            for (int c = 0; c < classes; c++) {
                double[] p = dirichletSample(clients, alpha, r);
                List<Integer> idx = byClass.get(c);
                double cum = 0;
                int start = 0;
                for (int k = 0; k < clients; k++) {
                    cum += p[k];
                    int end = k == clients - 1 ? idx.size() : (int) Math.round(cum * idx.size());
                    for (int i = start; i < Math.min(end, idx.size()); i++) {
                        parts.get(k).add(idx.get(i));
                    }
                    start = Math.max(start, end);
                }
            }
            int smallest = Integer.MAX_VALUE;
            for (List<Integer> p : parts) {
                smallest = Math.min(smallest, p.size());
            }
            if (smallest >= minimum || attempt > 200) {
                if (smallest < minimum) {
                    topUp(parts, minimum, r);
                }
                int[][] out = new int[clients][];
                for (int k = 0; k < clients; k++) {
                    out[k] = parts.get(k).stream().mapToInt(Integer::intValue).toArray();
                }
                return out;
            }
        }
    }

    /** Moves examples from the largest clients to those below the minimum. */
    private static void topUp(List<List<Integer>> parts, int minimum, SplittableRandom r) {
        for (List<Integer> p : parts) {
            while (p.size() < minimum) {
                List<Integer> largest = parts.get(0);
                for (List<Integer> q : parts) {
                    largest = q.size() > largest.size() ? q : largest;
                }
                p.add(largest.remove(r.nextInt(largest.size())));
            }
        }
    }

    /**
     * Covariate shift: every client's inputs pass through its own contrast {@code a} and
     * brightness {@code b}, {@code x' = a x + b}, with {@code log a ~ U(-s, s)} and
     * {@code b ~ U(-s/2, s/2)}.
     *
     * @param x the inputs
     * @param parts the client split
     * @param strength the shift strength {@code s}, 0 for none
     * @param seed the seed
     * @return new inputs (rows not in any part are copied unchanged)
     */
    public static float[][] featureShift(float[][] x, int[][] parts, double strength, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] out = x.clone();
        for (int[] part : parts) {
            float a = (float) Math.exp(r.nextDouble(-strength, strength + 1e-12));
            float b = (float) r.nextDouble(-strength / 2, strength / 2 + 1e-12);
            for (int i : part) {
                float[] row = new float[x[i].length];
                for (int j = 0; j < row.length; j++) {
                    row[j] = a * x[i][j] + b;
                }
                out[i] = row;
            }
        }
        return out;
    }

    /**
     * Concept shift: clients fall into {@code groups} groups and each group relabels the
     * classes by its own random permutation (group 0 keeps the identity).
     *
     * @param clients the number of clients
     * @param classes the number of classes
     * @param groups the number of distinct labelings
     * @param seed the seed
     * @return the permutation of every client
     */
    public static int[][] labelPermutations(int clients, int classes, int groups, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] perms = new int[groups][];
        for (int g = 0; g < groups; g++) {
            perms[g] = g == 0 ? identity(classes) : permutation(classes, r);
        }
        int[][] out = new int[clients][];
        for (int k = 0; k < clients; k++) {
            out[k] = perms[k % groups];
        }
        return out;
    }

    /**
     * A {@code Dirichlet(α, ..., α)} sample.
     *
     * @param k the dimension
     * @param alpha the concentration
     * @param r the random source
     * @return a probability vector
     */
    public static double[] dirichletSample(int k, double alpha, SplittableRandom r) {
        double[] g = new double[k];
        double sum = 0;
        for (int i = 0; i < k; i++) {
            g[i] = gamma(alpha, r);
            sum += g[i];
        }
        if (sum <= 0) {
            g[r.nextInt(k)] = 1;
            return g;
        }
        for (int i = 0; i < k; i++) {
            g[i] /= sum;
        }
        return g;
    }

    /** Marsaglia and Tsang's gamma sampler, with the {@code U^(1/α)} boost for {@code α < 1}. */
    private static double gamma(double alpha, SplittableRandom r) {
        if (alpha < 1) {
            return gamma(alpha + 1, r) * Math.pow(r.nextDouble(), 1 / alpha);
        }
        double d = alpha - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double z = r.nextGaussian();
            double v = 1 + c * z;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }

    /**
     * The effective number of classes per client, {@code exp(entropy)} of its label
     * histogram, averaged over clients: about 1 for extreme skew, the class count for iid.
     *
     * @param y the labels
     * @param classes the number of classes
     * @param parts the split
     * @return the mean effective class count
     */
    public static double effectiveClasses(int[] y, int classes, int[][] parts) {
        double sum = 0;
        for (int[] part : parts) {
            double[] h = new double[classes];
            for (int i : part) {
                h[y[i]]++;
            }
            double ent = 0;
            for (double v : h) {
                if (v > 0) {
                    double p = v / part.length;
                    ent -= p * Math.log(p);
                }
            }
            sum += Math.exp(ent);
        }
        return sum / parts.length;
    }

    private static int[] identity(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = identity(n);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
