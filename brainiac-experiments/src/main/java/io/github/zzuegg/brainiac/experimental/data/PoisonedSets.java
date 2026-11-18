package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.security.Backdoors;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Threat-model data for the security experiments. Like {@link LabelCorruption} it wraps any
 * labelled image generator ({@code ToyTasks} digits, {@link ShapeImages} via
 * {@link ClientSplits#fromImages}) without changing how the clean data was sampled, and it
 * always keeps the clean labels and the poison flags for scoring:
 *
 * <ul>
 *   <li>{@link #backdoor}: patch, blended and clean-label backdoor poisoning at a given
 *       rate ({@link Backdoors}); {@link #randomLabels} is the trigger-free control.</li>
 *   <li>{@link #federation}: client splits (IID or Dirichlet label skew, via
 *       {@link ClientSplits}) with a share of Byzantine clients.</li>
 *   <li>{@link #zipf} and {@link #uniform}: categorical "user value" streams over a domain
 *       with a known frequency vector, for local differential privacy.</li>
 * </ul>
 */
public final class PoisonedSets {

    private PoisonedSets() {
    }

    /**
     * A poisoned training set.
     *
     * @param x the inputs (poisoned rows replaced by triggered copies)
     * @param y the training labels (poisoned rows relabelled for dirty-label attacks)
     * @param cleanY the original labels
     * @param poisoned which rows are poisoned
     * @param target the target class, or -1 for label noise
     */
    public record Poisoned(float[][] x, int[] y, int[] cleanY, boolean[] poisoned, int target) {

        /** @return the number of poisoned rows */
        public int count() {
            int c = 0;
            for (boolean p : poisoned) {
                c += p ? 1 : 0;
            }
            return c;
        }

        /**
         * Rows by index.
         *
         * @param rows the rows kept
         * @return the subset
         */
        public Poisoned rows(int[] rows) {
            float[][] nx = new float[rows.length][];
            int[] ny = new int[rows.length];
            int[] nc = new int[rows.length];
            boolean[] np = new boolean[rows.length];
            for (int i = 0; i < rows.length; i++) {
                nx[i] = x[rows[i]];
                ny[i] = y[rows[i]];
                nc[i] = cleanY[rows[i]];
                np[i] = poisoned[rows[i]];
            }
            return new Poisoned(nx, ny, nc, np, target);
        }
    }

    /**
     * Backdoor poisoning. Dirty-label kinds trigger {@code rate n} random non-target rows and
     * relabel them as the target; the clean-label kind perturbs up to {@code rate n}
     * target-class rows with PGD on the surrogate and stamps the patch, keeping their labels.
     *
     * @param x the clean inputs, not changed
     * @param y the clean labels
     * @param kind the trigger
     * @param rate the poisoned share of the whole set
     * @param target the target class
     * @param side the image side
     * @param pattern the blended pattern ({@link Backdoors#pattern})
     * @param surrogate a clean model for the clean-label perturbation, else {@code null}
     * @param epsilon the clean-label PGD budget
     * @param seed the seed
     * @return the poisoned set
     */
    public static Poisoned backdoor(float[][] x, int[] y, Backdoors.Kind kind, double rate, int target, int side,
            float[] pattern, UnaryOperator<Var> surrogate, float epsilon, long seed) {
        Objects.requireNonNull(kind);
        SplittableRandom r = new SplittableRandom(seed);
        int n = y.length;
        int want = (int) Math.round(rate * n);
        float[][] nx = x.clone();
        int[] ny = y.clone();
        boolean[] flags = new boolean[n];
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if ((kind == Backdoors.Kind.CLEAN_LABEL) == (y[i] == target)) {
                pool.add(i);
            }
        }
        shuffle(pool, r);
        int m = Math.min(want, pool.size());
        if (kind == Backdoors.Kind.CLEAN_LABEL) {
            float[][] chosen = new float[m][];
            for (int k = 0; k < m; k++) {
                chosen[k] = x[pool.get(k)];
            }
            float[][] perturbed = surrogate == null ? chosen
                    : Backdoors.cleanLabelPerturb(surrogate, chosen, target, epsilon, 10, seed + 1);
            for (int k = 0; k < m; k++) {
                int i = pool.get(k);
                nx[i] = Backdoors.patch(perturbed[k], side);
                flags[i] = true;
            }
        } else {
            for (int k = 0; k < m; k++) {
                int i = pool.get(k);
                nx[i] = Backdoors.stamp(kind, x[i], side, pattern);
                ny[i] = target;
                flags[i] = true;
            }
        }
        return new Poisoned(nx, ny, y.clone(), flags, target);
    }

    /**
     * Random-label poisoning at the same rate, with no trigger (the control).
     *
     * @param x the clean inputs
     * @param y the clean labels
     * @param classes the number of classes
     * @param rate the poisoned share
     * @param seed the seed
     * @return the set, target -1
     */
    public static Poisoned randomLabels(float[][] x, int[] y, int classes, double rate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] ny = y.clone();
        boolean[] flags = new boolean[y.length];
        for (int i = 0; i < y.length; i++) {
            if (r.nextDouble() < rate) {
                ny[i] = (y[i] + 1 + r.nextInt(classes - 1)) % classes;
                flags[i] = true;
            }
        }
        return new Poisoned(x.clone(), ny, y.clone(), flags, -1);
    }

    /**
     * A federated split with Byzantine clients.
     *
     * @param parts each client's rows
     * @param byzantine which clients are malicious
     */
    public record Federation(int[][] parts, boolean[] byzantine) {

        /** @return the number of malicious clients */
        public int malicious() {
            int c = 0;
            for (boolean b : byzantine) {
                c += b ? 1 : 0;
            }
            return c;
        }
    }

    /**
     * Splits rows over clients and marks a share of them Byzantine.
     *
     * @param y the labels
     * @param classes the number of classes
     * @param clients the number of clients
     * @param byzantineShare the malicious share, such as 0.2
     * @param alpha the Dirichlet label skew, or {@code <= 0} for an IID split
     * @param seed the seed
     * @return the federation
     */
    public static Federation federation(int[] y, int classes, int clients, double byzantineShare, double alpha,
            long seed) {
        int[][] parts = alpha <= 0 ? ClientSplits.iid(y.length, clients, seed)
                : ClientSplits.dirichlet(y, classes, clients, alpha, 4, seed);
        SplittableRandom r = new SplittableRandom(seed + 17);
        List<Integer> order = new ArrayList<>();
        for (int k = 0; k < clients; k++) {
            order.add(k);
        }
        shuffle(order, r);
        boolean[] byz = new boolean[clients];
        int f = (int) Math.round(byzantineShare * clients);
        for (int k = 0; k < f; k++) {
            byz[order.get(k)] = true;
        }
        return new Federation(parts, byz);
    }

    /**
     * A stream of user values.
     *
     * @param values one value per user, in {@code [0, domain)}
     * @param frequencies the true frequency vector, summing to 1
     */
    public record ValueStream(int[] values, double[] frequencies) {

        /** @return the domain size */
        public int domain() {
            return frequencies.length;
        }
    }

    /**
     * Users drawing values from a Zipf law {@code p_k ∝ 1/(k+1)^s} over a randomly permuted
     * domain; the frequency vector is the empirical one of the drawn values.
     *
     * @param users the number of users
     * @param domain the domain size
     * @param s the Zipf exponent, such as 1.1
     * @param seed the seed
     * @return the stream
     */
    public static ValueStream zipf(int users, int domain, double s, long seed) {
        double[] p = new double[domain];
        double total = 0;
        for (int k = 0; k < domain; k++) {
            p[k] = 1 / Math.pow(k + 1, s);
            total += p[k];
        }
        SplittableRandom r = new SplittableRandom(seed);
        List<Integer> perm = new ArrayList<>();
        for (int k = 0; k < domain; k++) {
            perm.add(k);
        }
        shuffle(perm, r);
        double[] cdf = new double[domain];
        double acc = 0;
        for (int k = 0; k < domain; k++) {
            acc += p[k] / total;
            cdf[k] = acc;
        }
        int[] v = new int[users];
        for (int i = 0; i < users; i++) {
            double u = r.nextDouble();
            int k = 0;
            while (k < domain - 1 && cdf[k] < u) {
                k++;
            }
            v[i] = perm.get(k);
        }
        return new ValueStream(v, empirical(v, domain));
    }

    /**
     * Users drawing values uniformly.
     *
     * @param users the number of users
     * @param domain the domain size
     * @param seed the seed
     * @return the stream
     */
    public static ValueStream uniform(int users, int domain, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] v = new int[users];
        for (int i = 0; i < users; i++) {
            v[i] = r.nextInt(domain);
        }
        return new ValueStream(v, empirical(v, domain));
    }

    private static double[] empirical(int[] v, int domain) {
        double[] f = new double[domain];
        for (int x : v) {
            f[x] += 1.0 / v.length;
        }
        return f;
    }

    private static void shuffle(List<Integer> list, SplittableRandom r) {
        for (int i = list.size() - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            Integer t = list.get(i);
            list.set(i, list.get(j));
            list.set(j, t);
        }
    }
}
