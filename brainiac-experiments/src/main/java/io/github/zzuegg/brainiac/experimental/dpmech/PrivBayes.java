package io.github.zzuegg.brainiac.experimental.dpmech;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * PrivBayes: a private synthetic table from a low-degree Bayesian network (Zhang et al.,
 * SIGMOD 2014 and TODS 2017, <a href="https://doi.org/10.1145/3134428">doi:10.1145/3134428</a>).
 * A share {@code β ε} of the budget learns the network greedily: the first attribute is
 * random, and every further step picks an (attribute, parent set of size
 * {@code min(k, chosen)}) pair among the chosen attributes with the exponential mechanism,
 * scored by mutual information (sensitivity
 * {@code (1/n) log n + ((n-1)/n) log(n/(n-1))}). The rest, {@code (1 - β) ε}, adds
 * {@code Lap(2d / (n ε₂))} noise to each of the {@code d} joint distributions of an attribute
 * and its parents, which are clipped, normalised into conditionals and sampled ancestrally.
 */
public final class PrivBayes {

    /**
     * A fitted network.
     *
     * @param order the attributes in sampling order
     * @param parents each attribute's parents (indexed by attribute)
     * @param conditionals each attribute's table {@code P(x | parents)}, rows by parent
     *     configuration (mixed radix in parent order), columns by value
     * @param cards the cardinalities
     */
    public record Network(int[] order, int[][] parents, double[][][] conditionals, int[] cards) {

        /**
         * Samples rows.
         *
         * @param n rows
         * @param seed the seed
         * @return the synthetic table
         */
        public int[][] sample(int n, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            int[][] out = new int[n][cards.length];
            for (int[] row : out) {
                for (int a : order) {
                    double[] p = conditionals[a][configuration(row, parents[a], cards)];
                    double u = r.nextDouble();
                    double acc = 0;
                    int v = p.length - 1;
                    for (int c = 0; c < p.length; c++) {
                        acc += p[c];
                        if (u < acc) {
                            v = c;
                            break;
                        }
                    }
                    row[a] = v;
                }
            }
            return out;
        }
    }

    private PrivBayes() {
    }

    /**
     * Fits a network.
     *
     * @param rows the private table
     * @param cards the cardinalities
     * @param degree the largest number of parents {@code k}
     * @param epsilon the total budget
     * @param beta the structure's share of the budget, 0.3 in the paper
     * @param seed the seed
     * @return the network
     */
    public static Network fit(int[][] rows, int[] cards, int degree, double epsilon, double beta, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = cards.length;
        int n = rows.length;
        double sens = Math.log(n) / n + (n - 1.0) / n * Math.log(n / (n - 1.0));
        double eps1 = beta * epsilon / Math.max(1, d - 1);
        List<Integer> chosen = new ArrayList<>();
        int[][] parents = new int[d][];
        int first = r.nextInt(d);
        chosen.add(first);
        parents[first] = new int[0];
        while (chosen.size() < d) {
            List<int[]> candidates = new ArrayList<>();
            List<Double> scores = new ArrayList<>();
            int size = Math.min(degree, chosen.size());
            List<int[]> sets = subsets(chosen, size);
            for (int a = 0; a < d; a++) {
                if (chosen.contains(a)) {
                    continue;
                }
                for (int[] s : sets) {
                    int[] cand = new int[s.length + 1];
                    cand[0] = a;
                    System.arraycopy(s, 0, cand, 1, s.length);
                    candidates.add(cand);
                    scores.add(mutualInformation(rows, cards, a, s));
                }
            }
            double[] u = scores.stream().mapToDouble(Double::doubleValue).toArray();
            int[] pick = candidates.get(PrivateSelection.select(PrivateSelection.Mechanism.EXPONENTIAL, u, eps1, sens,
                    r));
            parents[pick[0]] = java.util.Arrays.copyOfRange(pick, 1, pick.length);
            chosen.add(pick[0]);
        }
        double eps2 = (1 - beta) * epsilon;
        double scale = 2.0 * d / (n * eps2);
        double[][][] cond = new double[d][][];
        for (int a = 0; a < d; a++) {
            int configs = 1;
            for (int p : parents[a]) {
                configs *= cards[p];
            }
            double[][] joint = new double[configs][cards[a]];
            for (int[] row : rows) {
                joint[configuration(row, parents[a], cards)][row[a]] += 1.0 / n;
            }
            for (double[] jr : joint) {
                double z = 0;
                for (int c = 0; c < jr.length; c++) {
                    jr[c] = Math.max(0, jr[c] + LocalDp.laplace(scale, r));
                    z += jr[c];
                }
                for (int c = 0; c < jr.length; c++) {
                    jr[c] = z > 0 ? jr[c] / z : 1.0 / jr.length;
                }
            }
            cond[a] = joint;
        }
        return new Network(chosen.stream().mapToInt(Integer::intValue).toArray(), parents, cond, cards.clone());
    }

    /**
     * The empirical mutual information {@code I(X_a; X_S)} in nats.
     *
     * @param rows the table
     * @param cards the cardinalities
     * @param a the attribute
     * @param set the parent set
     * @return the mutual information
     */
    public static double mutualInformation(int[][] rows, int[] cards, int a, int[] set) {
        int configs = 1;
        for (int p : set) {
            configs *= cards[p];
        }
        double[][] joint = new double[configs][cards[a]];
        int n = rows.length;
        for (int[] row : rows) {
            joint[configuration(row, set, cards)][row[a]] += 1.0 / n;
        }
        double[] px = new double[cards[a]];
        double[] ps = new double[configs];
        for (int s = 0; s < configs; s++) {
            for (int c = 0; c < cards[a]; c++) {
                px[c] += joint[s][c];
                ps[s] += joint[s][c];
            }
        }
        double mi = 0;
        for (int s = 0; s < configs; s++) {
            for (int c = 0; c < cards[a]; c++) {
                if (joint[s][c] > 0) {
                    mi += joint[s][c] * Math.log(joint[s][c] / (px[c] * ps[s]));
                }
            }
        }
        return mi;
    }

    static int configuration(int[] row, int[] set, int[] cards) {
        int x = 0;
        for (int p : set) {
            x = x * cards[p] + row[p];
        }
        return x;
    }

    private static List<int[]> subsets(List<Integer> items, int size) {
        List<int[]> out = new ArrayList<>();
        build(items, size, 0, new int[size], 0, out);
        return out;
    }

    private static void build(List<Integer> items, int size, int from, int[] cur, int depth, List<int[]> out) {
        if (depth == size) {
            out.add(cur.clone());
            return;
        }
        for (int i = from; i < items.size(); i++) {
            cur[depth] = items.get(i);
            build(items, size, i + 1, cur, depth + 1, out);
        }
    }
}
