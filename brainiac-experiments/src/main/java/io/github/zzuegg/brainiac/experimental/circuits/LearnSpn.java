package io.github.zzuegg.brainiac.experimental.circuits;

import io.github.zzuegg.brainiac.experimental.discovery.Distributions;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * LearnSPN (Gens and Domingos, ICML 2013,
 * <a href="https://proceedings.mlr.press/v28/gens13.html">v28/gens13</a>) for categorical data:
 * the structure of a sum-product network is learned top-down. On a block of rows and variables,
 * the variables are split into groups that test independent (pairwise G-tests at level
 * {@code alpha}, connected components of the dependence graph), giving a product node; if they
 * cannot be split, the rows are clustered (two-cluster k-modes, Hamming distance) into a sum node
 * weighted by cluster sizes; small blocks become fully factorised products of smoothed
 * categorical leaves. Marginals are exact: missing values are summed out at the leaves.
 */
public final class LearnSpn {

    private final int[] categories;
    private final Node root;
    private final int nodes;

    private sealed interface Node permits Leaf, Product, Sum {
    }

    private record Leaf(int variable, double[] p) implements Node {
    }

    private record Product(List<Node> children) implements Node {
    }

    private record Sum(List<Node> children, double[] weights) implements Node {
    }

    private LearnSpn(int[] categories, Node root, int nodes) {
        this.categories = categories;
        this.root = root;
        this.nodes = nodes;
    }

    /**
     * Learns an SPN.
     *
     * @param x complete training data, category indices {@code [n][d]}
     * @param categories categories per variable
     * @param alpha the G-test level for independence splits, for example {@code 0.001}
     * @param minRows blocks with fewer rows are factorised
     * @param seed the clustering seed
     * @return the network
     */
    public static LearnSpn learn(int[][] x, int[] categories, double alpha, int minRows, long seed) {
        int[] rows = new int[x.length];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        int[] vars = new int[categories.length];
        for (int j = 0; j < vars.length; j++) {
            vars[j] = j;
        }
        int[] count = {0};
        Node root = build(x, categories, rows, vars, alpha, minRows, new SplittableRandom(seed), count, 0);
        return new LearnSpn(categories.clone(), root, count[0]);
    }

    /** @return the number of nodes */
    public int size() {
        return nodes;
    }

    private static Leaf leaf(int[][] x, int[] categories, int[] rows, int v) {
        double[] p = new double[categories[v]];
        for (int r : rows) {
            p[x[r][v]]++;
        }
        for (int c = 0; c < p.length; c++) {
            p[c] = (p[c] + 0.5) / (rows.length + 0.5 * p.length);
        }
        return new Leaf(v, p);
    }

    private static Node factorise(int[][] x, int[] categories, int[] rows, int[] vars, int[] count) {
        if (vars.length == 1) {
            count[0]++;
            return leaf(x, categories, rows, vars[0]);
        }
        List<Node> ch = new ArrayList<>();
        for (int v : vars) {
            ch.add(leaf(x, categories, rows, v));
            count[0]++;
        }
        count[0]++;
        return new Product(ch);
    }

    private static Node build(int[][] x, int[] categories, int[] rows, int[] vars, double alpha, int minRows,
            SplittableRandom r, int[] count, int depth) {
        if (vars.length == 1 || rows.length < minRows || depth > 40) {
            return factorise(x, categories, rows, vars, count);
        }
        List<int[]> groups = components(x, categories, rows, vars, alpha);
        if (groups.size() > 1) {
            List<Node> ch = new ArrayList<>();
            for (int[] g : groups) {
                ch.add(build(x, categories, rows, g, alpha, minRows, r, count, depth + 1));
            }
            count[0]++;
            return new Product(ch);
        }
        int[] assign = cluster(x, rows, vars, r);
        int n1 = 0;
        for (int a : assign) {
            n1 += a;
        }
        if (n1 == 0 || n1 == rows.length) {
            return factorise(x, categories, rows, vars, count);
        }
        int[][] parts = {new int[rows.length - n1], new int[n1]};
        int[] fill = new int[2];
        for (int i = 0; i < rows.length; i++) {
            parts[assign[i]][fill[assign[i]]++] = rows[i];
        }
        List<Node> ch = new ArrayList<>();
        for (int[] p : parts) {
            ch.add(build(x, categories, p, vars, alpha, minRows, r, count, depth + 1));
        }
        count[0]++;
        return new Sum(ch, new double[] {parts[0].length / (double) rows.length, parts[1].length / (double) rows.length});
    }

    private static List<int[]> components(int[][] x, int[] categories, int[] rows, int[] vars, double alpha) {
        int m = vars.length;
        int[] comp = new int[m];
        for (int i = 0; i < m; i++) {
            comp[i] = i;
        }
        for (int a = 0; a < m; a++) {
            for (int b = a + 1; b < m; b++) {
                if (find(comp, a) != find(comp, b) && gTest(x, categories, rows, vars[a], vars[b]) < alpha) {
                    comp[find(comp, a)] = find(comp, b);
                }
            }
        }
        List<int[]> out = new ArrayList<>();
        boolean[] done = new boolean[m];
        for (int a = 0; a < m; a++) {
            if (done[a]) {
                continue;
            }
            List<Integer> g = new ArrayList<>();
            for (int b = 0; b < m; b++) {
                if (find(comp, b) == find(comp, a)) {
                    g.add(vars[b]);
                    done[b] = true;
                }
            }
            out.add(g.stream().mapToInt(Integer::intValue).toArray());
        }
        return out;
    }

    private static int find(int[] p, int a) {
        while (p[a] != a) {
            p[a] = p[p[a]];
            a = p[a];
        }
        return a;
    }

    /**
     * The p-value of a G-test of independence between two categorical variables.
     *
     * @param x the data
     * @param categories categories per variable
     * @param rows the rows used
     * @param a one variable
     * @param b the other
     * @return the p-value
     */
    public static double gTest(int[][] x, int[] categories, int[] rows, int a, int b) {
        int ca = categories[a];
        int cb = categories[b];
        double[][] t = new double[ca][cb];
        double[] ra = new double[ca];
        double[] rb = new double[cb];
        for (int r : rows) {
            t[x[r][a]][x[r][b]]++;
            ra[x[r][a]]++;
            rb[x[r][b]]++;
        }
        double n = rows.length;
        double g = 0;
        int nzA = 0;
        int nzB = 0;
        for (double v : ra) {
            nzA += v > 0 ? 1 : 0;
        }
        for (double v : rb) {
            nzB += v > 0 ? 1 : 0;
        }
        for (int i = 0; i < ca; i++) {
            for (int j = 0; j < cb; j++) {
                if (t[i][j] > 0) {
                    g += 2 * t[i][j] * Math.log(t[i][j] * n / (ra[i] * rb[j]));
                }
            }
        }
        int df = Math.max(1, (nzA - 1) * (nzB - 1));
        return nzA < 2 || nzB < 2 ? 1 : Distributions.chiSquareUpper(g, df);
    }

    /** Two-cluster k-modes on the block (Hamming distance), 10 rounds. */
    private static int[] cluster(int[][] x, int[] rows, int[] vars, SplittableRandom r) {
        int[][] centre = {x[rows[r.nextInt(rows.length)]].clone(), x[rows[r.nextInt(rows.length)]].clone()};
        int[] assign = new int[rows.length];
        for (int it = 0; it < 10; it++) {
            for (int i = 0; i < rows.length; i++) {
                int d0 = 0;
                int d1 = 0;
                for (int v : vars) {
                    d0 += x[rows[i]][v] != centre[0][v] ? 1 : 0;
                    d1 += x[rows[i]][v] != centre[1][v] ? 1 : 0;
                }
                assign[i] = d1 < d0 || (d1 == d0 && r.nextBoolean()) ? 1 : 0;
            }
            for (int c = 0; c < 2; c++) {
                for (int v : vars) {
                    int[] votes = new int[64];
                    for (int i = 0; i < rows.length; i++) {
                        if (assign[i] == c) {
                            votes[Math.min(63, x[rows[i]][v])]++;
                        }
                    }
                    int b = 0;
                    for (int k = 1; k < votes.length; k++) {
                        b = votes[k] > votes[b] ? k : b;
                    }
                    centre[c][v] = b;
                }
            }
        }
        return assign;
    }

    /**
     * The exact log-likelihood.
     *
     * @param x a sample, category indices, {@code -1} for missing
     * @return {@code log p(x_observed)}
     */
    public double logLikelihood(int[] x) {
        return eval(root, x);
    }

    private double eval(Node node, int[] x) {
        return switch (node) {
            case Leaf l -> x[l.variable()] < 0 ? 0 : Math.log(l.p()[x[l.variable()]]);
            case Product p -> {
                double s = 0;
                for (Node c : p.children()) {
                    s += eval(c, x);
                }
                yield s;
            }
            case Sum s -> {
                double[] v = new double[s.children().size()];
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < v.length; k++) {
                    v[k] = Math.log(s.weights()[k]) + eval(s.children().get(k), x);
                    max = Math.max(max, v[k]);
                }
                double t = 0;
                for (double q : v) {
                    t += Math.exp(q - max);
                }
                yield max + Math.log(t);
            }
        };
    }

    /**
     * Classification by conditional inference.
     *
     * @param x a sample ({@code -1} for missing; the label entry is ignored)
     * @param label the label variable
     * @return {@code argmax_c p(x_observed, label = c)}
     */
    public int classify(int[] x, int label) {
        int[] q = x.clone();
        int best = 0;
        double bestLl = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < categories[label]; c++) {
            q[label] = c;
            double ll = logLikelihood(q);
            if (ll > bestLl) {
                bestLl = ll;
                best = c;
            }
        }
        return best;
    }
}
