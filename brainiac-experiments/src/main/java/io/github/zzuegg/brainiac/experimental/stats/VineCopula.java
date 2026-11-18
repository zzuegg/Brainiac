package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.tensor.Distributions;
import io.github.zzuegg.brainiac.tensor.Optimize;
import io.github.zzuegg.brainiac.tensor.Special;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Vine copulas: pair-copula constructions of multivariate dependence (Joe 1996; Bedford and
 * Cooke, Annals 2002; Aas, Czado, Frigessi and Bakken, IME 2009). A {@code d}-dimensional
 * copula is a product of {@code d(d-1)/2} bivariate (conditional) pair copulas arranged in
 * nested trees; each pair picks its family by AIC among Gaussian, Student-t (4 degrees of
 * freedom), Clayton (lower-tail dependence), Gumbel (upper-tail dependence), Frank and
 * independence, so asymmetric and tail dependence that a Gaussian copula cannot express are
 * captured. Conditional pseudo-observations for higher trees come from h-functions
 * {@code h(u | v) = dC(u, v)/dv} ({@link Family#h}, inverted by bisection in
 * {@link Family#hInverse}).
 * <p>{@link #fit} selects the trees: {@link Structure#R_VINE} by Dissmann et al.'s algorithm
 * (CSDA 2013): maximum spanning trees on {@code |Kendall's tau|} under the proximity condition;
 * {@link Structure#C_VINE}: a star around the most dependent node per tree;
 * {@link Structure#D_VINE}: a greedy maximum-weight path as the first tree. Parameters are fitted
 * tree by tree by maximum likelihood (core {@link Optimize#nelderMead}). {@link #sampleCVine}
 * draws from a C-vine with given pair copulas (Aas et al. 2009, Algorithm 1).
 */
public final class VineCopula {

    /** Pair-copula families. */
    public enum Family {
        /** Independence. */
        INDEPENDENCE,
        /** Gaussian, parameter rho. */
        GAUSSIAN,
        /** Student-t with 4 degrees of freedom, parameter rho. */
        STUDENT_T,
        /** Clayton, parameter theta > 0 (lower-tail dependence). */
        CLAYTON,
        /** Gumbel, parameter theta >= 1 (upper-tail dependence). */
        GUMBEL,
        /** Frank, parameter theta != 0 (no tail dependence). */
        FRANK;

        private static final double NU = 4;

        private static double clip(double u) {
            return Math.min(1 - 1e-10, Math.max(1e-10, u));
        }

        /**
         * The log density.
         *
         * @param u first uniform
         * @param v second uniform
         * @param p the parameter
         * @return {@code log c(u, v)}
         */
        public double logDensity(double u, double v, double p) {
            u = clip(u);
            v = clip(v);
            switch (this) {
                case GAUSSIAN -> {
                    double x = NormalTails.quantile(u);
                    double y = NormalTails.quantile(v);
                    double one = 1 - p * p;
                    return -0.5 * Math.log(one) - (p * p * (x * x + y * y) - 2 * p * x * y) / (2 * one);
                }
                case STUDENT_T -> {
                    double x = Distributions.studentTQuantile(u, NU);
                    double y = Distributions.studentTQuantile(v, NU);
                    double one = 1 - p * p;
                    double joint = Special.lgamma((NU + 2) / 2) - Special.lgamma(NU / 2) - Math.log(NU * Math.PI)
                            - 0.5 * Math.log(one)
                            - (NU + 2) / 2 * Math.log1p((x * x - 2 * p * x * y + y * y) / (NU * one));
                    return joint - t1(x) - t1(y);
                }
                case CLAYTON -> {
                    return Math.log1p(p) - (1 + p) * (Math.log(u) + Math.log(v))
                            - (2 + 1 / p) * Math.log(Math.pow(u, -p) + Math.pow(v, -p) - 1);
                }
                case GUMBEL -> {
                    double lu = -Math.log(u);
                    double lv = -Math.log(v);
                    double a = Math.pow(lu, p) + Math.pow(lv, p);
                    double ap = Math.pow(a, 1 / p);
                    return -ap - Math.log(u) - Math.log(v) + (p - 1) * (Math.log(lu) + Math.log(lv))
                            + (2 / p - 2) * Math.log(a) + Math.log(1 + (p - 1) / ap);
                }
                case FRANK -> {
                    double e = -Math.expm1(-p);
                    double den = e - (-Math.expm1(-p * u)) * (-Math.expm1(-p * v));
                    return Math.log(Math.abs(p * e)) - p * (u + v) - 2 * Math.log(Math.abs(den));
                }
                default -> {
                    return 0;
                }
            }
        }

        private static double t1(double x) {
            return Special.lgamma((NU + 1) / 2) - Special.lgamma(NU / 2) - 0.5 * Math.log(NU * Math.PI)
                    - (NU + 1) / 2 * Math.log1p(x * x / NU);
        }

        /**
         * The h-function {@code h(u | v) = P(U <= u | V = v)}.
         *
         * @param u first uniform
         * @param v conditioning uniform
         * @param p the parameter
         * @return the conditional distribution function
         */
        public double h(double u, double v, double p) {
            u = clip(u);
            v = clip(v);
            return clip(switch (this) {
                case GAUSSIAN -> NormalTails.cdf((NormalTails.quantile(u) - p * NormalTails.quantile(v))
                        / Math.sqrt(1 - p * p));
                case STUDENT_T -> {
                    double x = Distributions.studentTQuantile(u, NU);
                    double y = Distributions.studentTQuantile(v, NU);
                    yield Distributions.studentTCdf((x - p * y) / Math.sqrt((NU + y * y) * (1 - p * p) / (NU + 1)),
                            NU + 1);
                }
                case CLAYTON -> Math.pow(v, -p - 1) * Math.pow(Math.pow(u, -p) + Math.pow(v, -p) - 1, -1 - 1 / p);
                case GUMBEL -> {
                    double lu = -Math.log(u);
                    double lv = -Math.log(v);
                    double a = Math.pow(lu, p) + Math.pow(lv, p);
                    yield Math.exp(-Math.pow(a, 1 / p)) * Math.pow(lv, p - 1) / v * Math.pow(a, 1 / p - 1);
                }
                case FRANK -> {
                    double ev = Math.expm1(-p * v);
                    double eu = Math.expm1(-p * u);
                    yield (ev + 1) * eu / (Math.expm1(-p) + eu * ev);
                }
                default -> u;
            });
        }

        /**
         * The inverse of {@link #h} in its first argument, by bisection.
         *
         * @param w the target probability
         * @param v the conditioning uniform
         * @param p the parameter
         * @return {@code u} with {@code h(u | v) = w}
         */
        public double hInverse(double w, double v, double p) {
            if (this == INDEPENDENCE) {
                return w;
            }
            double lo = 0;
            double hi = 1;
            for (int i = 0; i < 60; i++) {
                double mid = 0.5 * (lo + hi);
                if (h(mid, v, p) < w) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return 0.5 * (lo + hi);
        }

        /**
         * The copula {@code C(u, v)} by quadrature of {@link #h} over the second argument.
         *
         * @param u first uniform
         * @param v second uniform
         * @param p the parameter
         * @return {@code P(U <= u, V <= v)}
         */
        public double cdf(double u, double v, double p) {
            int n = 400;
            double s = 0;
            for (int i = 0; i < n; i++) {
                s += h(u, v * (i + 0.5) / n, p);
            }
            return s * v / n;
        }

        /**
         * The lower and upper tail-dependence coefficients.
         *
         * @param p the parameter
         * @return {@code {lambda_L, lambda_U}}
         */
        public double[] tailDependence(double p) {
            return switch (this) {
                case CLAYTON -> new double[] {Math.pow(2, -1 / p), 0};
                case GUMBEL -> new double[] {0, 2 - Math.pow(2, 1 / p)};
                case STUDENT_T -> {
                    double l = 2 * (1 - Distributions.studentTCdf(Math.sqrt((NU + 1) * (1 - p) / (1 + p)), NU + 1));
                    yield new double[] {l, l};
                }
                default -> new double[] {0, 0};
            };
        }

        /**
         * The parameter matching a Kendall's tau (an inversion used as the MLE start), or NaN
         * when the family cannot express it.
         *
         * @param tau Kendall's tau
         * @return the parameter
         */
        public double fromTau(double tau) {
            return switch (this) {
                case GAUSSIAN, STUDENT_T -> Math.sin(Math.PI * tau / 2);
                case CLAYTON -> tau > 0.01 ? 2 * tau / (1 - tau) : Double.NaN;
                case GUMBEL -> tau > 0.01 ? 1 / (1 - tau) : Double.NaN;
                case FRANK -> Math.abs(tau) > 0.01 ? 9 * tau / (1 - Math.abs(tau)) : Double.NaN;
                default -> 0;
            };
        }

        private double toParam(double z) {
            return switch (this) {
                case GAUSSIAN, STUDENT_T -> 0.98 * Math.tanh(z);
                case CLAYTON -> Math.min(30, Math.exp(z));
                case GUMBEL -> Math.min(20, 1 + Math.exp(z));
                case FRANK -> Math.max(-40, Math.min(40, z));
                default -> 0;
            };
        }

        private double fromParam(double p) {
            return switch (this) {
                case GAUSSIAN, STUDENT_T -> atanh(Math.max(-0.97, Math.min(0.97, p / 0.98)));
                case CLAYTON -> Math.log(p);
                case GUMBEL -> Math.log(Math.max(p - 1, 1e-3));
                default -> p;
            };
        }

        private static double atanh(double x) {
            return 0.5 * Math.log((1 + x) / (1 - x));
        }
    }

    /**
     * A fitted pair copula.
     *
     * @param family the family
     * @param parameter its parameter
     * @param logLikelihood the training log likelihood
     */
    public record Pair(Family family, double parameter, double logLikelihood) {
    }

    /** How the trees are chosen. */
    public enum Structure {
        /** Dissmann's maximum spanning trees. */
        R_VINE,
        /** Stars around the most dependent node. */
        C_VINE,
        /** A path as the first tree. */
        D_VINE
    }

    /** An edge of a vine tree: conditioned pair {@code (a, b)} given {@code cond}. */
    private static final class Edge {
        final int a;
        final int b;
        final Set<Integer> cond;
        final int left;
        final int right;
        final boolean leftIsA;
        Pair pair;
        double[] ua;
        double[] ub;

        Edge(int a, int b, Set<Integer> cond, int left, int right, boolean leftIsA) {
            this.a = a;
            this.b = b;
            this.cond = cond;
            this.left = left;
            this.right = right;
            this.leftIsA = leftIsA;
        }

        Set<Integer> constraint() {
            Set<Integer> s = new HashSet<>(cond);
            s.add(a);
            s.add(b);
            return s;
        }

        /** F(var | constraint \ var) after this edge, on the training data. */
        double[] conditional(int var) {
            double[] out = new double[ua.length];
            for (int i = 0; i < ua.length; i++) {
                out[i] = var == a ? pair.family().h(ua[i], ub[i], pair.parameter())
                        : pair.family().h(ub[i], ua[i], pair.parameter());
            }
            return out;
        }
    }

    private final int dimension;
    private final List<List<Edge>> trees;

    private VineCopula(int dimension, List<List<Edge>> trees) {
        this.dimension = dimension;
        this.trees = trees;
    }

    /**
     * Pseudo-observations: per-column ranks divided by {@code n + 1}.
     *
     * @param x data {@code [n][d]}
     * @return uniforms {@code [n][d]}
     */
    public static double[][] pseudoObservations(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[][] u = new double[n][d];
        for (int j = 0; j < d; j++) {
            int col = j;
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) {
                idx[i] = i;
            }
            Arrays.sort(idx, (p, q) -> Double.compare(x[p][col], x[q][col]));
            for (int r = 0; r < n; r++) {
                u[idx[r]][j] = (r + 1.0) / (n + 1);
            }
        }
        return u;
    }

    /**
     * Kendall's tau (O(n^2)).
     *
     * @param a first sample
     * @param b second sample
     * @return tau
     */
    public static double kendallTau(double[] a, double[] b) {
        int n = Math.min(a.length, 600);
        long c = 0;
        long dcount = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = Math.signum(a[i] - a[j]) * Math.signum(b[i] - b[j]);
                if (s > 0) {
                    c++;
                } else if (s < 0) {
                    dcount++;
                }
            }
        }
        return (c - dcount) / (double) Math.max(1, c + dcount);
    }

    /**
     * Fits the best family by AIC to one pair of uniform samples.
     *
     * @param u first uniforms
     * @param v second uniforms
     * @param families the candidates
     * @return the best pair copula
     */
    public static Pair fitPair(double[] u, double[] v, Family... families) {
        double tau = kendallTau(u, v);
        Pair best = new Pair(Family.INDEPENDENCE, 0, 0);
        double bestAic = 0;
        for (Family f : families) {
            if (f == Family.INDEPENDENCE) {
                continue;
            }
            double p0 = f.fromTau(tau);
            if (Double.isNaN(p0)) {
                continue;
            }
            Optimize.Minimum m = Optimize.nelderMead(z -> {
                double p = f.toParam(z[0]);
                double ll = 0;
                for (int i = 0; i < u.length; i++) {
                    ll += f.logDensity(u[i], v[i], p);
                }
                return Double.isFinite(ll) ? -ll : 1e300;
            }, new double[] {f.fromParam(p0)}, 0.2, 200, 1e-8);
            double ll = -m.value();
            double aic = -2 * ll + 2;
            if (aic < bestAic) {
                bestAic = aic;
                best = new Pair(f, f.toParam(m.x()[0]), ll);
            }
        }
        return best;
    }

    /**
     * Fits a vine to uniform data.
     *
     * @param u pseudo-observations {@code [n][d]}
     * @param structure the tree selection
     * @param families candidate pair families
     * @return the fitted vine
     */
    public static VineCopula fit(double[][] u, Structure structure, Family... families) {
        int d = u[0].length;
        int n = u.length;
        List<List<Edge>> trees = new ArrayList<>();
        // tree 1 nodes: the variables, with their uniforms
        List<double[]> nodeData = new ArrayList<>();
        List<Set<Integer>> nodeConstraint = new ArrayList<>();
        for (int j = 0; j < d; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = u[i][j];
            }
            nodeData.add(col);
            nodeConstraint.add(Set.of(j));
        }
        List<Edge> previous = null;
        for (int t = 0; t < d - 1; t++) {
            int m = nodeConstraint.size();
            double[][] w = new double[m][m];
            boolean[][] ok = new boolean[m][m];
            for (int i = 0; i < m; i++) {
                for (int j = i + 1; j < m; j++) {
                    Set<Integer> inter = new HashSet<>(nodeConstraint.get(i));
                    inter.retainAll(nodeConstraint.get(j));
                    ok[i][j] = inter.size() == nodeConstraint.get(i).size() - 1;
                    ok[j][i] = ok[i][j];
                    if (ok[i][j]) {
                        w[i][j] = Math.abs(kendallTau(data(previous, nodeData, i, nodeConstraint, j),
                                data(previous, nodeData, j, nodeConstraint, i)));
                        w[j][i] = w[i][j];
                    }
                }
            }
            List<int[]> treeEdges = select(w, ok, t == 0 ? structure : (structure == Structure.C_VINE
                    ? Structure.C_VINE : Structure.R_VINE));
            List<Edge> tree = new ArrayList<>();
            for (int[] e : treeEdges) {
                int i = e[0];
                int j = e[1];
                Set<Integer> ci = nodeConstraint.get(i);
                Set<Integer> cj = nodeConstraint.get(j);
                Set<Integer> inter = new HashSet<>(ci);
                inter.retainAll(cj);
                int a = ci.stream().filter(x -> !cj.contains(x)).findFirst().orElseThrow();
                int b = cj.stream().filter(x -> !ci.contains(x)).findFirst().orElseThrow();
                Edge edge = new Edge(a, b, inter, i, j, true);
                edge.ua = data(previous, nodeData, i, nodeConstraint, j);
                edge.ub = data(previous, nodeData, j, nodeConstraint, i);
                edge.pair = fitPair(edge.ua, edge.ub, families);
                tree.add(edge);
            }
            trees.add(tree);
            previous = tree;
            nodeData = null;
            nodeConstraint = new ArrayList<>();
            for (Edge e : tree) {
                nodeConstraint.add(e.constraint());
            }
        }
        return new VineCopula(d, trees);
    }

    /** The pseudo-observation of node {@code i} towards neighbour {@code j}. */
    private static double[] data(List<Edge> previous, List<double[]> nodeData, int i, List<Set<Integer>> cons,
            int j) {
        if (previous == null) {
            return nodeData.get(i);
        }
        Set<Integer> ci = cons.get(i);
        Set<Integer> cj = cons.get(j);
        int x = ci.stream().filter(v -> !cj.contains(v)).findFirst().orElseThrow();
        return previous.get(i).conditional(x);
    }

    private static List<int[]> select(double[][] w, boolean[][] ok, Structure s) {
        int m = w.length;
        List<int[]> out = new ArrayList<>();
        if (s == Structure.C_VINE) {
            int root = -1;
            double best = -1;
            for (int i = 0; i < m; i++) {
                double tot = 0;
                boolean all = true;
                for (int j = 0; j < m; j++) {
                    if (j != i) {
                        all &= ok[i][j];
                        tot += w[i][j];
                    }
                }
                if (all && tot > best) {
                    best = tot;
                    root = i;
                }
            }
            if (root >= 0) {
                for (int j = 0; j < m; j++) {
                    if (j != root) {
                        out.add(new int[] {root, j});
                    }
                }
                return out;
            }
        }
        if (s == Structure.D_VINE) {
            // greedy maximum-weight path, best start
            List<int[]> bestPath = null;
            double bestW = -1;
            for (int start = 0; start < m; start++) {
                boolean[] used = new boolean[m];
                used[start] = true;
                int cur = start;
                double tot = 0;
                List<int[]> path = new ArrayList<>();
                for (int k = 1; k < m; k++) {
                    int nxt = -1;
                    for (int j = 0; j < m; j++) {
                        if (!used[j] && ok[cur][j] && (nxt < 0 || w[cur][j] > w[cur][nxt])) {
                            nxt = j;
                        }
                    }
                    if (nxt < 0) {
                        break;
                    }
                    path.add(new int[] {cur, nxt});
                    tot += w[cur][nxt];
                    used[nxt] = true;
                    cur = nxt;
                }
                if (path.size() == m - 1 && tot > bestW) {
                    bestW = tot;
                    bestPath = path;
                }
            }
            if (bestPath != null) {
                return bestPath;
            }
        }
        // Prim's maximum spanning tree over allowed edges
        boolean[] in = new boolean[m];
        in[0] = true;
        for (int k = 1; k < m; k++) {
            int bi = -1;
            int bj = -1;
            for (int i = 0; i < m; i++) {
                if (!in[i]) {
                    continue;
                }
                for (int j = 0; j < m; j++) {
                    if (!in[j] && ok[i][j] && (bi < 0 || w[i][j] > w[bi][bj])) {
                        bi = i;
                        bj = j;
                    }
                }
            }
            if (bi < 0) {
                break;
            }
            in[bj] = true;
            out.add(new int[] {bi, bj});
        }
        return out;
    }

    /**
     * The log copula density of new uniform rows.
     *
     * @param u uniforms {@code [n][d]}
     * @return per-row {@code log c(u)}
     */
    public double[] logDensity(double[][] u) {
        int n = u.length;
        double[] out = new double[n];
        List<double[][]> prevOut = null;   // per edge: {F(a|...), F(b|...)} on the new data
        List<Edge> prev = null;
        for (List<Edge> tree : trees) {
            List<double[][]> cur = new ArrayList<>();
            for (Edge e : tree) {
                double[] ua = new double[n];
                double[] ub = new double[n];
                for (int i = 0; i < n; i++) {
                    if (prev == null) {
                        ua[i] = u[i][e.a];
                        ub[i] = u[i][e.b];
                    }
                }
                if (prev != null) {
                    ua = pick(prev, prevOut, e.left, e.a);
                    ub = pick(prev, prevOut, e.right, e.b);
                }
                double[] ha = new double[n];
                double[] hb = new double[n];
                for (int i = 0; i < n; i++) {
                    out[i] += e.pair.family().logDensity(ua[i], ub[i], e.pair.parameter());
                    ha[i] = e.pair.family().h(ua[i], ub[i], e.pair.parameter());
                    hb[i] = e.pair.family().h(ub[i], ua[i], e.pair.parameter());
                }
                cur.add(new double[][] {ha, hb});
            }
            prev = tree;
            prevOut = cur;
        }
        return out;
    }

    private static double[] pick(List<Edge> prev, List<double[][]> prevOut, int node, int var) {
        Edge p = prev.get(node);
        return var == p.a ? prevOut.get(node)[0] : prevOut.get(node)[1];
    }

    /**
     * The first tree's edges as variable pairs.
     *
     * @return {@code [d-1][2]}
     */
    public int[][] firstTree() {
        int[][] out = new int[trees.get(0).size()][];
        for (int i = 0; i < out.length; i++) {
            Edge e = trees.get(0).get(i);
            out[i] = new int[] {Math.min(e.a, e.b), Math.max(e.a, e.b)};
        }
        return out;
    }

    /**
     * The fitted pair copula of a first-tree edge, or null if the pair is not in the first tree.
     *
     * @param a one variable
     * @param b the other variable
     * @return the pair copula, oriented as {@code (a, b)} up to symmetry
     */
    public Pair firstTreePair(int a, int b) {
        for (Edge e : trees.get(0)) {
            if ((e.a == a && e.b == b) || (e.a == b && e.b == a)) {
                return e.pair;
            }
        }
        return null;
    }

    /**
     * The total number of pair copulas.
     *
     * @return {@code d(d-1)/2}
     */
    public int pairs() {
        int s = 0;
        for (List<Edge> t : trees) {
            s += t.size();
        }
        return s;
    }

    /**
     * The dimension.
     *
     * @return {@code d}
     */
    public int dimension() {
        return dimension;
    }

    /**
     * Draws from a C-vine whose tree {@code j} (0-based) links root variable {@code j} to every
     * later variable {@code j + 1 + i} given variables {@code 0..j-1} (Aas et al. 2009,
     * Algorithm 1).
     *
     * @param families {@code families[j][i]} for tree {@code j}, edge {@code i}
     * @param parameters matching parameters
     * @param n the number of rows
     * @param random the random source
     * @return uniforms {@code [n][d]}
     */
    public static double[][] sampleCVine(Family[][] families, double[][] parameters, int n,
            java.util.SplittableRandom random) {
        int d = families.length + 1;
        double[][] out = new double[n][d];
        for (int r = 0; r < n; r++) {
            double[][] v = new double[d][d];
            double[] w = new double[d];
            for (int i = 0; i < d; i++) {
                w[i] = random.nextDouble();
            }
            out[r][0] = w[0];
            v[0][0] = w[0];
            for (int i = 1; i < d; i++) {
                v[i][0] = w[i];
                for (int k = i - 1; k >= 0; k--) {
                    v[i][0] = families[k][i - k - 1].hInverse(v[i][0], v[k][k], parameters[k][i - k - 1]);
                }
                out[r][i] = v[i][0];
                if (i == d - 1) {
                    break;
                }
                for (int j = 0; j < i; j++) {
                    v[i][j + 1] = families[j][i - j - 1].h(v[i][j], v[j][j], parameters[j][i - j - 1]);
                }
            }
        }
        return out;
    }
}
