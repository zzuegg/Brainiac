package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Multivariate adaptive regression splines (Friedman, Annals of Statistics 1991). The model is a
 * linear combination of basis functions, each a product of hinges {@code max(0, x_v - t)} or
 * {@code max(0, t - x_v)}. The forward pass greedily adds the mirrored pair (parent times hinge)
 * that most reduces the residual sum of squares, over every existing parent (up to the maximum
 * interaction degree), variable and knot; candidates are scored by projecting the residual on
 * their components orthogonalised against the current basis. The backward pass deletes terms one
 * at a time while generalised cross-validation {@code RSS / n / (1 - C(M) / n)^2},
 * {@code C(M) = M + d (M - 1) / 2}, improves.
 */
public final class Mars {

    /** A basis function: a product of hinges, given as variable, knot and sign triples. */
    private record Term(int[] vars, double[] knots, int[] signs) {

        double eval(double[] x) {
            double v = 1;
            for (int k = 0; k < vars.length; k++) {
                double h = signs[k] > 0 ? x[vars[k]] - knots[k] : knots[k] - x[vars[k]];
                if (h <= 0) {
                    return 0;
                }
                v *= h;
            }
            return v;
        }

        Term times(int var, double knot, int sign) {
            int d = vars.length;
            int[] nv = Arrays.copyOf(vars, d + 1);
            double[] nk = Arrays.copyOf(knots, d + 1);
            int[] ns = Arrays.copyOf(signs, d + 1);
            nv[d] = var;
            nk[d] = knot;
            ns[d] = sign;
            return new Term(nv, nk, ns);
        }

        boolean uses(int var) {
            for (int v : vars) {
                if (v == var) {
                    return true;
                }
            }
            return false;
        }
    }

    private final List<Term> terms;
    private final double[] coef;

    private Mars(List<Term> terms, double[] coef) {
        this.terms = terms;
        this.coef = coef;
    }

    /**
     * Fits MARS.
     *
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param maxTerms the forward-pass limit on terms (including the constant)
     * @param maxDegree the largest interaction order
     * @param knotsPerVariable candidate knots per variable (evenly spaced order statistics)
     * @return the pruned model
     */
    public static Mars fit(double[][] x, double[] y, int maxTerms, int maxDegree, int knotsPerVariable) {
        int n = y.length;
        int p = x[0].length;
        double[][] knots = new double[p][];
        for (int v = 0; v < p; v++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = x[i][v];
            }
            Arrays.sort(col);
            knots[v] = new double[knotsPerVariable];
            for (int k = 0; k < knotsPerVariable; k++) {
                knots[v][k] = col[(int) ((k + 1.0) / (knotsPerVariable + 1) * (n - 1))];
            }
        }
        List<Term> terms = new ArrayList<>();
        terms.add(new Term(new int[0], new double[0], new int[0]));
        List<double[]> cols = new ArrayList<>(); // basis columns
        List<double[]> ortho = new ArrayList<>(); // orthonormal basis of their span
        double[] one = new double[n];
        Arrays.fill(one, 1);
        cols.add(one);
        addOrtho(ortho, one);
        double[] resid = residual(y, ortho);
        while (terms.size() + 2 <= maxTerms) {
            double bestGain = 1e-9;
            Term bestA = null;
            Term bestB = null;
            int parents = terms.size();
            for (int m = 0; m < parents; m++) {
                Term parent = terms.get(m);
                if (parent.vars().length >= maxDegree) {
                    continue;
                }
                double[] pc = cols.get(m);
                for (int v = 0; v < p; v++) {
                    if (parent.uses(v)) {
                        continue;
                    }
                    for (double t : knots[v]) {
                        double[] a = new double[n];
                        double[] b = new double[n];
                        for (int i = 0; i < n; i++) {
                            a[i] = pc[i] * Math.max(0, x[i][v] - t);
                            b[i] = pc[i] * Math.max(0, t - x[i][v]);
                        }
                        double gain = gain(ortho, resid, a, b);
                        if (gain > bestGain) {
                            bestGain = gain;
                            bestA = parent.times(v, t, 1);
                            bestB = parent.times(v, t, -1);
                        }
                    }
                }
            }
            if (bestA == null) {
                break;
            }
            for (Term t : new Term[] {bestA, bestB}) {
                double[] c = column(t, x);
                terms.add(t);
                cols.add(c);
                addOrtho(ortho, c);
            }
            resid = residual(y, ortho);
        }
        // backward pruning by GCV
        List<Integer> keep = new ArrayList<>();
        for (int k = 0; k < terms.size(); k++) {
            keep.add(k);
        }
        List<Integer> best = new ArrayList<>(keep);
        double bestGcv = gcv(cols, keep, y);
        while (keep.size() > 1) {
            double roundBest = Double.POSITIVE_INFINITY;
            int drop = -1;
            for (int a = 1; a < keep.size(); a++) {
                List<Integer> trial = new ArrayList<>(keep);
                trial.remove(a);
                double g = gcv(cols, trial, y);
                if (g < roundBest) {
                    roundBest = g;
                    drop = a;
                }
            }
            keep.remove(drop);
            if (roundBest < bestGcv) {
                bestGcv = roundBest;
                best = new ArrayList<>(keep);
            }
        }
        List<Term> chosen = new ArrayList<>();
        for (int k : best) {
            chosen.add(terms.get(k));
        }
        return new Mars(chosen, leastSquares(cols, best, y));
    }

    private static double[] column(Term t, double[][] x) {
        double[] c = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            c[i] = t.eval(x[i]);
        }
        return c;
    }

    private static void addOrtho(List<double[]> ortho, double[] c) {
        double[] v = c.clone();
        for (int pass = 0; pass < 2; pass++) {
            for (double[] q : ortho) {
                double d = dot(q, v);
                for (int i = 0; i < v.length; i++) {
                    v[i] -= d * q[i];
                }
            }
        }
        double norm = Math.sqrt(dot(v, v));
        if (norm > 1e-8 * Math.sqrt(dot(c, c)) && norm > 1e-12) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= norm;
            }
            ortho.add(v);
        }
    }

    private static double[] residual(double[] y, List<double[]> ortho) {
        double[] r = y.clone();
        for (double[] q : ortho) {
            double d = dot(q, r);
            for (int i = 0; i < r.length; i++) {
                r[i] -= d * q[i];
            }
        }
        return r;
    }

    /** RSS reduction from adding columns a and b (projected on the residual). */
    private static double gain(List<double[]> ortho, double[] resid, double[] a, double[] b) {
        double[] qa = orthogonalise(ortho, a, null);
        double na = dot(qa, qa);
        double g = 0;
        if (na > 1e-10) {
            double d = dot(qa, resid);
            g += d * d / na;
        }
        double[] qb = orthogonalise(ortho, b, na > 1e-10 ? qa : null);
        double nb = dot(qb, qb);
        if (nb > 1e-10) {
            double d = dot(qb, resid);
            g += d * d / nb;
        }
        return g;
    }

    private static double[] orthogonalise(List<double[]> ortho, double[] c, double[] extra) {
        double[] v = c.clone();
        for (double[] q : ortho) {
            double d = dot(q, v);
            if (d != 0) {
                for (int i = 0; i < v.length; i++) {
                    v[i] -= d * q[i];
                }
            }
        }
        if (extra != null) {
            double d = dot(extra, v) / dot(extra, extra);
            for (int i = 0; i < v.length; i++) {
                v[i] -= d * extra[i];
            }
        }
        return v;
    }

    private static double[] leastSquares(List<double[]> cols, List<Integer> use, double[] y) {
        int m = use.size();
        int n = y.length;
        double[][] g = new double[m][m];
        double[] b = new double[m];
        for (int a = 0; a < m; a++) {
            double[] ca = cols.get(use.get(a));
            b[a] = dot(ca, y);
            for (int c = a; c < m; c++) {
                g[a][c] = g[c][a] = dot(ca, cols.get(use.get(c)));
            }
            g[a][a] += 1e-8 * n;
        }
        return Recalibrators.solve(g, b);
    }

    private static double gcv(List<double[]> cols, List<Integer> use, double[] y) {
        double[] beta = leastSquares(cols, use, y);
        int n = y.length;
        double rss = 0;
        for (int i = 0; i < n; i++) {
            double f = 0;
            for (int a = 0; a < use.size(); a++) {
                f += beta[a] * cols.get(use.get(a))[i];
            }
            rss += (y[i] - f) * (y[i] - f);
        }
        int m = use.size();
        double c = m + 3.0 * (m - 1) / 2;
        double denom = 1 - c / n;
        return denom <= 0 ? Double.POSITIVE_INFINITY : rss / n / (denom * denom);
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * Predicts one row.
     *
     * @param row the input
     * @return the prediction
     */
    public double predict(double[] row) {
        double s = 0;
        for (int k = 0; k < terms.size(); k++) {
            s += coef[k] * terms.get(k).eval(row);
        }
        return s;
    }

    /**
     * The number of terms after pruning (including the constant).
     *
     * @return the size
     */
    public int size() {
        return terms.size();
    }
}
