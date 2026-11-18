package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Sparse integer risk scores: a few features with small integer points whose total maps to a risk
 * through the logistic curve {@code risk = 1 / (1 + exp(-(sum_j points_j x_j + b) / m))}.
 *
 * <ul>
 *   <li>{@link #riskSlim} (Ustun and Rudin, JMLR 2019, <a href="https://arxiv.org/abs/1610.00168">arXiv:1610.00168</a>):
 *       the integer, {@code l0}-constrained logistic-loss problem solved by the lattice cutting
 *       plane algorithm: the convex loss is replaced by the maximum of its tangent planes
 *       {@code V >= L(t) + g(t) (lambda - t)}, the mixed-integer program over integer points in
 *       {@code [-limit, limit]} with indicator variables for the support is solved by branch and
 *       bound over the core {@link Optimize#linearProgram} relaxation, and a new cut is added at
 *       every integer solution whose true loss exceeds its cutting-plane value (lazy cuts), until
 *       the gap closes or the node budget runs out ({@code m = 1}).</li>
 *   <li>{@link #fasterRisk} (Liu et al., NeurIPS 2022, <a href="https://arxiv.org/abs/2210.05846">arXiv:2210.05846</a>):
 *       a beam search grows sparse continuous logistic models one feature at a time, a
 *       swap step collects a pool of near-optimal sparse models with different supports, and
 *       each is rounded by a star-ray search over multipliers {@code m} with integer sequential
 *       rounding (repeatedly rounding the coordinate whose floor or ceiling costs least).</li>
 * </ul>
 */
public final class ScoringSystems {

    private ScoringSystems() {
    }

    /**
     * A scoring system.
     *
     * @param points integer points per feature
     * @param intercept the integer offset
     * @param multiplier the scale {@code m} of the logistic link
     * @param loss the mean logistic loss on the training rows
     * @param millis the solve time
     * @param certified whether the method proved optimality (RiskSLIM only)
     */
    public record Score(int[] points, int intercept, double multiplier, double loss, double millis,
            boolean certified) {

        /**
         * @param x a row
         * @return the predicted risk
         */
        public double risk(double[] x) {
            double z = intercept;
            for (int j = 0; j < points.length; j++) {
                z += points[j] * x[j];
            }
            return 1 / (1 + Math.exp(-z / multiplier));
        }

        /** @return the number of non-zero points */
        public int size() {
            return (int) Arrays.stream(points).filter(v -> v != 0).count();
        }
    }

    /**
     * The mean logistic loss of a linear score.
     *
     * @param x rows
     * @param y 0/1 labels
     * @param w coefficients
     * @param b intercept
     * @return the loss
     */
    public static double loss(double[][] x, double[] y, double[] w, double b) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double z = b;
            for (int j = 0; j < w.length; j++) {
                if (w[j] != 0) {
                    z += w[j] * x[i][j];
                }
            }
            double m = y[i] > 0.5 ? z : -z;
            s += m > 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m));
        }
        return s / x.length;
    }

    /** The gradient of {@link #loss}: {@code p} coefficient entries then the intercept. */
    private static double[] gradient(double[][] x, double[] y, double[] w, double b) {
        double[] g = new double[w.length + 1];
        for (int i = 0; i < x.length; i++) {
            double z = b;
            for (int j = 0; j < w.length; j++) {
                z += w[j] * x[i][j];
            }
            double r = 1 / (1 + Math.exp(-z)) - y[i];
            for (int j = 0; j < w.length; j++) {
                g[j] += r * x[i][j] / x.length;
            }
            g[w.length] += r / x.length;
        }
        return g;
    }

    // ------------------------------------------------------------------ RiskSLIM

    /**
     * RiskSLIM by lattice cutting planes.
     *
     * @param x binary rows
     * @param y 0/1 labels
     * @param k the most non-zero points
     * @param limit the largest absolute point value, such as 5
     * @param maxNodes the branch-and-bound budget
     * @return the score ({@code certified} when the gap closed within the budget)
     */
    public static Score riskSlim(double[][] x, double[] y, int k, int limit, int maxNodes) {
        long t0 = System.nanoTime();
        int p = x[0].length;
        int interceptLimit = 2 * limit;
        // variables: u_j, v_j (points = u - v), a_j (support), bu, bv (intercept), V
        int nv = 3 * p + 3;
        int iu = 0;
        int iv = p;
        int ia = 2 * p;
        int ib = 3 * p;
        int iV = 3 * p + 2;
        double[] c = new double[nv];
        c[iV] = 1;
        for (int j = 0; j < p; j++) {
            c[ia + j] = 1e-6;
        }
        List<double[]> rows = new ArrayList<>();
        List<Double> rhs = new ArrayList<>();
        for (int j = 0; j < p; j++) {
            double[] r = new double[nv];
            r[iu + j] = 1;
            r[iv + j] = 1;
            r[ia + j] = -limit;
            rows.add(r);
            rhs.add(0.0);
            double[] a = new double[nv];
            a[ia + j] = 1;
            rows.add(a);
            rhs.add(1.0);
        }
        double[] card = new double[nv];
        for (int j = 0; j < p; j++) {
            card[ia + j] = 1;
        }
        rows.add(card);
        rhs.add((double) k);
        for (int s = 0; s < 2; s++) {
            double[] r = new double[nv];
            r[ib + s] = 1;
            rows.add(r);
            rhs.add((double) interceptLimit);
        }
        List<double[]> cuts = new ArrayList<>();
        List<Double> cutRhs = new ArrayList<>();
        // warm cuts at the origin and along the relaxation (Kelley's cutting planes)
        addCut(x, y, new double[p], 0, cuts, cutRhs, nv, iu, iv, ib, iV);
        for (int it = 0; it < 30; it++) {
            Optimize.LinearSolution s = solve(c, rows, rhs, cuts, cutRhs, List.of());
            if (s.status() != Optimize.Status.OPTIMAL) {
                break;
            }
            double[] w = points(s.x(), p, iu, iv);
            double b = s.x()[ib] - s.x()[ib + 1];
            double l = loss(x, y, w, b);
            if (l - s.x()[iV] < 1e-4) {
                break;
            }
            addCut(x, y, w, b, cuts, cutRhs, nv, iu, iv, ib, iV);
        }
        double best = Double.POSITIVE_INFINITY;
        double[] bestW = new double[p];
        double bestB = 0;
        List<List<double[]>> stack = new ArrayList<>();
        stack.add(new ArrayList<>());
        int nodes = 0;
        boolean certified = true;
        while (!stack.isEmpty()) {
            if (nodes >= maxNodes) {
                certified = false;
                break;
            }
            List<double[]> branch = stack.remove(stack.size() - 1);
            nodes++;
            for (int round = 0; round < 50; round++) {
                Optimize.LinearSolution s = solve(c, rows, rhs, cuts, cutRhs, branch);
                if (s.status() != Optimize.Status.OPTIMAL || s.x()[iV] >= best - 1e-6) {
                    break;
                }
                int frac = -1;
                double fracBest = 1e-6;
                for (int v = 0; v < 3 * p + 2; v++) {
                    double f = Math.abs(s.x()[v] - Math.rint(s.x()[v]));
                    if (f > fracBest) {
                        fracBest = f;
                        frac = v;
                    }
                }
                double[] polished = polish(x, y, roundTop(s.x(), p, iu, iv, k, limit),
                        Math.max(-interceptLimit, Math.min(interceptLimit, Math.rint(s.x()[ib] - s.x()[ib + 1]))), k,
                        limit, interceptLimit);
                double pl = loss(x, y, Arrays.copyOf(polished, p), polished[p]);
                if (pl < best) {
                    best = pl;
                    bestW = Arrays.copyOf(polished, p);
                    bestB = polished[p];
                }
                if (frac >= 0) {
                    double val = s.x()[frac];
                    List<double[]> down = new ArrayList<>(branch);
                    down.add(new double[] {frac, Math.floor(val), 1});
                    List<double[]> up = new ArrayList<>(branch);
                    up.add(new double[] {frac, Math.ceil(val), -1});
                    stack.add(down);
                    stack.add(up);
                    break;
                }
                double[] w = points(s.x(), p, iu, iv);
                double b = Math.rint(s.x()[ib] - s.x()[ib + 1]);
                double l = loss(x, y, w, b);
                if (l < best) {
                    best = l;
                    bestW = w;
                    bestB = b;
                }
                if (l - s.x()[iV] < 1e-6) {
                    break;
                }
                addCut(x, y, w, b, cuts, cutRhs, nv, iu, iv, ib, iV);   // lazy cut, then re-solve the node
            }
        }
        int[] pts = new int[p];
        for (int j = 0; j < p; j++) {
            pts[j] = (int) Math.rint(bestW[j]);
        }
        return new Score(pts, (int) Math.rint(bestB), 1, best, (System.nanoTime() - t0) / 1e6,
                certified);
    }

    /** The rounding heuristic: the relaxation's points rounded, keeping the {@code k} largest. */
    private static double[] roundTop(double[] sol, int p, int iu, int iv, int k, int limit) {
        double[] raw = new double[p];
        Integer[] order = new Integer[p];
        for (int j = 0; j < p; j++) {
            raw[j] = sol[iu + j] - sol[iv + j];
            order[j] = j;
        }
        Arrays.sort(order, (a, b) -> Double.compare(Math.abs(raw[b]), Math.abs(raw[a])));
        double[] w = new double[p];
        for (int r = 0; r < Math.min(k, p); r++) {
            int j = order[r];
            w[j] = Math.max(-limit, Math.min(limit, Math.rint(raw[j])));
        }
        return w;
    }

    /**
     * The polishing heuristic (discrete coordinate descent): moves one point or the intercept by
     * one step while the loss falls, keeping at most {@code k} non-zero points.
     *
     * @return the points followed by the intercept
     */
    private static double[] polish(double[][] x, double[] y, double[] w0, double b0, int k, int limit,
            int interceptLimit) {
        int p = w0.length;
        double[] w = w0.clone();
        double b = b0;
        double cur = loss(x, y, w, b);
        boolean improved = true;
        while (improved) {
            improved = false;
            int nonZero = 0;
            for (double v : w) {
                nonZero += v != 0 ? 1 : 0;
            }
            for (int j = -1; j < p; j++) {
                for (int d = -1; d <= 1; d += 2) {
                    if (j < 0) {
                        if (Math.abs(b + d) <= interceptLimit) {
                            double l = loss(x, y, w, b + d);
                            if (l < cur - 1e-12) {
                                b += d;
                                cur = l;
                                improved = true;
                            }
                        }
                        continue;
                    }
                    double old = w[j];
                    double nv = old + d;
                    if (Math.abs(nv) > limit || (old == 0 && nonZero >= k)) {
                        continue;
                    }
                    w[j] = nv;
                    double l = loss(x, y, w, b);
                    if (l < cur - 1e-12) {
                        cur = l;
                        improved = true;
                        nonZero += (old == 0 ? 1 : 0) - (nv == 0 ? 1 : 0);
                    } else {
                        w[j] = old;
                    }
                }
            }
        }
        double[] out = Arrays.copyOf(w, p + 1);
        out[p] = b;
        return out;
    }

    private static double[] points(double[] sol, int p, int iu, int iv) {
        double[] w = new double[p];
        for (int j = 0; j < p; j++) {
            w[j] = Math.rint(sol[iu + j] - sol[iv + j]);
        }
        return w;
    }

    private static void addCut(double[][] x, double[] y, double[] w, double b, List<double[]> cuts,
            List<Double> cutRhs, int nv, int iu, int iv, int ib, int iV) {
        int p = w.length;
        double l = loss(x, y, w, b);
        double[] g = gradient(x, y, w, b);
        // l + g (theta - t) <= V  ->  g theta - V <= g t - l
        double[] row = new double[nv];
        double gt = 0;
        for (int j = 0; j < p; j++) {
            row[iu + j] = g[j];
            row[iv + j] = -g[j];
            gt += g[j] * w[j];
        }
        row[ib] = g[p];
        row[ib + 1] = -g[p];
        gt += g[p] * b;
        row[iV] = -1;
        cuts.add(row);
        cutRhs.add(gt - l);
    }

    /** Solves the relaxation with branching bounds {@code {var, value, +1 for <=, -1 for >=}}. */
    private static Optimize.LinearSolution solve(double[] c, List<double[]> rows, List<Double> rhs, List<double[]> cuts,
            List<Double> cutRhs, List<double[]> branch) {
        int m = rows.size() + cuts.size() + branch.size();
        double[][] a = new double[m][];
        double[] b = new double[m];
        int r = 0;
        for (int i = 0; i < rows.size(); i++, r++) {
            a[r] = rows.get(i);
            b[r] = rhs.get(i);
        }
        for (int i = 0; i < cuts.size(); i++, r++) {
            a[r] = cuts.get(i);
            b[r] = cutRhs.get(i);
        }
        for (double[] br : branch) {
            double[] row = new double[c.length];
            row[(int) br[0]] = br[2];
            a[r] = row;
            b[r] = br[2] * br[1];
            r++;
        }
        try {
            return Optimize.linearProgram(c, a, b, null, null);
        } catch (IllegalStateException numericalFailure) {
            // the core simplex can stall on badly scaled cut rows; such a node is pruned
            return new Optimize.LinearSolution(Optimize.Status.INFEASIBLE, new double[c.length], Double.NaN);
        }
    }

    // ------------------------------------------------------------------ FasterRisk

    /** A sparse continuous logistic model. */
    private record Sparse(double[] w, double b, double loss) {
    }

    /** Coordinate Newton steps on the support, coefficients clipped to the box. */
    private static Sparse refit(double[][] x, double[] y, double[] w0, double b0, boolean[] support, double box,
            int sweeps) {
        int p = w0.length;
        int n = x.length;
        double[] w = w0.clone();
        double b = b0;
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = b;
            for (int j = 0; j < p; j++) {
                z[i] += w[j] * x[i][j];
            }
        }
        for (int s = 0; s < sweeps; s++) {
            for (int j = -1; j < p; j++) {
                if (j >= 0 && !support[j]) {
                    continue;
                }
                double g = 0;
                double h = 1e-6;
                for (int i = 0; i < n; i++) {
                    double xi = j < 0 ? 1 : x[i][j];
                    if (xi == 0) {
                        continue;
                    }
                    double q = 1 / (1 + Math.exp(-z[i]));
                    g += (q - y[i]) * xi;
                    h += q * (1 - q) * xi * xi;
                }
                double step = -g / h;
                double old = j < 0 ? b : w[j];
                double nu = j < 0 ? Math.max(-2 * box, Math.min(2 * box, old + step))
                        : Math.max(-box, Math.min(box, old + step));
                double d = nu - old;
                if (j < 0) {
                    b = nu;
                } else {
                    w[j] = nu;
                }
                for (int i = 0; i < n; i++) {
                    z[i] += d * (j < 0 ? 1 : x[i][j]);
                }
            }
        }
        return new Sparse(w, b, loss(x, y, w, b));
    }

    /**
     * FasterRisk: sparse beam search, a diverse pool and star-ray integer rounding.
     *
     * @param x binary rows
     * @param y 0/1 labels
     * @param k the most non-zero points
     * @param limit the largest absolute point value
     * @param beam the beam width
     * @return the best rounded score
     */
    public static Score fasterRisk(double[][] x, double[] y, int k, int limit, int beam) {
        long t0 = System.nanoTime();
        int p = x[0].length;
        double ybar = Arrays.stream(y).average().orElse(0.5);
        double b0 = Math.log(Math.max(ybar, 1e-6) / Math.max(1 - ybar, 1e-6));
        List<Sparse> current = List.of(new Sparse(new double[p], b0, loss(x, y, new double[p], b0)));
        for (int size = 1; size <= k; size++) {
            List<Sparse> next = new ArrayList<>();
            for (Sparse s : current) {
                for (int j = 0; j < p; j++) {
                    if (s.w()[j] != 0) {
                        continue;
                    }
                    boolean[] sup = support(s.w());
                    sup[j] = true;
                    next.add(refit(x, y, s.w(), s.b(), sup, limit, 2));
                }
            }
            next.sort(Comparator.comparingDouble(Sparse::loss));
            List<Sparse> kept = new ArrayList<>();
            for (Sparse s : next) {
                if (kept.stream().noneMatch(q -> Arrays.equals(support(q.w()), support(s.w())))) {
                    kept.add(refit(x, y, s.w(), s.b(), support(s.w()), limit, 6));
                }
                if (kept.size() == beam) {
                    break;
                }
            }
            current = kept;
        }
        Sparse best = current.stream().min(Comparator.comparingDouble(Sparse::loss)).orElseThrow();
        // the diverse pool: swap one feature of the best support for another
        List<Sparse> pool = new ArrayList<>(current);
        for (int out = 0; out < p; out++) {
            if (best.w()[out] == 0) {
                continue;
            }
            for (int in = 0; in < p; in++) {
                if (best.w()[in] != 0) {
                    continue;
                }
                boolean[] sup = support(best.w());
                sup[out] = false;
                sup[in] = true;
                double[] w = best.w().clone();
                w[out] = 0;
                Sparse s = refit(x, y, w, best.b(), sup, limit, 4);
                if (s.loss() <= best.loss() * 1.003) {
                    pool.add(s);
                }
            }
        }
        Score bestScore = null;
        for (Sparse s : pool) {
            double maxAbs = Arrays.stream(s.w()).map(Math::abs).max().orElse(1);
            double top = Math.max(1, limit / Math.max(maxAbs, 1e-9));
            for (int g = 0; g < 20; g++) {
                double m = 1 + (top - 1) * g / 19.0;
                Score r = roundSequentially(x, y, s, m, limit);
                if (bestScore == null || r.loss() < bestScore.loss()) {
                    bestScore = r;
                }
            }
        }
        return new Score(bestScore.points(), bestScore.intercept(), bestScore.multiplier(), bestScore.loss(),
                (System.nanoTime() - t0) / 1e6, false);
    }

    private static boolean[] support(double[] w) {
        boolean[] s = new boolean[w.length];
        for (int j = 0; j < w.length; j++) {
            s[j] = w[j] != 0;
        }
        return s;
    }

    /** Integer sequential rounding of {@code m * (w, b)}, the loss judged at {@code (lambda, b) / m}. */
    private static Score roundSequentially(double[][] x, double[] y, Sparse s, double m, int limit) {
        int p = s.w().length;
        double[] v = new double[p + 1];
        for (int j = 0; j < p; j++) {
            v[j] = s.w()[j] * m;
        }
        v[p] = s.b() * m;
        boolean[] done = new boolean[p + 1];
        while (true) {
            int bestJ = -1;
            double bestVal = 0;
            double bestLoss = Double.POSITIVE_INFINITY;
            for (int j = 0; j <= p; j++) {
                if (done[j]) {
                    continue;
                }
                if (v[j] == Math.rint(v[j])) {
                    done[j] = true;
                    continue;
                }
                for (double cand : new double[] {Math.floor(v[j]), Math.ceil(v[j])}) {
                    if (j < p && Math.abs(cand) > limit) {
                        continue;
                    }
                    double keep = v[j];
                    v[j] = cand;
                    double l = scaledLoss(x, y, v, m);
                    v[j] = keep;
                    if (l < bestLoss) {
                        bestLoss = l;
                        bestJ = j;
                        bestVal = cand;
                    }
                }
            }
            if (bestJ < 0) {
                break;
            }
            v[bestJ] = bestVal;
            done[bestJ] = true;
        }
        int[] pts = new int[p];
        for (int j = 0; j < p; j++) {
            pts[j] = (int) Math.rint(v[j]);
        }
        return new Score(pts, (int) Math.rint(v[p]), m, scaledLoss(x, y, v, m), 0, false);
    }

    private static double scaledLoss(double[][] x, double[] y, double[] v, double m) {
        int p = v.length - 1;
        double[] w = new double[p];
        for (int j = 0; j < p; j++) {
            w[j] = v[j] / m;
        }
        return loss(x, y, w, v[p] / m);
    }

    /**
     * The naive baseline: continuous coefficients rounded to the nearest integers (clipped to the
     * range), intercept rounded, multiplier 1.
     *
     * @param x rows
     * @param y labels
     * @param w continuous coefficients
     * @param b continuous intercept
     * @param limit the largest absolute point value
     * @return the rounded score
     */
    public static Score naiveRounding(double[][] x, double[] y, double[] w, double b, int limit) {
        int[] pts = new int[w.length];
        double[] wr = new double[w.length];
        for (int j = 0; j < w.length; j++) {
            pts[j] = (int) Math.max(-limit, Math.min(limit, Math.rint(w[j])));
            wr[j] = pts[j];
        }
        int bi = (int) Math.rint(b);
        return new Score(pts, bi, 1, loss(x, y, wr, bi), 0, false);
    }
}
