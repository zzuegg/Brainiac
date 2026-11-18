package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Actionable recourse and counterfactual explanations that respect what a person can change.
 *
 * <ul>
 *   <li>{@link #linear} (Ustun, Spangher and Liu, FAT* 2019,
 *       <a href="https://arxiv.org/abs/1809.06514">arXiv:1809.06514</a>): for a linear
 *       classifier {@code w x + b}, every mutable feature gets a grid of allowed changes (only
 *       up, only down, or both), each with a cost; one change per feature is chosen by a 0/1
 *       integer program that minimises the total cost subject to the score reaching the margin.
 *       It is solved exactly by branch and bound over the core {@link Optimize#linearProgram}
 *       relaxation (the existing {@code combopt.BranchAndBound} only handles packing problems
 *       with non-negative rows, and this covering constraint has signed coefficients).</li>
 *   <li>{@code linear} with {@code epsilon > 0} is robust recourse (Upadhyay, Joshi and
 *       Lakkaraju, NeurIPS 2021, <a href="https://arxiv.org/abs/2102.13620">arXiv:2102.13620</a>):
 *       the margin must hold for every model update with {@code |dw|_inf, |db| <= epsilon},
 *       whose worst case for a linear model is {@code w x' + b - epsilon (|x'|_1 + 1)}; the
 *       per-level gains of the IP absorb the {@code |x'_j|} term exactly.</li>
 *   <li>{@link #dice} (Mothilal, Sharma and Tan, FAT* 2020,
 *       <a href="https://arxiv.org/abs/1905.07697">arXiv:1905.07697</a>): {@code k}
 *       counterfactuals optimised jointly for validity (a hinge on the score), proximity (the
 *       weighted L1 cost) and diversity (the determinant of the kernel
 *       {@code 1 / (1 + d(c_a, c_b))}), with finite-difference gradients so any black box works,
 *       projected onto the actionable set after every step.</li>
 *   <li>{@link #face} (Poyiadzi et al., AIES 2020, <a href="https://arxiv.org/abs/1909.09369">arXiv:1909.09369</a>):
 *       the cheapest path through a k-nearest-neighbour graph of the data, edges weighted by
 *       length over the local density and allowed only where the step is actionable, to a
 *       data point the model accepts; the recourse stays in high-density regions.</li>
 * </ul>
 */
public final class Recourse {

    private Recourse() {
    }

    /** How a feature may change. */
    public enum Mutability {
        /** Never. */
        IMMUTABLE,
        /** Only upwards. */
        INCREASE,
        /** Only downwards. */
        DECREASE,
        /** Either way. */
        FREE
    }

    /**
     * The actionability constraints.
     *
     * @param mutability how each feature may change
     * @param cost the cost of changing each feature by one unit of its scale
     * @param scale each feature's scale, such as its standard deviation
     */
    public record Constraints(Mutability[] mutability, double[] cost, double[] scale) {

        /**
         * The cost of a change; infinite if it violates the constraints.
         *
         * @param x the original row
         * @param c the changed row
         * @return {@code sum_j cost_j |c_j - x_j| / scale_j}
         */
        public double cost(double[] x, double[] c) {
            double s = 0;
            for (int j = 0; j < x.length; j++) {
                double d = c[j] - x[j];
                if (Math.abs(d) < 1e-9 * scale[j]) {
                    continue;
                }
                if (!allowed(j, d)) {
                    return Double.POSITIVE_INFINITY;
                }
                s += cost[j] * Math.abs(d) / scale[j];
            }
            return s;
        }

        /**
         * @param x the original row
         * @param c the changed row
         * @return whether every change respects the mutability
         */
        public boolean actionable(double[] x, double[] c) {
            for (int j = 0; j < x.length; j++) {
                double d = c[j] - x[j];
                if (Math.abs(d) > 1e-6 * scale[j] && !allowed(j, d)) {
                    return false;
                }
            }
            return true;
        }

        boolean allowed(int j, double d) {
            return switch (mutability[j]) {
                case IMMUTABLE -> false;
                case INCREASE -> d > 0;
                case DECREASE -> d < 0;
                case FREE -> true;
            };
        }

        /** Clamps a change (in scale units) onto the actionable set. */
        double project(int j, double u) {
            return switch (mutability[j]) {
                case IMMUTABLE -> 0;
                case INCREASE -> Math.max(0, u);
                case DECREASE -> Math.min(0, u);
                case FREE -> u;
            };
        }
    }

    // ------------------------------------------------------------------ linear IP

    /**
     * Minimum-cost actionable recourse for a linear classifier, optionally robust.
     *
     * @param w the weights
     * @param b the intercept
     * @param x the row (denied: {@code w x + b < margin})
     * @param constraints the constraints
     * @param margin the score the recourse must reach, such as a small positive value
     * @param levels change levels per direction
     * @param step the change per level in units of each feature's scale
     * @param epsilon the robustness radius on the weights and intercept, 0 for plain recourse
     * @return the changed row, or null if no actionable change reaches the margin
     */
    public static double[] linear(double[] w, double b, double[] x, Constraints constraints, double margin, int levels,
            double step, double epsilon) {
        int p = x.length;
        double base = b - epsilon;
        for (int j = 0; j < p; j++) {
            base += w[j] * x[j] - epsilon * Math.abs(x[j]);
        }
        double need = margin - base;
        if (need <= 0) {
            return x.clone();
        }
        List<int[]> vars = new ArrayList<>();   // {feature, signed level}
        List<Double> gain = new ArrayList<>();
        List<Double> cost = new ArrayList<>();
        for (int j = 0; j < p; j++) {
            for (int dir : new int[] {1, -1}) {
                for (int k = 1; k <= levels; k++) {
                    double d = dir * k * step * constraints.scale()[j];
                    if (!constraints.allowed(j, d)) {
                        continue;
                    }
                    double xj = x[j] + d;
                    vars.add(new int[] {j, dir * k});
                    gain.add(w[j] * d - epsilon * (Math.abs(xj) - Math.abs(x[j])));
                    cost.add(constraints.cost()[j] * k * step);
                }
            }
        }
        int m = vars.size();
        if (m == 0) {
            return null;
        }
        double[] c = cost.stream().mapToDouble(Double::doubleValue).toArray();
        double[][] aUb = new double[p + 1 + m][m];
        double[] bUb = new double[p + 1 + m];
        for (int v = 0; v < m; v++) {
            aUb[vars.get(v)[0]][v] = 1;
            aUb[p][v] = -gain.get(v);
            aUb[p + 1 + v][v] = 1;
        }
        for (int j = 0; j < p; j++) {
            bUb[j] = 1;
        }
        bUb[p] = -need;
        for (int v = 0; v < m; v++) {
            bUb[p + 1 + v] = 1;
        }
        double[] best = branchAndBound(c, aUb, bUb, m);
        if (best == null) {
            return null;
        }
        double[] out = x.clone();
        for (int v = 0; v < m; v++) {
            if (best[v] > 0.5) {
                int[] var = vars.get(v);
                out[var[0]] += var[1] * step * constraints.scale()[var[0]];
            }
        }
        return out;
    }

    /** Depth-first branch and bound for a 0/1 program {@code min c z, A z <= b}. */
    private static double[] branchAndBound(double[] c, double[][] aUb, double[] bUb, int m) {
        double[] incumbent = null;
        double incumbentValue = Double.POSITIVE_INFINITY;
        List<int[]> stack = new ArrayList<>();
        stack.add(new int[0]);   // pairs (var, value) flattened
        int nodes = 0;
        while (!stack.isEmpty() && nodes < 5000) {
            int[] fix = stack.remove(stack.size() - 1);
            nodes++;
            double[][] aEq = new double[fix.length / 2][m];
            double[] bEq = new double[fix.length / 2];
            for (int k = 0; k < fix.length / 2; k++) {
                aEq[k][fix[2 * k]] = 1;
                bEq[k] = fix[2 * k + 1];
            }
            Optimize.LinearSolution s = Optimize.linearProgram(c, aUb, bUb, aEq.length == 0 ? null : aEq,
                    aEq.length == 0 ? null : bEq);
            if (s.status() != Optimize.Status.OPTIMAL || s.value() >= incumbentValue - 1e-9) {
                continue;
            }
            int branch = -1;
            double frac = 0;
            for (int v = 0; v < m; v++) {
                double f = Math.abs(s.x()[v] - Math.rint(s.x()[v]));
                if (f > 1e-7 && f > frac) {
                    frac = f;
                    branch = v;
                }
            }
            if (branch < 0) {
                incumbent = s.x().clone();
                incumbentValue = s.value();
                continue;
            }
            for (int val : new int[] {0, 1}) {
                int[] next = Arrays.copyOf(fix, fix.length + 2);
                next[fix.length] = branch;
                next[fix.length + 1] = val;
                stack.add(next);
            }
        }
        return incumbent;
    }

    // ------------------------------------------------------------------ DiCE

    /**
     * Diverse counterfactuals by joint optimisation.
     *
     * @param score the model's batch score (accepted when {@code >= threshold})
     * @param threshold the decision threshold
     * @param x the denied row
     * @param constraints the constraints
     * @param k how many counterfactuals
     * @param steps gradient steps
     * @param diversityWeight the weight of the determinant term
     * @param seed the initialisation seed
     * @return {@code k} changed rows
     */
    public static double[][] dice(Function<double[][], double[]> score, double threshold, double[] x,
            Constraints constraints, int k, int steps, double diversityWeight, long seed) {
        int p = x.length;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] u = new double[k][p];
        for (double[] row : u) {
            for (int j = 0; j < p; j++) {
                row[j] = constraints.project(j, r.nextGaussian());
            }
        }
        double h = 0.05;
        double lr = 0.05;
        double target = threshold + 0.1;
        for (int s = 0; s < steps; s++) {
            double[][] probe = new double[k * (2 * p + 1)][];
            for (int a = 0; a < k; a++) {
                double[] cur = row(x, u[a], constraints);
                probe[a * (2 * p + 1)] = cur;
                for (int j = 0; j < p; j++) {
                    double[] plus = cur.clone();
                    plus[j] += h * constraints.scale()[j];
                    double[] minus = cur.clone();
                    minus[j] -= h * constraints.scale()[j];
                    probe[a * (2 * p + 1) + 1 + 2 * j] = plus;
                    probe[a * (2 * p + 1) + 2 + 2 * j] = minus;
                }
            }
            double[] f = score.apply(probe);
            double[][] grad = new double[k][p];
            for (int a = 0; a < k; a++) {
                double fa = f[a * (2 * p + 1)];
                boolean below = fa < target;
                for (int j = 0; j < p; j++) {
                    double df = (f[a * (2 * p + 1) + 1 + 2 * j] - f[a * (2 * p + 1) + 2 + 2 * j]) / (2 * h);
                    if (Double.isInfinite(constraints.cost()[j])) {
                        continue;
                    }
                    if (below) {
                        grad[a][j] -= 2 * df;   // d/du of the hinge (target - f)
                    }
                    grad[a][j] += 0.1 * constraints.cost()[j] * Math.signum(u[a][j]) / k;
                }
            }
            if (k > 1 && diversityWeight > 0) {
                double[][] dd = diversityGradient(u, constraints.cost());
                for (int a = 0; a < k; a++) {
                    for (int j = 0; j < p; j++) {
                        grad[a][j] -= diversityWeight * dd[a][j];
                    }
                }
            }
            for (int a = 0; a < k; a++) {
                for (int j = 0; j < p; j++) {
                    u[a][j] = constraints.project(j, u[a][j] - lr * grad[a][j]);
                    if (Double.isInfinite(constraints.cost()[j])) {
                        u[a][j] = 0;
                    }
                }
            }
        }
        double[][] out = new double[k][];
        for (int a = 0; a < k; a++) {
            out[a] = row(x, u[a], constraints);
        }
        return out;
    }

    private static double[] row(double[] x, double[] u, Constraints c) {
        double[] out = x.clone();
        for (int j = 0; j < x.length; j++) {
            out[j] += u[j] * c.scale()[j];
        }
        return out;
    }

    /** The finite-difference gradient of {@code det K} with {@code K_ab = 1 / (1 + mean|u_a - u_b|)}. */
    private static double[][] diversityGradient(double[][] u, double[] cost) {
        int k = u.length;
        int p = u[0].length;
        double[][] g = new double[k][p];
        double base = determinant(u);
        double h = 1e-4;
        for (int a = 0; a < k; a++) {
            for (int j = 0; j < p; j++) {
                if (Double.isInfinite(cost[j])) {
                    continue;
                }
                double keep = u[a][j];
                u[a][j] = keep + h;
                g[a][j] = (determinant(u) - base) / h;
                u[a][j] = keep;
            }
        }
        return g;
    }

    static double determinant(double[][] u) {
        int k = u.length;
        double[][] kernel = new double[k][k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                double d = 0;
                for (int j = 0; j < u[a].length; j++) {
                    d += Math.abs(u[a][j] - u[b][j]);
                }
                kernel[a][b] = 1 / (1 + d / u[a].length);
            }
        }
        return io.github.zzuegg.brainiac.tensor.Matrices.determinant(kernel);
    }

    // ------------------------------------------------------------------ FACE

    /**
     * Feasible and actionable counterfactual explanations by a density-weighted shortest path.
     *
     * @param score the model's batch score
     * @param threshold the decision threshold
     * @param x the denied row
     * @param constraints the constraints (immutable features of the result are reset to {@code x})
     * @param data the data rows forming the graph
     * @param neighbours edges per node
     * @return the end point of the cheapest path to an accepted, dense data point, or null
     */
    public static double[] face(Function<double[][], double[]> score, double threshold, double[] x,
            Constraints constraints, double[][] data, int neighbours) {
        int n = data.length;
        int p = x.length;
        double[][] nodes = new double[n + 1][];
        nodes[0] = x;
        System.arraycopy(data, 0, nodes, 1, n);
        double[][] projected = new double[n + 1][];
        for (int i = 0; i <= n; i++) {
            projected[i] = nodes[i].clone();
            for (int j = 0; j < p; j++) {
                if (constraints.mutability()[j] == Mutability.IMMUTABLE) {
                    projected[i][j] = x[j];
                }
            }
        }
        double[] f = score.apply(projected);   // judged with the person's own immutable features
        double[] density = new double[n + 1];
        for (int i = 0; i <= n; i++) {
            density[i] = localDensity(nodes[i], data, constraints.scale(), 5);
        }
        double medianDensity = median(Arrays.copyOfRange(density, 1, n + 1));
        double[] dist = new double[n + 1];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        dist[0] = 0;
        boolean[] done = new boolean[n + 1];
        PriorityQueue<double[]> queue = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        queue.add(new double[] {0, 0});
        while (!queue.isEmpty()) {
            double[] top = queue.poll();
            int a = (int) top[1];
            if (done[a]) {
                continue;
            }
            done[a] = true;
            if (a > 0 && f[a] >= threshold + 0.1 && density[a] >= medianDensity
                    && actionableStep(x, projected[a], constraints, 0)) {
                return projected[a];
            }
            int[] nb = nearest(nodes, a, constraints.scale(), neighbours);
            for (int b : nb) {
                if (done[b] || !actionableStep(nodes[a], nodes[b], constraints, 0.25)) {
                    continue;
                }
                double len = 0;
                double[] mid = new double[p];
                for (int j = 0; j < p; j++) {
                    len += constraints.cost()[j] < Double.POSITIVE_INFINITY
                            ? constraints.cost()[j] * Math.abs(nodes[a][j] - nodes[b][j]) / constraints.scale()[j] : 0;
                    mid[j] = (nodes[a][j] + nodes[b][j]) / 2;
                }
                double w = len / Math.max(localDensity(mid, data, constraints.scale(), 5), 1e-9);
                if (dist[a] + w < dist[b]) {
                    dist[b] = dist[a] + w;
                    queue.add(new double[] {dist[b], b});
                }
            }
        }
        return null;
    }

    /**
     * A move is actionable when monotone features move against their direction by at most
     * {@code tolerance} scales and immutable features stay within half a scale. Graph edges use
     * a quarter scale (single edges between data points are noisy); the end point must respect
     * the directions exactly relative to the denied row.
     */
    private static boolean actionableStep(double[] a, double[] b, Constraints c, double tolerance) {
        for (int j = 0; j < a.length; j++) {
            double d = b[j] - a[j];
            switch (c.mutability()[j]) {
                case IMMUTABLE -> {
                    if (Math.abs(d) > 0.5 * c.scale()[j]) {
                        return false;
                    }
                }
                case INCREASE -> {
                    if (d < -tolerance * c.scale()[j]) {
                        return false;
                    }
                }
                case DECREASE -> {
                    if (d > tolerance * c.scale()[j]) {
                        return false;
                    }
                }
                case FREE -> {
                }
            }
        }
        return true;
    }

    private static int[] nearest(double[][] nodes, int a, double[] scale, int k) {
        int n = nodes.length;
        double[] d = new double[n];
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
            d[i] = i == a ? Double.POSITIVE_INFINITY : distance(nodes[a], nodes[i], scale);
        }
        Arrays.sort(idx, (p, q) -> Double.compare(d[p], d[q]));
        int[] out = new int[Math.min(k, n - 1)];
        for (int i = 0; i < out.length; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    static double distance(double[] a, double[] b, double[] scale) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            double d = (a[j] - b[j]) / scale[j];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    /**
     * A k-nearest-neighbour density proxy: one over the mean distance to the {@code k} nearest
     * data rows (in scale units).
     *
     * @param point the point
     * @param data the data
     * @param scale the feature scales
     * @param k the neighbour count
     * @return the density proxy, larger is denser
     */
    public static double localDensity(double[] point, double[][] data, double[] scale, int k) {
        double[] best = new double[k];
        Arrays.fill(best, Double.POSITIVE_INFINITY);
        for (double[] row : data) {
            double d = distance(point, row, scale);
            if (d < 1e-12) {
                continue;
            }
            if (d < best[k - 1]) {
                int i = k - 1;
                while (i > 0 && best[i - 1] > d) {
                    best[i] = best[i - 1];
                    i--;
                }
                best[i] = d;
            }
        }
        return 1 / Arrays.stream(best).average().orElse(1);
    }

    private static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }
}
