package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Projection-free and proximal methods for {@code min f(x)} over a convex set: Frank-Wolfe with
 * away and pairwise steps, projected gradient, FISTA and ADMM.
 *
 * <ul>
 *   <li>{@link Method#FRANK_WOLFE} (Frank and Wolfe 1956; Jaggi, ICML 2013): a linear minimisation
 *       oracle (LMO) {@code s = argmin_{s in C} <grad f(x), s>} replaces the projection, and
 *       {@code x <- x + gamma (s - x)}. Iterates are sparse (simplex, l1 ball) or low-rank
 *       (trace-norm ball: the LMO is the top singular pair) combinations of atoms;</li>
 *   <li>{@link Method#AWAY_STEP_FW} and {@link Method#PAIRWISE_FW} (Lacoste-Julien and Jaggi,
 *       NeurIPS 2015, <a href="https://arxiv.org/abs/1511.05932">arXiv:1511.05932</a>): keep the
 *       active atoms with their weights and also move away from the worst active atom (or shift
 *       weight from it to the FW atom), which gives linear convergence on polytopes;</li>
 *   <li>{@link Method#PROJECTED_GRADIENT} and {@link Method#FISTA} (Beck and Teboulle 2009):
 *       gradient steps of {@code 1 / L} followed by the Euclidean projection, the latter with
 *       Nesterov momentum;</li>
 *   <li>{@link Method#ADMM} (Boyd et al. 2011): splits {@code x = z}, alternates the proximal step
 *       of {@code f} with the projection onto the set, and ascends on the scaled dual.</li>
 * </ul>
 * FW variants use the short-step rule {@code gamma = min(gamma_max, -<g, d> / (L |d|^2))}. Sets:
 * the probability simplex, the l1 ball and (for FW, projected gradient, FISTA and ADMM) the
 * trace-norm ball of matrices, flattened row-major.
 */
public final class ConstrainedSolvers {

    /** The solvers. */
    public enum Method {
        /** Vanilla Frank-Wolfe. */
        FRANK_WOLFE,
        /** Frank-Wolfe with away steps. */
        AWAY_STEP_FW,
        /** Pairwise Frank-Wolfe. */
        PAIRWISE_FW,
        /** Projected gradient descent. */
        PROJECTED_GRADIENT,
        /** Accelerated projected gradient. */
        FISTA,
        /** Alternating direction method of multipliers. */
        ADMM
    }

    /** A smooth objective with a proximal operator for ADMM. */
    public interface Objective {

        /**
         * The value.
         *
         * @param x the point
         * @return {@code f(x)}
         */
        double value(double[] x);

        /**
         * The gradient.
         *
         * @param x the point
         * @return {@code grad f(x)}
         */
        double[] gradient(double[] x);

        /**
         * {@code argmin_x f(x) + rho/2 |x - v|^2}; only ADMM uses it.
         *
         * @param v the centre
         * @param rho the penalty
         * @return the proximal point
         */
        default double[] prox(double[] v, double rho) {
            throw new UnsupportedOperationException("no proximal operator");
        }
    }

    /** A feasible set. */
    public enum Domain {
        /** The probability simplex {@code x >= 0, sum x = radius}. */
        SIMPLEX,
        /** The l1 ball {@code |x|_1 <= radius}. */
        L1_BALL,
        /** The trace-norm ball {@code |X|_* <= radius} of {@code rows x cols} matrices. */
        TRACE_NORM_BALL
    }

    /**
     * A set with its size.
     *
     * @param domain the kind
     * @param radius the sum, l1 radius or trace-norm radius
     * @param rows for matrices, the row count (0 otherwise)
     * @param cols for matrices, the column count (0 otherwise)
     */
    public record Set(Domain domain, double radius, int rows, int cols) {

        /**
         * The simplex scaled to {@code radius}.
         *
         * @param radius the sum
         * @return the set
         */
        public static Set simplex(double radius) {
            return new Set(Domain.SIMPLEX, radius, 0, 0);
        }

        /**
         * The l1 ball.
         *
         * @param radius the radius
         * @return the set
         */
        public static Set l1Ball(double radius) {
            return new Set(Domain.L1_BALL, radius, 0, 0);
        }

        /**
         * The trace-norm ball.
         *
         * @param radius the radius
         * @param rows rows
         * @param cols columns
         * @return the set
         */
        public static Set traceNormBall(double radius, int rows, int cols) {
            return new Set(Domain.TRACE_NORM_BALL, radius, rows, cols);
        }
    }

    /**
     * A run.
     *
     * @param x the final point
     * @param objective {@code f} after every iteration (index 0 at the start)
     * @param millis wall time per iteration
     */
    public record Result(double[] x, double[] objective, double millis) {
    }

    private ConstrainedSolvers() {
    }

    /**
     * Least squares {@code 1/2 |A x - b|^2} with its proximal operator.
     *
     * @param a the design
     * @param b the targets
     * @return the objective
     */
    public static Objective leastSquares(double[][] a, double[] b) {
        int n = a[0].length;
        double[][] ata = new double[n][n];
        double[] atb = new double[n];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < n; j++) {
                atb[j] += a[i][j] * b[i];
                for (int k = 0; k < n; k++) {
                    ata[j][k] += a[i][j] * a[i][k];
                }
            }
        }
        return new Objective() {
            @Override
            public double value(double[] x) {
                double s = 0;
                for (int i = 0; i < a.length; i++) {
                    double r = -b[i];
                    for (int j = 0; j < n; j++) {
                        r += a[i][j] * x[j];
                    }
                    s += 0.5 * r * r;
                }
                return s;
            }

            @Override
            public double[] gradient(double[] x) {
                double[] g = new double[n];
                for (int j = 0; j < n; j++) {
                    double s = -atb[j];
                    for (int k = 0; k < n; k++) {
                        s += ata[j][k] * x[k];
                    }
                    g[j] = s;
                }
                return g;
            }

            @Override
            public double[] prox(double[] v, double rho) {
                double[][] m = new double[n][n];
                double[] rhs = new double[n];
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        m[j][k] = ata[j][k] + (j == k ? rho : 0);
                    }
                    rhs[j] = atb[j] + rho * v[j];
                }
                return QpLayer.luSolve(m, rhs);
            }
        };
    }

    /**
     * Matrix completion {@code 1/2 sum_{observed} (X_ij - Y_ij)^2} on a flattened matrix.
     *
     * @param observed per entry, whether it is observed
     * @param values the observed values (others ignored)
     * @return the objective
     */
    public static Objective maskedSquares(boolean[] observed, double[] values) {
        return new Objective() {
            @Override
            public double value(double[] x) {
                double s = 0;
                for (int i = 0; i < x.length; i++) {
                    if (observed[i]) {
                        s += 0.5 * (x[i] - values[i]) * (x[i] - values[i]);
                    }
                }
                return s;
            }

            @Override
            public double[] gradient(double[] x) {
                double[] g = new double[x.length];
                for (int i = 0; i < x.length; i++) {
                    g[i] = observed[i] ? x[i] - values[i] : 0;
                }
                return g;
            }

            @Override
            public double[] prox(double[] v, double rho) {
                double[] out = new double[v.length];
                for (int i = 0; i < v.length; i++) {
                    out[i] = observed[i] ? (values[i] + rho * v[i]) / (1 + rho) : v[i];
                }
                return out;
            }
        };
    }

    /**
     * Runs a solver from the set's centre (the uniform point, or zero), or for the l1 ball and
     * Frank-Wolfe from the midpoint of two opposite vertices.
     *
     * @param method the solver
     * @param f the objective
     * @param set the feasible set
     * @param lipschitz the gradient's Lipschitz constant
     * @param iterations iterations
     * @param dimension the vector length
     * @return the run
     */
    public static Result solve(Method method, Objective f, Set set, double lipschitz, int iterations, int dimension) {
        Objects.requireNonNull(method, "method");
        double[] objective = new double[iterations + 1];
        long t0 = System.nanoTime();
        double[] x;
        switch (method) {
            case FRANK_WOLFE, AWAY_STEP_FW, PAIRWISE_FW -> x = frankWolfe(method, f, set, lipschitz, iterations,
                    dimension, objective);
            case PROJECTED_GRADIENT, FISTA -> {
                x = start(set, dimension);
                double[] y = x.clone();
                double tk = 1;
                objective[0] = f.value(x);
                for (int it = 1; it <= iterations; it++) {
                    double[] base = method == Method.FISTA ? y : x;
                    double[] g = f.gradient(base);
                    double[] z = new double[dimension];
                    for (int i = 0; i < dimension; i++) {
                        z[i] = base[i] - g[i] / lipschitz;
                    }
                    double[] next = project(z, set);
                    if (method == Method.FISTA) {
                        double tn = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
                        for (int i = 0; i < dimension; i++) {
                            y[i] = next[i] + (tk - 1) / tn * (next[i] - x[i]);
                        }
                        tk = tn;
                    }
                    x = next;
                    objective[it] = f.value(x);
                }
            }
            case ADMM -> {
                double rho = lipschitz;
                x = start(set, dimension);
                double[] z = x.clone();
                double[] u = new double[dimension];
                objective[0] = f.value(z);
                for (int it = 1; it <= iterations; it++) {
                    double[] v = new double[dimension];
                    for (int i = 0; i < dimension; i++) {
                        v[i] = z[i] - u[i];
                    }
                    x = f.prox(v, rho);
                    double[] w = new double[dimension];
                    for (int i = 0; i < dimension; i++) {
                        w[i] = x[i] + u[i];
                    }
                    z = project(w, set);
                    for (int i = 0; i < dimension; i++) {
                        u[i] += x[i] - z[i];
                    }
                    objective[it] = f.value(z);
                }
                x = z;
            }
            default -> throw new IllegalStateException();
        }
        return new Result(x, objective, (System.nanoTime() - t0) / 1e6 / iterations);
    }

    private static double[] start(Set set, int d) {
        double[] x = new double[d];
        if (set.domain() == Domain.SIMPLEX) {
            Arrays.fill(x, set.radius() / d);
        }
        return x;
    }

    /** An atom: a vertex index with sign for the simplex / l1 ball, or a rank-one matrix. */
    private record Atom(int index, double sign, double[] u, double[] v) {
        double[] dense(Set set, int d) {
            double[] a = new double[d];
            if (u == null) {
                a[index] = sign * set.radius();
            } else {
                int cols = set.cols();
                for (int i = 0; i < u.length; i++) {
                    for (int j = 0; j < cols; j++) {
                        a[i * cols + j] = set.radius() * u[i] * v[j];
                    }
                }
            }
            return a;
        }

        boolean same(Atom o) {
            return u == null && o.u == null && index == o.index && sign == o.sign;
        }
    }

    private static Atom lmo(double[] g, Set set) {
        switch (set.domain()) {
            case SIMPLEX -> {
                int best = 0;
                for (int i = 1; i < g.length; i++) {
                    if (g[i] < g[best]) {
                        best = i;
                    }
                }
                return new Atom(best, 1, null, null);
            }
            case L1_BALL -> {
                int best = 0;
                for (int i = 1; i < g.length; i++) {
                    if (Math.abs(g[i]) > Math.abs(g[best])) {
                        best = i;
                    }
                }
                return new Atom(best, g[best] > 0 ? -1 : 1, null, null);
            }
            default -> {
                // the top singular pair of -G by power iteration
                int rows = set.rows();
                int cols = set.cols();
                double[] v = new double[cols];
                SplittableRandom r = new SplittableRandom(17);
                for (int j = 0; j < cols; j++) {
                    v[j] = r.nextGaussian();
                }
                double[] u = new double[rows];
                for (int it = 0; it < 30; it++) {
                    Arrays.fill(u, 0);
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            u[i] -= g[i * cols + j] * v[j];
                        }
                    }
                    normalise(u);
                    Arrays.fill(v, 0);
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            v[j] -= g[i * cols + j] * u[i];
                        }
                    }
                    normalise(v);
                }
                return new Atom(-1, 1, u, v);
            }
        }
    }

    private static void normalise(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v * v;
        }
        s = Math.sqrt(s) + 1e-300;
        for (int i = 0; i < x.length; i++) {
            x[i] /= s;
        }
    }

    private static double[] frankWolfe(Method method, Objective f, Set set, double lipschitz, int iterations, int d,
            double[] objective) {
        boolean matrix = set.domain() == Domain.TRACE_NORM_BALL;
        if (matrix && method != Method.FRANK_WOLFE) {
            throw new UnsupportedOperationException("away and pairwise steps need a finite atom set");
        }
        List<Atom> atoms = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        double[] x = new double[d];
        if (set.domain() == Domain.SIMPLEX) {
            Atom first = new Atom(0, 1, null, null);
            atoms.add(first);
            weights.add(1.0);
            x = first.dense(set, d);
        } else if (set.domain() == Domain.L1_BALL) {
            // start at the centre: the origin as the midpoint of +-e_0
            atoms.add(new Atom(0, 1, null, null));
            weights.add(0.5);
            atoms.add(new Atom(0, -1, null, null));
            weights.add(0.5);
        }
        objective[0] = f.value(x);
        for (int it = 1; it <= iterations; it++) {
            double[] g = f.gradient(x);
            Atom s = lmo(g, set);
            double[] sd = s.dense(set, d);
            double[] dir = new double[d];
            double gamMax = 1;
            int awayIdx = -1;
            boolean awayStep = false;
            if (method != Method.FRANK_WOLFE) {
                double worst = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < atoms.size(); k++) {
                    double val = dot(g, atoms.get(k).dense(set, d));
                    if (val > worst) {
                        worst = val;
                        awayIdx = k;
                    }
                }
            }
            double[] ad = awayIdx >= 0 ? atoms.get(awayIdx).dense(set, d) : null;
            if (method == Method.PAIRWISE_FW) {
                for (int i = 0; i < d; i++) {
                    dir[i] = sd[i] - ad[i];
                }
                gamMax = weights.get(awayIdx);
            } else if (method == Method.AWAY_STEP_FW) {
                double fwGap = 0;
                double awayGap = 0;
                for (int i = 0; i < d; i++) {
                    fwGap += -g[i] * (sd[i] - x[i]);
                    awayGap += -g[i] * (x[i] - ad[i]);
                }
                if (fwGap >= awayGap) {
                    for (int i = 0; i < d; i++) {
                        dir[i] = sd[i] - x[i];
                    }
                } else {
                    awayStep = true;
                    double alpha = weights.get(awayIdx);
                    for (int i = 0; i < d; i++) {
                        dir[i] = x[i] - ad[i];
                    }
                    gamMax = alpha / (1 - alpha + 1e-300);
                }
            } else {
                for (int i = 0; i < d; i++) {
                    dir[i] = sd[i] - x[i];
                }
            }
            double gd = dot(g, dir);
            double dd = dot(dir, dir);
            if (dd < 1e-300 || gd >= 0) {
                objective[it] = objective[it - 1];
                continue;
            }
            double gamma = Math.min(gamMax, -gd / (lipschitz * dd));
            for (int i = 0; i < d; i++) {
                x[i] += gamma * dir[i];
            }
            if (!matrix) {
                updateWeights(method, atoms, weights, s, awayIdx, awayStep, gamma);
            }
            objective[it] = f.value(x);
        }
        return x;
    }

    private static void updateWeights(Method method, List<Atom> atoms, List<Double> weights, Atom s, int awayIdx,
            boolean awayStep, double gamma) {
        int sIdx = -1;
        for (int k = 0; k < atoms.size(); k++) {
            if (atoms.get(k).same(s)) {
                sIdx = k;
            }
        }
        if (method == Method.PAIRWISE_FW) {
            if (sIdx < 0) {
                atoms.add(s);
                weights.add(0.0);
                sIdx = atoms.size() - 1;
            }
            weights.set(sIdx, weights.get(sIdx) + gamma);
            weights.set(awayIdx, weights.get(awayIdx) - gamma);
        } else if (awayStep) {
            for (int k = 0; k < weights.size(); k++) {
                weights.set(k, weights.get(k) * (1 + gamma));
            }
            weights.set(awayIdx, weights.get(awayIdx) - gamma);
        } else {
            for (int k = 0; k < weights.size(); k++) {
                weights.set(k, weights.get(k) * (1 - gamma));
            }
            if (sIdx < 0) {
                atoms.add(s);
                weights.add(gamma);
            } else {
                weights.set(sIdx, weights.get(sIdx) + gamma);
            }
        }
        for (int k = weights.size() - 1; k >= 0; k--) {
            if (weights.get(k) <= 1e-12) {
                weights.remove(k);
                atoms.remove(k);
            }
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * The Euclidean projection onto a set.
     *
     * @param z the point
     * @param set the set
     * @return the projection
     */
    public static double[] project(double[] z, Set set) {
        return switch (set.domain()) {
            case SIMPLEX -> projectSimplex(z, set.radius());
            case L1_BALL -> {
                double l1 = 0;
                for (double v : z) {
                    l1 += Math.abs(v);
                }
                if (l1 <= set.radius()) {
                    yield z.clone();
                }
                double[] abs = new double[z.length];
                for (int i = 0; i < z.length; i++) {
                    abs[i] = Math.abs(z[i]);
                }
                double[] p = projectSimplex(abs, set.radius());
                for (int i = 0; i < z.length; i++) {
                    p[i] *= Math.signum(z[i]);
                }
                yield p;
            }
            case TRACE_NORM_BALL -> {
                int rows = set.rows();
                int cols = set.cols();
                double[][] m = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(z, i * cols, m[i], 0, cols);
                }
                ThinSvd svd = ThinSvd.of(m);
                double total = 0;
                for (double s : svd.s()) {
                    total += s;
                }
                if (total <= set.radius()) {
                    yield z.clone();
                }
                double[] s = projectSimplex(svd.s(), set.radius());
                double[] out = new double[rows * cols];
                for (int k = 0; k < s.length; k++) {
                    if (s[k] == 0) {
                        continue;
                    }
                    for (int i = 0; i < rows; i++) {
                        double us = svd.u()[i][k] * s[k];
                        for (int j = 0; j < cols; j++) {
                            out[i * cols + j] += us * svd.v()[j][k];
                        }
                    }
                }
                yield out;
            }
        };
    }

    /**
     * The projection onto {@code {x >= 0, sum x = r}} by sorting (Held, Wolfe and Crowder 1974).
     *
     * @param z the point
     * @param r the sum
     * @return the projection
     */
    public static double[] projectSimplex(double[] z, double r) {
        double[] s = z.clone();
        Arrays.sort(s);
        double cum = 0;
        double theta = 0;
        for (int i = s.length - 1, k = 1; i >= 0; i--, k++) {
            cum += s[i];
            double t = (cum - r) / k;
            if (s[i] - t > 0) {
                theta = t;
            }
        }
        double[] out = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            out[i] = Math.max(0, z[i] - theta);
        }
        return out;
    }

    /**
     * The number of non-zero entries (above {@code 1e-8} in magnitude).
     *
     * @param x the vector
     * @return the support size
     */
    public static int support(double[] x) {
        int c = 0;
        for (double v : x) {
            c += Math.abs(v) > 1e-8 ? 1 : 0;
        }
        return c;
    }
}
