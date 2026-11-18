package io.github.zzuegg.brainiac.experimental.randla;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Sketch-and-precondition least squares (Rokhlin and Tygert 2008; Avron, Maymounkov and Toledo,
 * "Blendenpik", SIAM J. Sci. Comput. 2010; Meng, Saunders and Mahoney, "LSRN", 2014).
 *
 * <p>For a tall {@code A [m][n]}, a random sketch {@code S A} with {@code s ~ 2-4 n} rows keeps
 * the column geometry of {@code A} up to a constant distortion. Its QR factor {@code R} therefore
 * makes {@code A R^-1} well conditioned whatever the condition number of {@code A}, and LSQR
 * (Paige and Saunders 1982) on {@code A R^-1} converges to machine-precision in a number of
 * iterations that depends only on the sketch's distortion. The solution is {@code x = R^-1 y}.
 * Plain LSQR on {@code A} needs {@code O(kappa)} iterations, and the normal equations square the
 * condition number, so both fail at {@code kappa = 10^8}.
 *
 * <p>Sketches: dense Gaussian, CountSketch (each row of {@code A} lands in one random bucket with
 * a random sign: {@code O(nnz(A))}) and the subsampled randomised Hadamard transform (random
 * signs, a fast Walsh-Hadamard transform, row sampling).
 */
public final class SketchedLeastSquares {

    /** The sketching distribution. */
    public enum Sketch {
        /** Dense Gaussian sketch. */
        GAUSSIAN,
        /** CountSketch (Clarkson-Woodruff). */
        COUNT_SKETCH,
        /** Subsampled randomised Hadamard transform. */
        SRHT
    }

    /** A linear operator and its adjoint. */
    public interface Operator {

        /**
         * The number of rows.
         *
         * @return rows
         */
        int rows();

        /**
         * The number of columns.
         *
         * @return columns
         */
        int columns();

        /**
         * {@code A x}.
         *
         * @param x a vector of length {@link #columns()}
         * @return the product
         */
        double[] apply(double[] x);

        /**
         * {@code A^T y}.
         *
         * @param y a vector of length {@link #rows()}
         * @return the product
         */
        double[] applyTransposed(double[] y);

        /**
         * A dense matrix as an operator.
         *
         * @param a the matrix
         * @return the operator
         */
        static Operator of(double[][] a) {
            return new Operator() {
                @Override
                public int rows() {
                    return a.length;
                }

                @Override
                public int columns() {
                    return a[0].length;
                }

                @Override
                public double[] apply(double[] x) {
                    double[] y = new double[a.length];
                    for (int i = 0; i < a.length; i++) {
                        double s = 0;
                        for (int j = 0; j < x.length; j++) {
                            s += a[i][j] * x[j];
                        }
                        y[i] = s;
                    }
                    return y;
                }

                @Override
                public double[] applyTransposed(double[] y) {
                    double[] x = new double[a[0].length];
                    for (int i = 0; i < a.length; i++) {
                        double v = y[i];
                        for (int j = 0; j < x.length; j++) {
                            x[j] += a[i][j] * v;
                        }
                    }
                    return x;
                }
            };
        }
    }

    /**
     * A solve.
     *
     * @param x the solution
     * @param iterations LSQR iterations used
     * @param converged whether the stopping test was met
     */
    public record Result(double[] x, int iterations, boolean converged) {
    }

    private SketchedLeastSquares() {
    }

    /**
     * The sketch {@code S A}.
     *
     * @param a the matrix {@code [m][n]}
     * @param rows sketch rows {@code s}
     * @param sketch the family
     * @param random the random source
     * @return {@code S A} as {@code [s][n]}
     */
    public static double[][] sketch(double[][] a, int rows, Sketch sketch, SplittableRandom random) {
        int m = a.length;
        int n = a[0].length;
        double[][] out = new double[rows][n];
        switch (Objects.requireNonNull(sketch, "sketch")) {
            case GAUSSIAN -> {
                double scale = 1 / Math.sqrt(rows);
                for (int i = 0; i < m; i++) {
                    double[] ai = a[i];
                    for (int r = 0; r < rows; r++) {
                        double g = random.nextGaussian() * scale;
                        double[] o = out[r];
                        for (int j = 0; j < n; j++) {
                            o[j] += g * ai[j];
                        }
                    }
                }
            }
            case COUNT_SKETCH -> {
                for (int i = 0; i < m; i++) {
                    double[] o = out[random.nextInt(rows)];
                    double sign = random.nextBoolean() ? 1 : -1;
                    for (int j = 0; j < n; j++) {
                        o[j] += sign * a[i][j];
                    }
                }
            }
            case SRHT -> {
                int p = Integer.highestOneBit(Math.max(1, m - 1)) << 1;
                if (p < m) {
                    p <<= 1;
                }
                double[] col = new double[p];
                int[] pick = new int[rows];
                for (int r = 0; r < rows; r++) {
                    pick[r] = random.nextInt(p);
                }
                double[] sign = new double[m];
                for (int i = 0; i < m; i++) {
                    sign[i] = random.nextBoolean() ? 1 : -1;
                }
                double scale = 1 / Math.sqrt(rows);
                for (int j = 0; j < n; j++) {
                    java.util.Arrays.fill(col, 0);
                    for (int i = 0; i < m; i++) {
                        col[i] = sign[i] * a[i][j];
                    }
                    walshHadamard(col);
                    for (int r = 0; r < rows; r++) {
                        out[r][j] = col[pick[r]] * scale;
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
        return out;
    }

    /**
     * The in-place unnormalised fast Walsh-Hadamard transform of a power-of-two-length vector.
     *
     * @param x the vector
     */
    public static void walshHadamard(double[] x) {
        int n = x.length;
        for (int h = 1; h < n; h <<= 1) {
            for (int i = 0; i < n; i += h << 1) {
                for (int j = i; j < i + h; j++) {
                    double u = x[j];
                    double v = x[j + h];
                    x[j] = u + v;
                    x[j + h] = u - v;
                }
            }
        }
    }

    /**
     * Sketch-and-precondition: QR of a sketch, then LSQR on {@code A R^-1}.
     *
     * @param a the tall matrix {@code [m][n]}
     * @param b the right-hand side {@code [m]}
     * @param sketchRows sketch rows, for example {@code 4 n}
     * @param sketch the sketch family
     * @param tolerance LSQR's relative tolerance on {@code |A^T r|}, for example 1e-12
     * @param maxIterations the iteration cap
     * @param seed the seed
     * @return the solution and iteration count
     */
    public static Result solve(double[][] a, double[] b, int sketchRows, Sketch sketch, double tolerance,
            int maxIterations, long seed) {
        double[][] r = HouseholderQr.of(sketch(a, sketchRows, sketch, new SplittableRandom(seed))).r();
        Operator base = Operator.of(a);
        Operator pre = new Operator() {
            @Override
            public int rows() {
                return base.rows();
            }

            @Override
            public int columns() {
                return base.columns();
            }

            @Override
            public double[] apply(double[] y) {
                return base.apply(HouseholderQr.solveUpper(r, y));
            }

            @Override
            public double[] applyTransposed(double[] y) {
                return HouseholderQr.solveUpperTransposed(r, base.applyTransposed(y));
            }
        };
        Result inner = lsqr(pre, b, tolerance, maxIterations);
        return new Result(HouseholderQr.solveUpper(r, inner.x()), inner.iterations(), inner.converged());
    }

    /**
     * LSQR (Paige and Saunders 1982) for {@code min |A x - b|}, stopping when
     * {@code |A^T r| <= tol |A|_est |r|}.
     *
     * @param op the operator
     * @param b the right-hand side
     * @param tolerance the relative tolerance
     * @param maxIterations the cap
     * @return the solution
     */
    public static Result lsqr(Operator op, double[] b, double tolerance, int maxIterations) {
        int n = op.columns();
        double[] u = b.clone();
        double beta = norm(u);
        scale(u, 1 / beta);
        double[] v = op.applyTransposed(u);
        double alpha = norm(v);
        scale(v, 1 / alpha);
        double[] w = v.clone();
        double[] x = new double[n];
        double phiBar = beta;
        double rhoBar = alpha;
        double anorm2 = 0;
        for (int it = 1; it <= maxIterations; it++) {
            double[] av = op.apply(v);
            for (int i = 0; i < u.length; i++) {
                u[i] = av[i] - alpha * u[i];
            }
            beta = norm(u);
            if (beta > 0) {
                scale(u, 1 / beta);
            }
            anorm2 += alpha * alpha + beta * beta;
            double[] atu = op.applyTransposed(u);
            for (int j = 0; j < n; j++) {
                v[j] = atu[j] - beta * v[j];
            }
            alpha = norm(v);
            if (alpha > 0) {
                scale(v, 1 / alpha);
            }
            double rho = Math.hypot(rhoBar, beta);
            double c = rhoBar / rho;
            double s = beta / rho;
            double theta = s * alpha;
            rhoBar = -c * alpha;
            double phi = c * phiBar;
            phiBar = s * phiBar;
            for (int j = 0; j < n; j++) {
                x[j] += phi / rho * w[j];
                w[j] = v[j] - theta / rho * w[j];
            }
            // |A^T r| = phiBar * alpha * |c|
            double arNorm = phiBar * alpha * Math.abs(c);
            if (arNorm <= tolerance * Math.sqrt(anorm2) * Math.max(phiBar, 1e-300) || phiBar == 0) {
                return new Result(x, it, true);
            }
        }
        return new Result(x, maxIterations, false);
    }

    /**
     * The normal-equations solution {@code (A^T A)^-1 A^T b} by Cholesky (with a tiny jitter if
     * the Gram matrix is numerically singular): the classical fast baseline that squares the
     * condition number.
     *
     * @param a the matrix
     * @param b the right-hand side
     * @return the solution
     */
    public static double[] normalEquations(double[][] a, double[] b) {
        int n = a[0].length;
        double[][] g = RandomizedSvd.multiplyTransposed(a, a);
        double[] rhs = Operator.of(a).applyTransposed(b);
        double jitter = 0;
        for (int attempt = 0; attempt < 30; attempt++) {
            double[][] l = new double[n][n];
            boolean ok = true;
            for (int i = 0; i < n && ok; i++) {
                for (int j = 0; j <= i; j++) {
                    double s = g[i][j] + (i == j ? jitter : 0);
                    for (int k = 0; k < j; k++) {
                        s -= l[i][k] * l[j][k];
                    }
                    if (i == j) {
                        if (s <= 0) {
                            ok = false;
                            break;
                        }
                        l[i][i] = Math.sqrt(s);
                    } else {
                        l[i][j] = s / l[j][j];
                    }
                }
            }
            if (ok) {
                double[] y = new double[n];
                for (int i = 0; i < n; i++) {
                    double s = rhs[i];
                    for (int k = 0; k < i; k++) {
                        s -= l[i][k] * y[k];
                    }
                    y[i] = s / l[i][i];
                }
                double[] x = new double[n];
                for (int i = n - 1; i >= 0; i--) {
                    double s = y[i];
                    for (int k = i + 1; k < n; k++) {
                        s -= l[k][i] * x[k];
                    }
                    x[i] = s / l[i][i];
                }
                return x;
            }
            jitter = jitter == 0 ? 1e-14 * trace(g) / n : jitter * 10;
        }
        throw new ArithmeticException("normal equations failed");
    }

    private static double trace(double[][] g) {
        double t = 0;
        for (int i = 0; i < g.length; i++) {
            t += g[i][i];
        }
        return t;
    }

    static double norm(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v * v;
        }
        return Math.sqrt(s);
    }

    private static void scale(double[] x, double c) {
        for (int i = 0; i < x.length; i++) {
            x[i] *= c;
        }
    }
}
