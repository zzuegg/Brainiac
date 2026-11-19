package io.github.zzuegg.brainiac.tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Dense linear algebra on small {@code double[][]} matrices (rows first), for code that works
 * on the host: LU, QR and Cholesky factorizations, solving, determinants and inverses,
 * symmetric and general eigenvalues, the matrix exponential, and a Lanczos solver for a few
 * eigenpairs of a large symmetric operator. For batched work on a device, use {@link Linalg}.
 */
public final class Matrices {

    private Matrices() {
    }

    /** Applies a symmetric linear operator: {@code out = A in}. */
    @FunctionalInterface
    public interface Operator {

        /**
         * @param in the vector
         * @param out receives {@code A in}
         */
        void apply(double[] in, double[] out);

        /**
         * A dense matrix as an operator.
         *
         * @param a the matrix, rows of equal length
         * @return {@code x -> A x}
         */
        static Operator of(double[][] a) {
            return (in, out) -> {
                for (int i = 0; i < a.length; i++) {
                    double s = 0;
                    for (int j = 0; j < in.length; j++) {
                        s += a[i][j] * in[j];
                    }
                    out[i] = s;
                }
            };
        }

        /**
         * A diagonal matrix as an operator, such as a Jacobi preconditioner {@code 1 / diag(A)}.
         *
         * @param d the diagonal
         * @return {@code x -> d * x}
         */
        static Operator diagonal(double[] d) {
            return (in, out) -> {
                for (int i = 0; i < d.length; i++) {
                    out[i] = d[i] * in[i];
                }
            };
        }
    }

    // ------------------------------------------------------------------ LU

    /**
     * An LU factorization with partial pivoting: {@code P A = L U}, both stored in {@code lu}.
     *
     * @param lu {@code L} below the diagonal (its unit diagonal implied) and {@code U} on and above
     * @param pivots the row of {@code A} in each row of {@code P A}
     * @param sign the determinant of {@code P}
     */
    public record Lu(double[][] lu, int[] pivots, int sign) {

        /**
         * Solves {@code A x = b}.
         *
         * @param b the right-hand side
         * @return {@code x}
         */
        public double[] solve(double[] b) {
            int n = lu.length;
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = b[pivots[i]];
            }
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < i; k++) {
                    x[i] -= lu[i][k] * x[k];
                }
            }
            for (int i = n - 1; i >= 0; i--) {
                for (int k = i + 1; k < n; k++) {
                    x[i] -= lu[i][k] * x[k];
                }
                x[i] /= lu[i][i];
            }
            return x;
        }

        /** @return the determinant of {@code A} */
        public double determinant() {
            double d = sign;
            for (int i = 0; i < lu.length; i++) {
                d *= lu[i][i];
            }
            return d;
        }
    }

    /**
     * Factorizes a square matrix.
     *
     * @param a the matrix, not changed
     * @return the factorization
     * @throws IllegalArgumentException if it is singular
     */
    public static Lu lu(double[][] a) {
        int n = square(a);
        double[][] m = copy(a);
        int[] piv = new int[n];
        for (int i = 0; i < n; i++) {
            piv[i] = i;
        }
        int sign = 1;
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(m[r][c]) > Math.abs(m[p][c])) {
                    p = r;
                }
            }
            if (m[p][c] == 0) {
                throw new IllegalArgumentException("the matrix is singular");
            }
            if (p != c) {
                double[] t = m[p];
                m[p] = m[c];
                m[c] = t;
                int ti = piv[p];
                piv[p] = piv[c];
                piv[c] = ti;
                sign = -sign;
            }
            for (int r = c + 1; r < n; r++) {
                double f = m[r][c] / m[c][c];
                m[r][c] = f;
                for (int k = c + 1; k < n; k++) {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
        return new Lu(m, piv, sign);
    }

    /**
     * @param a a square matrix
     * @param b the right-hand side
     * @return {@code x} with {@code A x = b}
     */
    public static double[] solve(double[][] a, double[] b) {
        return lu(a).solve(b);
    }

    /**
     * @param a a square matrix
     * @return its determinant, 0 when singular
     */
    public static double determinant(double[][] a) {
        try {
            return lu(a).determinant();
        } catch (IllegalArgumentException e) {
            return 0;
        }
    }

    /**
     * @param a a square, invertible matrix
     * @return its inverse
     */
    public static double[][] inverse(double[][] a) {
        Lu f = lu(a);
        int n = a.length;
        double[][] out = new double[n][n];
        double[] e = new double[n];
        for (int j = 0; j < n; j++) {
            Arrays.fill(e, 0);
            e[j] = 1;
            double[] col = f.solve(e);
            for (int i = 0; i < n; i++) {
                out[i][j] = col[i];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ QR and Cholesky

    /**
     * A QR factorization {@code A = Q R} of an {@code m x n} matrix with {@code m >= n}: the
     * thin one, {@code Q} {@code m x n} with orthonormal columns and {@code R} {@code n x n}
     * upper triangular.
     *
     * @param q the orthonormal factor
     * @param r the triangular factor
     */
    public record Qr(double[][] q, double[][] r) {
    }

    /**
     * Factorizes by Householder reflections.
     *
     * @param a the matrix, {@code m >= n}, not changed
     * @return the thin factorization
     */
    public static Qr qr(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (m < n) {
            throw new IllegalArgumentException("qr needs at least as many rows as columns, got " + m + "x" + n);
        }
        double[][] r = copy(a);
        double[][] vs = new double[n][];
        for (int k = 0; k < n; k++) {
            double norm = 0;
            for (int i = k; i < m; i++) {
                norm += r[i][k] * r[i][k];
            }
            norm = Math.sqrt(norm);
            double[] v = new double[m];
            if (norm == 0) {
                vs[k] = v;
                continue;
            }
            double alpha = r[k][k] > 0 ? -norm : norm;
            for (int i = k; i < m; i++) {
                v[i] = r[i][k];
            }
            v[k] -= alpha;
            double vv = 0;
            for (int i = k; i < m; i++) {
                vv += v[i] * v[i];
            }
            if (vv > 0) {
                for (int j = k; j < n; j++) {
                    double s = 0;
                    for (int i = k; i < m; i++) {
                        s += v[i] * r[i][j];
                    }
                    s = 2 * s / vv;
                    for (int i = k; i < m; i++) {
                        r[i][j] -= s * v[i];
                    }
                }
                double scale = Math.sqrt(vv);
                for (int i = k; i < m; i++) {
                    v[i] /= scale;
                }
            }
            vs[k] = v;
        }
        // Q = H_0 H_1 ... H_{n-1} applied to the first n columns of the identity
        double[][] q = new double[m][n];
        for (int j = 0; j < n; j++) {
            q[j][j] = 1;
        }
        for (int k = n - 1; k >= 0; k--) {
            double[] v = vs[k];
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = k; i < m; i++) {
                    s += v[i] * q[i][j];
                }
                for (int i = k; i < m; i++) {
                    q[i][j] -= 2 * s * v[i];
                }
            }
        }
        double[][] rr = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                rr[i][j] = r[i][j];
            }
        }
        return new Qr(q, rr);
    }

    /**
     * The Cholesky factor of a symmetric positive-definite matrix.
     *
     * @param a the matrix
     * @return {@code L}, lower triangular, with {@code A = L L^T}
     * @throws IllegalArgumentException if it is not positive definite
     */
    public static double[][] cholesky(double[][] a) {
        int n = square(a);
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = a[i][j];
                for (int k = 0; k < j; k++) {
                    s -= l[i][k] * l[j][k];
                }
                if (i == j) {
                    if (!(s > 0)) {
                        throw new IllegalArgumentException("the matrix is not positive definite");
                    }
                    l[i][i] = Math.sqrt(s);
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        return l;
    }

    // ------------------------------------------------------------------ eigenvalues

    /**
     * Eigenvalues and eigenvectors of a symmetric matrix.
     *
     * @param values ascending
     * @param vectors one eigenvector per column, in the same order
     */
    public record SymmetricEigen(double[] values, double[][] vectors) {
    }

    /**
     * The eigen decomposition of a symmetric matrix by cyclic Jacobi rotations: accurate,
     * for matrices up to a few hundred rows.
     *
     * @param a a symmetric matrix
     * @return the decomposition
     */
    public static SymmetricEigen symmetricEigen(double[][] a) {
        int n = square(a);
        double[][] m = copy(a);
        double[][] v = identity(n);
        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    off += m[i][j] * m[i][j];
                }
            }
            if (off < 1e-30) {
                break;
            }
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(m[p][q]) < 1e-300) {
                        continue;
                    }
                    double theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    if (theta == 0) {
                        t = 1;
                    }
                    double c = 1 / Math.sqrt(t * t + 1);
                    double s = t * c;
                    for (int k = 0; k < n; k++) {
                        double mkp = m[k][p];
                        double mkq = m[k][q];
                        m[k][p] = c * mkp - s * mkq;
                        m[k][q] = s * mkp + c * mkq;
                    }
                    for (int k = 0; k < n; k++) {
                        double mpk = m[p][k];
                        double mqk = m[q][k];
                        m[p][k] = c * mpk - s * mqk;
                        m[q][k] = s * mpk + c * mqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        double[][] mm = m;
        Arrays.sort(order, (x, y) -> Double.compare(mm[x][x], mm[y][y]));
        double[] values = new double[n];
        double[][] vectors = new double[n][n];
        for (int j = 0; j < n; j++) {
            values[j] = m[order[j]][order[j]];
            for (int i = 0; i < n; i++) {
                vectors[i][j] = v[i][order[j]];
            }
        }
        return new SymmetricEigen(values, vectors);
    }

    /**
     * The eigenvalues of a general square matrix, by reduction to Hessenberg form and the
     * shifted QR algorithm in complex arithmetic.
     *
     * @param a the matrix
     * @return {@code {real parts, imaginary parts}}, sorted by real part, then imaginary part
     */
    public static double[][] eigenvalues(double[][] a) {
        int n = square(a);
        double[][] hr = hessenberg(a);
        double[][] hi = new double[n][n];
        double[] re = new double[n];
        double[] im = new double[n];
        int hiEnd = n - 1;
        int iterations = 0;
        while (hiEnd >= 0) {
            if (hiEnd == 0) {
                re[0] = hr[0][0];
                im[0] = hi[0][0];
                break;
            }
            // find the active block [lo, hiEnd]
            int lo = hiEnd;
            while (lo > 0) {
                double sub = Math.hypot(hr[lo][lo - 1], hi[lo][lo - 1]);
                double diag = Math.hypot(hr[lo][lo], hi[lo][lo]) + Math.hypot(hr[lo - 1][lo - 1], hi[lo - 1][lo - 1]);
                if (sub <= 1e-15 * Math.max(diag, 1e-300)) {
                    hr[lo][lo - 1] = 0;
                    hi[lo][lo - 1] = 0;
                    break;
                }
                lo--;
            }
            if (lo == hiEnd) {
                re[hiEnd] = hr[hiEnd][hiEnd];
                im[hiEnd] = hi[hiEnd][hiEnd];
                hiEnd--;
                iterations = 0;
                continue;
            }
            if (++iterations > 1000) {
                throw new ArithmeticException("the QR iteration did not converge");
            }
            // Wilkinson shift: the eigenvalue of the trailing 2x2 block nearest its last entry
            double ar = hr[hiEnd - 1][hiEnd - 1];
            double ai = hi[hiEnd - 1][hiEnd - 1];
            double br = hr[hiEnd - 1][hiEnd];
            double bi = hi[hiEnd - 1][hiEnd];
            double cr = hr[hiEnd][hiEnd - 1];
            double ci = hi[hiEnd][hiEnd - 1];
            double dr = hr[hiEnd][hiEnd];
            double di = hi[hiEnd][hiEnd];
            double[] mu;
            if (iterations % 11 == 10) {
                mu = new double[] {dr + Math.abs(cr) * 0.75, di + 0.4375 * Math.abs(cr)};   // exceptional shift
            } else {
                double tr = (ar + dr) / 2;
                double ti = (ai + di) / 2;
                double hr2 = (ar - dr) / 2;
                double hi2 = (ai - di) / 2;
                // disc = ((a - d) / 2)^2 + b c
                double discR = hr2 * hr2 - hi2 * hi2 + br * cr - bi * ci;
                double discI = 2 * hr2 * hi2 + br * ci + bi * cr;
                double[] sq = csqrt(discR, discI);
                double[] m1 = {tr + sq[0], ti + sq[1]};
                double[] m2 = {tr - sq[0], ti - sq[1]};
                mu = Math.hypot(m1[0] - dr, m1[1] - di) < Math.hypot(m2[0] - dr, m2[1] - di) ? m1 : m2;
            }
            qrStep(hr, hi, lo, hiEnd, mu[0], mu[1]);
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
            if (Math.abs(im[i]) < 1e-12 * Math.max(1, Math.abs(re[i]))) {
                im[i] = 0;
            }
        }
        Arrays.sort(order, (x, y) -> re[x] != re[y] ? Double.compare(re[x], re[y]) : Double.compare(im[x], im[y]));
        double[][] out = new double[2][n];
        for (int i = 0; i < n; i++) {
            out[0][i] = re[order[i]];
            out[1][i] = im[order[i]];
        }
        return out;
    }

    // one shifted QR step on the block [lo, hi] of a complex Hessenberg matrix, by Givens rotations
    private static void qrStep(double[][] hr, double[][] hi, int lo, int end, double mr, double mi) {
        int n = hr.length;
        for (int i = lo; i <= end; i++) {
            hr[i][i] -= mr;
            hi[i][i] -= mi;
        }
        int count = end - lo;
        double[][] rot = new double[count][4];   // c (real), s (complex: re, im)
        for (int k = lo; k < end; k++) {
            double xr = hr[k][k];
            double xi = hi[k][k];
            double yr = hr[k + 1][k];
            double yi = hi[k + 1][k];
            double norm = Math.sqrt(xr * xr + xi * xi + yr * yr + yi * yi);
            double c;
            double sr;
            double si;
            double ax = Math.hypot(xr, xi);
            if (norm == 0) {
                c = 1;
                sr = 0;
                si = 0;
            } else if (ax == 0) {
                c = 0;
                // s = conj(y) / |y|
                double ay = Math.hypot(yr, yi);
                sr = yr / ay;
                si = -yi / ay;
            } else {
                c = ax / norm;
                // s = (x / |x|) conj(y) / norm
                double ur = xr / ax;
                double ui = xi / ax;
                sr = (ur * yr + ui * yi) / norm;
                si = (ui * yr - ur * yi) / norm;
            }
            rot[k - lo] = new double[] {c, sr, si};
            // rows k and k+1: [c s; -conj(s) c]
            for (int j = k; j < n; j++) {
                double ar = hr[k][j];
                double ai = hi[k][j];
                double br = hr[k + 1][j];
                double bi = hi[k + 1][j];
                hr[k][j] = c * ar + sr * br - si * bi;
                hi[k][j] = c * ai + sr * bi + si * br;
                hr[k + 1][j] = -(sr * ar + si * ai) + c * br;
                hi[k + 1][j] = -(sr * ai - si * ar) + c * bi;
            }
        }
        for (int k = lo; k < end; k++) {
            double[] g = rot[k - lo];
            double c = g[0];
            double sr = g[1];
            double si = g[2];
            // columns k and k+1 times the conjugate transpose: [c -s; conj(s) c]
            for (int i = 0; i <= Math.min(end, k + 2); i++) {
                double ar = hr[i][k];
                double ai = hi[i][k];
                double br = hr[i][k + 1];
                double bi = hi[i][k + 1];
                hr[i][k] = c * ar + sr * br + si * bi;
                hi[i][k] = c * ai + sr * bi - si * br;
                hr[i][k + 1] = -(sr * ar - si * ai) + c * br;
                hi[i][k + 1] = -(sr * ai + si * ar) + c * bi;
            }
        }
        for (int i = lo; i <= end; i++) {
            hr[i][i] += mr;
            hi[i][i] += mi;
        }
    }

    private static double[] csqrt(double re, double im) {
        double r = Math.hypot(re, im);
        double a = Math.sqrt((r + re) / 2);
        double b = Math.sqrt(Math.max(0, (r - re) / 2));
        return new double[] {a, im < 0 ? -b : b};
    }

    // reduces to upper Hessenberg form by Householder similarity transforms
    private static double[][] hessenberg(double[][] a) {
        int n = a.length;
        double[][] h = copy(a);
        for (int k = 0; k < n - 2; k++) {
            double norm = 0;
            for (int i = k + 1; i < n; i++) {
                norm += h[i][k] * h[i][k];
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                continue;
            }
            double alpha = h[k + 1][k] > 0 ? -norm : norm;
            double[] v = new double[n];
            for (int i = k + 1; i < n; i++) {
                v[i] = h[i][k];
            }
            v[k + 1] -= alpha;
            double vv = 0;
            for (int i = k + 1; i < n; i++) {
                vv += v[i] * v[i];
            }
            if (vv == 0) {
                continue;
            }
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = k + 1; i < n; i++) {
                    s += v[i] * h[i][j];
                }
                s = 2 * s / vv;
                for (int i = k + 1; i < n; i++) {
                    h[i][j] -= s * v[i];
                }
            }
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = k + 1; j < n; j++) {
                    s += h[i][j] * v[j];
                }
                s = 2 * s / vv;
                for (int j = k + 1; j < n; j++) {
                    h[i][j] -= s * v[j];
                }
            }
            for (int i = k + 2; i < n; i++) {
                h[i][k] = 0;
            }
        }
        return h;
    }

    // ------------------------------------------------------------------ exponential

    /**
     * The matrix exponential {@code e^A}, by scaling and squaring with a degree-6 Padé
     * approximant, as {@code scipy.linalg.expm} for well-scaled matrices.
     *
     * @param a a square matrix
     * @return {@code e^A}
     */
    public static double[][] exp(double[][] a) {
        int n = square(a);
        double norm = 0;
        for (double[] row : a) {
            double s = 0;
            for (double v : row) {
                s += Math.abs(v);
            }
            norm = Math.max(norm, s);
        }
        int squarings = Math.max(0, (int) Math.ceil(Math.log(Math.max(norm, 1e-300) / 0.5) / Math.log(2)));
        double scale = Math.pow(2, -squarings);
        double[][] x = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x[i][j] = a[i][j] * scale;
            }
        }
        double[] c = {1, 0.5, 5.0 / 44, 1.0 / 66, 1.0 / 792, 1.0 / 15840, 1.0 / 665280};
        double[][] power = identity(n);
        double[][] num = identity(n);
        double[][] den = identity(n);
        for (int k = 1; k < c.length; k++) {
            power = multiply(power, x);
            double sign = k % 2 == 0 ? 1 : -1;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    num[i][j] += c[k] * power[i][j];
                    den[i][j] += sign * c[k] * power[i][j];
                }
            }
        }
        Lu f = lu(den);
        double[][] r = new double[n][n];
        double[] col = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                col[i] = num[i][j];
            }
            double[] s = f.solve(col);
            for (int i = 0; i < n; i++) {
                r[i][j] = s[i];
            }
        }
        for (int s = 0; s < squarings; s++) {
            r = multiply(r, r);
        }
        return r;
    }

    // ------------------------------------------------------------------ Lanczos

    /**
     * A few eigenpairs of a symmetric operator.
     *
     * @param values the eigenvalues, largest first when the largest were asked for, else smallest first
     * @param vectors one unit eigenvector per value, each of length {@code n}
     * @param residuals {@code |A v - lambda v|} of each pair
     */
    public record Eigenpairs(double[] values, double[][] vectors, double[] residuals) {
    }

    /**
     * The {@code k} largest or smallest eigenpairs of a symmetric operator of size {@code n}, by
     * the Lanczos method with full reorthogonalization: only products {@code A v} are needed,
     * so {@code A} can be sparse or implicit (a graph Laplacian, a kernel matrix applied in
     * blocks). Up to {@code steps} products are taken; more give more accurate pairs.
     *
     * @param a the operator
     * @param n its size
     * @param k how many pairs
     * @param largest the largest (true) or smallest (false)
     * @param steps the Krylov dimension, at least {@code k}; {@code min(n, max(3k, k + 30))} is a good start
     * @param seed seeds the starting vector
     * @return the pairs
     */
    public static Eigenpairs lanczos(Operator a, int n, int k, boolean largest, int steps, long seed) {
        if (k < 1 || k > n) {
            throw new IllegalArgumentException("k must be in [1, " + n + "], not " + k);
        }
        int m = Math.min(n, Math.max(k, steps));
        double[][] v = new double[m + 1][];
        double[] alpha = new double[m];
        double[] beta = new double[m];
        SplittableRandom random = new SplittableRandom(seed);
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            q[i] = random.nextDouble(-1, 1);
        }
        scale(q, 1 / norm(q));
        v[0] = q;
        double[] w = new double[n];
        int used = m;
        for (int j = 0; j < m; j++) {
            a.apply(v[j], w);
            alpha[j] = dot(w, v[j]);
            // full reorthogonalization, twice for stability
            for (int pass = 0; pass < 2; pass++) {
                for (int i = 0; i <= j; i++) {
                    double d = dot(w, v[i]);
                    for (int t = 0; t < n; t++) {
                        w[t] -= d * v[i][t];
                    }
                }
            }
            double b = norm(w);
            beta[j] = b;
            if (j == m - 1) {
                break;
            }
            if (b < 1e-12) {   // an invariant subspace: restart with a fresh orthogonal vector
                double[] r = new double[n];
                for (int i = 0; i < n; i++) {
                    r[i] = random.nextDouble(-1, 1);
                }
                for (int i = 0; i <= j; i++) {
                    double d = dot(r, v[i]);
                    for (int t = 0; t < n; t++) {
                        r[t] -= d * v[i][t];
                    }
                }
                double rn = norm(r);
                if (rn < 1e-12) {
                    used = j + 1;
                    break;
                }
                scale(r, 1 / rn);
                v[j + 1] = r;
                beta[j] = 0;
            } else {
                v[j + 1] = w.clone();
                scale(v[j + 1], 1 / b);
            }
        }
        double[][] t = new double[used][used];
        for (int i = 0; i < used; i++) {
            t[i][i] = alpha[i];
            if (i + 1 < used) {
                t[i][i + 1] = beta[i];
                t[i + 1][i] = beta[i];
            }
        }
        SymmetricEigen e = symmetricEigen(t);
        int take = Math.min(k, used);
        double[] values = new double[take];
        double[][] vectors = new double[take][n];
        double[] residuals = new double[take];
        double[] av = new double[n];
        for (int r = 0; r < take; r++) {
            int col = largest ? used - 1 - r : r;
            values[r] = e.values()[col];
            for (int i = 0; i < used; i++) {
                double y = e.vectors()[i][col];
                for (int x = 0; x < n; x++) {
                    vectors[r][x] += y * v[i][x];
                }
            }
            scale(vectors[r], 1 / norm(vectors[r]));
            a.apply(vectors[r], av);
            double res = 0;
            for (int x = 0; x < n; x++) {
                double d = av[x] - values[r] * vectors[r][x];
                res += d * d;
            }
            residuals[r] = Math.sqrt(res);
        }
        return new Eigenpairs(values, vectors, residuals);
    }

    // ------------------------------------------------------------------ helpers

    /**
     * @param a an {@code m x k} matrix
     * @param b a {@code k x n} matrix
     * @return {@code A B}
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int k = b.length;
        int n = b[0].length;
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                double v = a[i][p];
                if (v != 0) {
                    for (int j = 0; j < n; j++) {
                        out[i][j] += v * b[p][j];
                    }
                }
            }
        }
        return out;
    }

    /**
     * @param n the size
     * @return the identity
     */
    public static double[][] identity(int n) {
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            out[i][i] = 1;
        }
        return out;
    }

    private static double[][] copy(double[][] a) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i].clone();
        }
        return out;
    }

    // ------------------------------------------------------------------ conjugate gradient

    /**
     * Where an iterative solve ended.
     *
     * @param x the solution
     * @param iterations the iterations taken
     * @param residual the final {@code |b - A x| / |b|}
     * @param converged whether the tolerance was met
     */
    public record Solution(double[] x, int iterations, double residual, boolean converged) {
    }

    /**
     * Solves {@code A x = b} for a symmetric positive-definite operator by (preconditioned)
     * conjugate gradients, as {@code scipy.sparse.linalg.cg}: only products with {@code A} are
     * needed, so {@code A} can be sparse, structured or matrix-free.
     *
     * @param a the operator
     * @param b the right-hand side
     * @param x0 the start, or null for zeros
     * @param preconditioner an approximation of {@code A^-1} (symmetric positive definite), or null
     * @param tolerance stop when {@code |b - A x| <= tolerance |b|}
     * @param maxIterations the most iterations
     * @return the solution
     */
    public static Solution conjugateGradient(Operator a, double[] b, double[] x0, Operator preconditioner,
            double tolerance, int maxIterations) {
        int n = b.length;
        double[] x = x0 == null ? new double[n] : x0.clone();
        double[] r = new double[n];
        a.apply(x, r);
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - r[i];
        }
        double bn = norm(b);
        if (bn == 0) {
            return new Solution(new double[n], 0, 0, true);
        }
        double[] z = new double[n];
        precondition(preconditioner, r, z);
        double[] p = z.clone();
        double[] ap = new double[n];
        double rz = dot(r, z);
        double res = norm(r) / bn;
        int it = 0;
        while (res > tolerance && it < maxIterations) {
            it++;
            a.apply(p, ap);
            double pap = dot(p, ap);
            if (!(pap > 0)) {
                break;   // not positive definite along p
            }
            double alpha = rz / pap;
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            res = norm(r) / bn;
            precondition(preconditioner, r, z);
            double rzNew = dot(r, z);
            double beta = rzNew / rz;
            rz = rzNew;
            for (int i = 0; i < n; i++) {
                p[i] = z[i] + beta * p[i];
            }
        }
        return new Solution(x, it, res, res <= tolerance);
    }

    private static void precondition(Operator m, double[] r, double[] z) {
        if (m == null) {
            System.arraycopy(r, 0, z, 0, r.length);
        } else {
            m.apply(r, z);
        }
    }

    private static int square(double[][] a) {
        int n = a.length;
        for (double[] row : a) {
            if (row.length != n) {
                throw new IllegalArgumentException("the matrix must be square, got " + n + " rows of " + row.length);
            }
        }
        return n;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    private static void scale(double[] a, double f) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= f;
        }
    }
}
