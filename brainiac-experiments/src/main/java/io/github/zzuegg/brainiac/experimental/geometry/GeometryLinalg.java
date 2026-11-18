package io.github.zzuegg.brainiac.experimental.geometry;

/**
 * Small dense double-precision linear algebra for geometry and estimation: products,
 * Gauss-Jordan solves and inverses, a damped Cholesky solve, a cyclic Jacobi symmetric
 * eigendecomposition, a one-sided Jacobi SVD (accurate for the tiny singular values that the
 * eight-point algorithm, DLT and EPnP read their null spaces from, unlike an SVD from the Gram
 * matrix), and the rotation tools every geometry block needs: Rodrigues' exponential and
 * logarithm on SO(3), the skew matrix, the angle between rotations, and Kabsch/Umeyama
 * alignment of corresponding point sets.
 *
 * <p>Core {@code Linalg} works on float tensors; these routines stay on {@code double[][]}
 * because the matrices are 3x3 to a few hundred square and conditioning matters.
 */
public final class GeometryLinalg {

    private GeometryLinalg() {
    }

    /**
     * A singular value decomposition {@code A = U diag(s) V^T}.
     *
     * @param u the left singular vectors {@code [m][k]}
     * @param s the singular values, descending
     * @param v the right singular vectors {@code [n][n]}, columns
     */
    public record Svd(double[][] u, double[] s, double[][] v) {

        /**
         * @param j a column
         * @return the right singular vector {@code j}
         */
        public double[] rightVector(int j) {
            double[] out = new double[v.length];
            for (int i = 0; i < v.length; i++) {
                out[i] = v[i][j];
            }
            return out;
        }

        /** @return the right singular vector of the smallest singular value */
        public double[] nullVector() {
            return rightVector(v[0].length - 1);
        }
    }

    /**
     * @param a a matrix
     * @param b a matrix
     * @return {@code a b}
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        int k = b.length;
        int m = b[0].length;
        double[][] c = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int p = 0; p < k; p++) {
                double v = a[i][p];
                if (v == 0) {
                    continue;
                }
                for (int j = 0; j < m; j++) {
                    c[i][j] += v * b[p][j];
                }
            }
        }
        return c;
    }

    /**
     * @param a a matrix
     * @param x a vector
     * @return {@code a x}
     */
    public static double[] multiply(double[][] a, double[] x) {
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

    /**
     * @param a a matrix
     * @return its transpose
     */
    public static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    /**
     * @param n the size
     * @return the identity
     */
    public static double[][] identity(int n) {
        double[][] e = new double[n][n];
        for (int i = 0; i < n; i++) {
            e[i][i] = 1;
        }
        return e;
    }

    /**
     * @param a a vector
     * @param b a vector
     * @return the dot product
     */
    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * @param a a vector
     * @return its Euclidean norm
     */
    public static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    /**
     * @param a a vector
     * @param b a vector
     * @return {@code a - b}
     */
    public static double[] subtract(double[] a, double[] b) {
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] - b[i];
        }
        return c;
    }

    /**
     * @param a a 3-vector
     * @param b a 3-vector
     * @return {@code a x b}
     */
    public static double[] cross(double[] a, double[] b) {
        return new double[] {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    /**
     * @param a a 3x3 matrix
     * @return its determinant
     */
    public static double det3(double[][] a) {
        return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    }

    /**
     * Solves {@code a x = b} by Gauss-Jordan elimination with partial pivoting.
     *
     * @param a a square matrix (not modified)
     * @param b right-hand sides {@code [n][k]} (not modified)
     * @return the solution, or {@code null} if {@code a} is singular
     */
    public static double[][] solve(double[][] a, double[][] b) {
        int n = a.length;
        int k = b[0].length;
        double[][] m = new double[n][n + k];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            System.arraycopy(b[i], 0, m[i], n, k);
        }
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(m[r][c]) > Math.abs(m[p][c])) {
                    p = r;
                }
            }
            if (Math.abs(m[p][c]) < 1e-300) {
                return null;
            }
            double[] t = m[p];
            m[p] = m[c];
            m[c] = t;
            double inv = 1 / m[c][c];
            for (int j = c; j < n + k; j++) {
                m[c][j] *= inv;
            }
            for (int r = 0; r < n; r++) {
                if (r != c && m[r][c] != 0) {
                    double f = m[r][c];
                    for (int j = c; j < n + k; j++) {
                        m[r][j] -= f * m[c][j];
                    }
                }
            }
        }
        double[][] x = new double[n][k];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], n, x[i], 0, k);
        }
        return x;
    }

    /**
     * Solves {@code a x = b} for one right-hand side.
     *
     * @param a a square matrix
     * @param b the right-hand side
     * @return the solution, or {@code null} if singular
     */
    public static double[] solve(double[][] a, double[] b) {
        double[][] bb = new double[b.length][1];
        for (int i = 0; i < b.length; i++) {
            bb[i][0] = b[i];
        }
        double[][] x = solve(a, bb);
        if (x == null) {
            return null;
        }
        double[] out = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            out[i] = x[i][0];
        }
        return out;
    }

    /**
     * Solves a symmetric positive (semi)definite system by Cholesky, adding jitter on the
     * diagonal until the factorisation succeeds.
     *
     * @param a a symmetric matrix (not modified)
     * @param b the right-hand side
     * @return the solution
     */
    public static double[] solveSpd(double[][] a, double[] b) {
        int n = a.length;
        double jitter = 0;
        double scale = 0;
        for (int i = 0; i < n; i++) {
            scale = Math.max(scale, Math.abs(a[i][i]));
        }
        scale = Math.max(scale, 1e-12);
        for (int attempt = 0; attempt < 12; attempt++) {
            double[][] l = new double[n][n];
            boolean ok = true;
            for (int i = 0; i < n && ok; i++) {
                for (int j = 0; j <= i; j++) {
                    double s = a[i][j] + (i == j ? jitter : 0);
                    for (int p = 0; p < j; p++) {
                        s -= l[i][p] * l[j][p];
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
                    double s = b[i];
                    for (int p = 0; p < i; p++) {
                        s -= l[i][p] * y[p];
                    }
                    y[i] = s / l[i][i];
                }
                double[] x = new double[n];
                for (int i = n - 1; i >= 0; i--) {
                    double s = y[i];
                    for (int p = i + 1; p < n; p++) {
                        s -= l[p][i] * x[p];
                    }
                    x[i] = s / l[i][i];
                }
                return x;
            }
            jitter = jitter == 0 ? 1e-12 * scale : jitter * 10;
        }
        double[] x = solve(a, b);
        return x == null ? new double[n] : x;
    }

    /**
     * @param a a square matrix
     * @return its inverse, or {@code null} if singular
     */
    public static double[][] inverse(double[][] a) {
        return solve(a, identity(a.length));
    }

    /**
     * The symmetric eigendecomposition by cyclic Jacobi rotations.
     *
     * @param a a symmetric matrix (not modified)
     * @return {@code {values (as a 1-row matrix), vectors (columns)}}, values descending
     */
    public static double[][][] eigh(double[][] a) {
        int n = a.length;
        double[][] m = new double[n][];
        for (int i = 0; i < n; i++) {
            m[i] = a[i].clone();
        }
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
        java.util.Arrays.sort(order, (x, y) -> Double.compare(m[y][y], m[x][x]));
        double[][] values = new double[1][n];
        double[][] vectors = new double[n][n];
        for (int j = 0; j < n; j++) {
            values[0][j] = m[order[j]][order[j]];
            for (int i = 0; i < n; i++) {
                vectors[i][j] = v[i][order[j]];
            }
        }
        return new double[][][] {values, vectors};
    }

    /**
     * The singular value decomposition by one-sided Jacobi rotations. For {@code m < n} the
     * matrix is padded with zero rows, so {@code V} is always the full {@code n x n} basis and
     * its last column spans the null space of a wide matrix.
     *
     * @param a the matrix {@code [m][n]} (not modified)
     * @return the decomposition, with {@code max(m, n)} rows padded in {@code U}
     */
    public static Svd svd(double[][] a) {
        int n = a[0].length;
        int m = Math.max(a.length, n);
        double[][] u = new double[m][n];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, u[i], 0, n);
        }
        double[][] v = identity(n);
        for (int sweep = 0; sweep < 80; sweep++) {
            double rotated = 0;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double alpha = 0;
                    double beta = 0;
                    double gamma = 0;
                    for (int i = 0; i < m; i++) {
                        alpha += u[i][p] * u[i][p];
                        beta += u[i][q] * u[i][q];
                        gamma += u[i][p] * u[i][q];
                    }
                    if (Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta) || gamma == 0) {
                        continue;
                    }
                    rotated = Math.max(rotated, Math.abs(gamma) / Math.sqrt(alpha * beta));
                    double zeta = (beta - alpha) / (2 * gamma);
                    double t = Math.signum(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
                    if (zeta == 0) {
                        t = 1;
                    }
                    double c = 1 / Math.sqrt(1 + t * t);
                    double s = c * t;
                    for (int i = 0; i < m; i++) {
                        double x = u[i][p];
                        double y = u[i][q];
                        u[i][p] = c * x - s * y;
                        u[i][q] = s * x + c * y;
                    }
                    for (int i = 0; i < n; i++) {
                        double x = v[i][p];
                        double y = v[i][q];
                        v[i][p] = c * x - s * y;
                        v[i][q] = s * x + c * y;
                    }
                }
            }
            if (rotated < 1e-15) {
                break;
            }
        }
        double[] s = new double[n];
        for (int j = 0; j < n; j++) {
            double t = 0;
            for (int i = 0; i < m; i++) {
                t += u[i][j] * u[i][j];
            }
            s[j] = Math.sqrt(t);
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(s[y], s[x]));
        double[][] uu = new double[m][n];
        double[][] vv = new double[n][n];
        double[] ss = new double[n];
        for (int j = 0; j < n; j++) {
            int o = order[j];
            ss[j] = s[o];
            for (int i = 0; i < m; i++) {
                uu[i][j] = s[o] > 1e-300 ? u[i][o] / s[o] : 0;
            }
            for (int i = 0; i < n; i++) {
                vv[i][j] = v[i][o];
            }
        }
        double tiny = 1e-12 * Math.max(ss[0], 1e-300);
        for (int j = 0; j < n; j++) {
            if (ss[j] > tiny) {
                continue;
            }
            for (int e = 0; e < m; e++) {
                double[] c = new double[m];
                c[e] = 1;
                for (int k = 0; k < n; k++) {
                    if (k == j || (k > j && ss[k] <= tiny)) {
                        continue;
                    }
                    double p = 0;
                    for (int i = 0; i < m; i++) {
                        p += c[i] * uu[i][k];
                    }
                    for (int i = 0; i < m; i++) {
                        c[i] -= p * uu[i][k];
                    }
                }
                double nn = 0;
                for (double x : c) {
                    nn += x * x;
                }
                if (nn > 1e-6) {
                    nn = Math.sqrt(nn);
                    for (int i = 0; i < m; i++) {
                        uu[i][j] = c[i] / nn;
                    }
                    break;
                }
            }
        }
        return new Svd(uu, ss, vv);
    }

    /**
     * The Moore-Penrose pseudo-inverse.
     *
     * @param a the matrix {@code [m][n]}
     * @param tolerance relative singular value cut-off, such as {@code 1e-10}
     * @return {@code [n][m]}
     */
    public static double[][] pseudoInverse(double[][] a, double tolerance) {
        int m = a.length;
        boolean tall = m >= a[0].length;
        double[][] b = tall ? a : transpose(a);
        Svd d = svd(b);
        int n = b[0].length;
        int rows = b.length;
        double[][] p = new double[n][rows];
        double cut = d.s()[0] * tolerance;
        for (int k = 0; k < n; k++) {
            if (d.s()[k] <= cut || d.s()[k] == 0) {
                continue;
            }
            double inv = 1 / d.s()[k];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < rows; j++) {
                    p[i][j] += d.v()[i][k] * inv * d.u()[j][k];
                }
            }
        }
        return tall ? p : transpose(p);
    }

    /**
     * The skew-symmetric cross-product matrix {@code [w]x}.
     *
     * @param w a 3-vector
     * @return {@code [w]x}
     */
    public static double[][] skew(double[] w) {
        return new double[][] {{0, -w[2], w[1]}, {w[2], 0, -w[0]}, {-w[1], w[0], 0}};
    }

    /**
     * Rodrigues' formula: the rotation by angle {@code |w|} about axis {@code w / |w|}.
     *
     * @param w a rotation vector
     * @return the rotation matrix
     */
    public static double[][] exp(double[] w) {
        double theta = norm(w);
        double[][] k = skew(w);
        double[][] k2 = multiply(k, k);
        double a;
        double b;
        if (theta < 1e-8) {
            a = 1 - theta * theta / 6;
            b = 0.5 - theta * theta / 24;
        } else {
            a = Math.sin(theta) / theta;
            b = (1 - Math.cos(theta)) / (theta * theta);
        }
        double[][] r = identity(3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i][j] += a * k[i][j] + b * k2[i][j];
            }
        }
        return r;
    }

    /**
     * The rotation vector of a rotation matrix (the inverse of {@link #exp}).
     *
     * @param r a rotation matrix
     * @return {@code w} with {@code |w|} in {@code [0, pi]}
     */
    public static double[] log(double[][] r) {
        double c = Math.max(-1, Math.min(1, (r[0][0] + r[1][1] + r[2][2] - 1) / 2));
        double theta = Math.acos(c);
        double[] v = {r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]};
        if (theta < 1e-8) {
            return new double[] {v[0] / 2, v[1] / 2, v[2] / 2};
        }
        if (Math.PI - theta < 1e-4) {
            double[] axis = new double[3];
            int k = 0;
            for (int i = 1; i < 3; i++) {
                if (r[i][i] > r[k][k]) {
                    k = i;
                }
            }
            axis[k] = Math.sqrt(Math.max(0, (r[k][k] + 1) / 2));
            for (int i = 0; i < 3; i++) {
                if (i != k) {
                    axis[i] = (r[i][k] + r[k][i]) / (4 * axis[k]);
                }
            }
            double nn = norm(axis);
            return new double[] {theta * axis[0] / nn, theta * axis[1] / nn, theta * axis[2] / nn};
        }
        double f = theta / (2 * Math.sin(theta));
        return new double[] {f * v[0], f * v[1], f * v[2]};
    }

    /**
     * The geodesic angle between two rotations.
     *
     * @param a a rotation
     * @param b a rotation
     * @return the angle of {@code a^T b} in degrees
     */
    public static double angleDegrees(double[][] a, double[][] b) {
        double[][] d = multiply(transpose(a), b);
        double c = Math.max(-1, Math.min(1, (d[0][0] + d[1][1] + d[2][2] - 1) / 2));
        return Math.toDegrees(Math.acos(c));
    }

    /**
     * The angle between two directions, ignoring sign if requested.
     *
     * @param a a vector
     * @param b a vector
     * @param unsigned whether {@code b} and {@code -b} count as the same direction
     * @return the angle in degrees
     */
    public static double directionDegrees(double[] a, double[] b, boolean unsigned) {
        double c = dot(a, b) / Math.max(1e-300, norm(a) * norm(b));
        if (unsigned) {
            c = Math.abs(c);
        }
        return Math.toDegrees(Math.acos(Math.max(-1, Math.min(1, c))));
    }

    /**
     * The nearest rotation to a 3x3 matrix in the Frobenius norm, with determinant +1.
     *
     * @param m a matrix
     * @return the rotation
     */
    public static double[][] nearestRotation(double[][] m) {
        Svd d = svd(m);
        double[][] r = multiply(d.u(), transpose(d.v()));
        if (det3(r) < 0) {
            double[][] u = new double[3][3];
            for (int i = 0; i < 3; i++) {
                u[i] = d.u()[i].clone();
                u[i][2] = -u[i][2];
            }
            r = multiply(u, transpose(d.v()));
        }
        return r;
    }

    /**
     * Kabsch alignment: the rotation {@code R} and translation {@code t} minimising the
     * weighted {@code sum w_i |R a_i + t - b_i|^2}.
     *
     * @param a source points {@code [n][3]}
     * @param b target points {@code [n][3]}
     * @param weights per-point weights, or {@code null} for equal weights
     * @return {@code {R (3 rows), t (1 row)}} packed as a 4x3 matrix: rows 0-2 are R, row 3 is t
     */
    public static double[][] kabsch(double[][] a, double[][] b, double[] weights) {
        int n = a.length;
        double[] ca = new double[3];
        double[] cb = new double[3];
        double total = 0;
        for (int i = 0; i < n; i++) {
            double w = weights == null ? 1 : weights[i];
            total += w;
            for (int k = 0; k < 3; k++) {
                ca[k] += w * a[i][k];
                cb[k] += w * b[i][k];
            }
        }
        for (int k = 0; k < 3; k++) {
            ca[k] /= Math.max(total, 1e-300);
            cb[k] /= Math.max(total, 1e-300);
        }
        double[][] h = new double[3][3];
        for (int i = 0; i < n; i++) {
            double w = weights == null ? 1 : weights[i];
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    h[p][q] += w * (b[i][p] - cb[p]) * (a[i][q] - ca[q]);
                }
            }
        }
        double[][] r = nearestRotation(h);
        double[] ra = multiply(r, ca);
        return new double[][] {r[0], r[1], r[2], {cb[0] - ra[0], cb[1] - ra[1], cb[2] - ra[2]}};
    }

    /**
     * Applies a rigid transform.
     *
     * @param r a rotation
     * @param t a translation
     * @param p a point
     * @return {@code R p + t}
     */
    public static double[] transform(double[][] r, double[] t, double[] p) {
        double[] q = multiply(r, p);
        return new double[] {q[0] + t[0], q[1] + t[1], q[2] + t[2]};
    }

    /**
     * Wraps an angle to {@code (-pi, pi]}.
     *
     * @param a an angle in radians
     * @return the wrapped angle
     */
    public static double wrap(double a) {
        double x = a % (2 * Math.PI);
        if (x > Math.PI) {
            x -= 2 * Math.PI;
        } else if (x <= -Math.PI) {
            x += 2 * Math.PI;
        }
        return x;
    }
}
