package io.github.zzuegg.brainiac.tensor;

/**
 * Matrix decompositions and solves on the last two dimensions of a tensor, batched over
 * the leading ones: symmetric eigendecomposition, Cholesky, triangular and general solves,
 * least squares, inverses and powers. For preconditioners (Shampoo, SOAP), low-rank
 * projections (GaLore) and closed-form readouts.
 *
 * <p>The decompositions run on the host in double precision and return tensors of the
 * input's type on its device: they suit matrices up to a few hundred rows, which these
 * uses have. {@link #matrixPower} with a non-negative power runs on the device.
 */
public final class Linalg {

    private Linalg() {
    }

    /**
     * The eigenvalues and eigenvectors of symmetric matrices (cyclic Jacobi), so that
     * {@code a = vectors * diag(values) * vectors^T}. Only the symmetric part of {@code a}
     * is used.
     *
     * @param a symmetric matrices {@code [..., n, n]}
     * @return {values {@code [..., n]} in ascending order, vectors {@code [..., n, n]} with
     *     one eigenvector per column}
     */
    public static Tensor[] eigh(Tensor a) {
        int n = square(a);
        int batch = batch(a);
        double[] in = a.toDoubleArray();
        double[] values = new double[batch * n];
        double[] vectors = new double[batch * n * n];
        double[][] m = new double[n][n];
        double[][] v = new double[n][n];
        for (int b = 0; b < batch; b++) {
            int base = b * n * n;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    m[i][j] = 0.5 * (in[base + i * n + j] + in[base + j * n + i]);
                    v[i][j] = i == j ? 1 : 0;
                }
            }
            jacobi(m, v);
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (x, y) -> Double.compare(m[x][x], m[y][y]));
            for (int k = 0; k < n; k++) {
                int c = order[k];
                values[b * n + k] = m[c][c];
                for (int i = 0; i < n; i++) {
                    vectors[base + i * n + k] = v[i][c];
                }
            }
        }
        int[] dims = a.shape().toArray();
        int[] valueDims = java.util.Arrays.copyOf(dims, dims.length - 1);
        return new Tensor[] {
            Tensor.of(Shape.of(valueDims), values, a.dtype(), a.device()),
            Tensor.of(a.shape(), vectors, a.dtype(), a.device())
        };
    }

    /** Diagonalizes {@code m} in place, accumulating the rotations into {@code v}. */
    private static void jacobi(double[][] m, double[][] v) {
        int n = m.length;
        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            double total = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double s = m[i][j] * m[i][j];
                    total += s;
                    if (i != j) {
                        off += s;
                    }
                }
            }
            if (off <= 1e-30 * total || off == 0) {
                return;
            }
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = m[p][q];
                    if (apq == 0) {
                        continue;
                    }
                    double theta = (m[q][q] - m[p][p]) / (2 * apq);
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
    }

    /**
     * A function of symmetric matrices through their eigenvalues:
     * {@code vectors * diag(max(values, floor)^power) * vectors^T}. Power {@code -1/4} gives
     * Shampoo's inverse fourth root, {@code 0.5} a square root.
     *
     * @param a symmetric matrices {@code [..., n, n]}
     * @param power the power
     * @param floor the smallest eigenvalue used, such as a small epsilon for negative powers
     * @return the matrices
     */
    public static Tensor symmetricPower(Tensor a, double power, double floor) {
        Tensor[] eig = eigh(a);
        double[] values = eig[0].toDoubleArray();
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.pow(Math.max(values[i], floor), power);
        }
        int n = square(a);
        int[] dims = a.shape().toArray();
        dims[dims.length - 2] = 1;
        Tensor scaled = Tensor.of(Shape.of(dims), values, a.dtype(), a.device());
        Tensor v = eig[1];
        return v.mul(scaled).matmul(v.transpose());
    }

    /**
     * The Cholesky factor of symmetric positive definite matrices: lower triangular
     * {@code l} with {@code a = l * l^T}.
     *
     * @param a matrices {@code [..., n, n]}
     * @return the lower factors, zeros above the diagonal
     * @throws IllegalArgumentException if a matrix is not positive definite
     */
    public static Tensor cholesky(Tensor a) {
        int n = square(a);
        int batch = batch(a);
        double[] x = a.toDoubleArray();
        double[] l = new double[x.length];
        for (int b = 0; b < batch; b++) {
            int o = b * n * n;
            choleskyInto(x, l, o, n);
        }
        return Tensor.of(a.shape(), l, a.dtype(), a.device());
    }

    private static void choleskyInto(double[] x, double[] l, int o, int n) {
        for (int j = 0; j < n; j++) {
            double d = x[o + j * n + j];
            for (int k = 0; k < j; k++) {
                d -= l[o + j * n + k] * l[o + j * n + k];
            }
            if (!(d > 0)) {
                throw new IllegalArgumentException("matrix is not positive definite (pivot " + j + " is " + d + ")");
            }
            double djj = Math.sqrt(d);
            l[o + j * n + j] = djj;
            for (int i = j + 1; i < n; i++) {
                double s = x[o + i * n + j];
                for (int k = 0; k < j; k++) {
                    s -= l[o + i * n + k] * l[o + j * n + k];
                }
                l[o + i * n + j] = s / djj;
            }
        }
    }

    /**
     * Solves {@code a x = b} for triangular {@code a}.
     *
     * @param a triangular matrices {@code [..., n, n]}; the other triangle is ignored
     * @param b right-hand sides {@code [..., n, k]}, with the same leading dimensions
     * @param lower whether {@code a} is lower triangular
     * @return {@code x}, shaped like {@code b}
     */
    public static Tensor solveTriangular(Tensor a, Tensor b, boolean lower) {
        int n = square(a);
        int k = rhs(a, b, n);
        double[] m = a.toDoubleArray();
        double[] x = b.toDoubleArray();
        for (int t = 0; t < batch(a); t++) {
            triangular(m, t * n * n, x, t * n * k, n, k, lower, false);
        }
        return Tensor.of(b.shape(), x, b.dtype(), b.device());
    }

    /** Solves in place; {@code transposed} solves with {@code a^T}. */
    private static void triangular(double[] m, int mo, double[] x, int xo, int n, int k, boolean lower,
            boolean transposed) {
        boolean forward = lower != transposed;
        for (int s = 0; s < n; s++) {
            int i = forward ? s : n - 1 - s;
            double d = m[mo + i * n + i];
            for (int c = 0; c < k; c++) {
                double v = x[xo + i * k + c];
                if (forward) {
                    for (int j = 0; j < i; j++) {
                        v -= (transposed ? m[mo + j * n + i] : m[mo + i * n + j]) * x[xo + j * k + c];
                    }
                } else {
                    for (int j = i + 1; j < n; j++) {
                        v -= (transposed ? m[mo + j * n + i] : m[mo + i * n + j]) * x[xo + j * k + c];
                    }
                }
                x[xo + i * k + c] = v / d;
            }
        }
    }

    /**
     * Solves {@code a x = b} for square {@code a} (LU with partial pivoting).
     *
     * @param a matrices {@code [..., n, n]}
     * @param b right-hand sides {@code [..., n, k]}
     * @return {@code x}, shaped like {@code b}
     * @throws IllegalArgumentException if a matrix is singular
     */
    public static Tensor solve(Tensor a, Tensor b) {
        int n = square(a);
        int k = rhs(a, b, n);
        double[] m = a.toDoubleArray();
        double[] x = b.toDoubleArray();
        for (int t = 0; t < batch(a); t++) {
            int mo = t * n * n;
            int xo = t * n * k;
            for (int col = 0; col < n; col++) {
                int pivot = col;
                for (int r = col + 1; r < n; r++) {
                    if (Math.abs(m[mo + r * n + col]) > Math.abs(m[mo + pivot * n + col])) {
                        pivot = r;
                    }
                }
                if (m[mo + pivot * n + col] == 0) {
                    throw new IllegalArgumentException("matrix is singular");
                }
                if (pivot != col) {
                    swapRows(m, mo, n, pivot, col);
                    swapRows(x, xo, k, pivot, col);
                }
                double d = m[mo + col * n + col];
                for (int r = col + 1; r < n; r++) {
                    double f = m[mo + r * n + col] / d;
                    if (f == 0) {
                        continue;
                    }
                    for (int c = col; c < n; c++) {
                        m[mo + r * n + c] -= f * m[mo + col * n + c];
                    }
                    for (int c = 0; c < k; c++) {
                        x[xo + r * k + c] -= f * x[xo + col * k + c];
                    }
                }
            }
            triangular(m, mo, x, xo, n, k, false, false);
        }
        return Tensor.of(b.shape(), x, b.dtype(), b.device());
    }

    private static void swapRows(double[] x, int o, int width, int r1, int r2) {
        for (int c = 0; c < width; c++) {
            double t = x[o + r1 * width + c];
            x[o + r1 * width + c] = x[o + r2 * width + c];
            x[o + r2 * width + c] = t;
        }
    }

    /**
     * The inverses of square matrices.
     *
     * @param a matrices {@code [..., n, n]}
     * @return the inverses
     * @throws IllegalArgumentException if a matrix is singular
     */
    public static Tensor inverse(Tensor a) {
        int n = square(a);
        return solve(a, identity(n, a.dtype(), a.device()).expand(a.shape()));
    }

    /**
     * Least squares with an optional ridge: {@code x} minimizing
     * {@code |a x - b|^2 + ridge |x|^2}, from the normal equations by Cholesky.
     *
     * @param a matrices {@code [..., m, n]}, usually with {@code m >= n}
     * @param b targets {@code [..., m, k]}
     * @param ridge the regularization; 0 for plain least squares (then {@code a} needs full
     *     column rank)
     * @return {@code x} {@code [..., n, k]}
     */
    public static Tensor lstsq(Tensor a, Tensor b, double ridge) {
        if (ridge < 0) {
            throw new IllegalArgumentException("ridge must not be negative: " + ridge);
        }
        Tensor at = a.transpose();
        Tensor gram = at.matmul(a);
        int n = gram.shape().dim(-1);
        if (ridge > 0) {
            gram = gram.add(identity(n, a.dtype(), a.device()).mul((float) ridge));
        }
        Tensor rhs = at.matmul(b);
        int k = rhs.shape().dim(-1);
        double[] l = cholesky(gram).toDoubleArray();
        double[] x = rhs.toDoubleArray();
        for (int t = 0; t < batch(gram); t++) {
            triangular(l, t * n * n, x, t * n * k, n, k, true, false);
            triangular(l, t * n * n, x, t * n * k, n, k, true, true);
        }
        return Tensor.of(rhs.shape(), x, b.dtype(), b.device());
    }

    /**
     * An integer power of square matrices, by repeated squaring; a negative power inverts
     * first.
     *
     * @param a matrices {@code [..., n, n]}
     * @param power the power; 0 gives identities
     * @return {@code a^power}
     */
    public static Tensor matrixPower(Tensor a, int power) {
        int n = square(a);
        if (power < 0) {
            return matrixPower(inverse(a), -power);
        }
        Tensor result = null;
        Tensor base = a;
        for (int p = power; p > 0; p >>= 1) {
            if ((p & 1) != 0) {
                result = result == null ? base : result.matmul(base);
            }
            if (p > 1) {
                base = base.matmul(base);
            }
        }
        return result != null ? result : identity(n, a.dtype(), a.device()).expand(a.shape()).add(0f);
    }

    /**
     * An identity matrix.
     *
     * @param n its size
     * @param dtype its type
     * @param device its device
     * @return {@code [n, n]} with ones on the diagonal
     */
    public static Tensor identity(int n, DType dtype, Device device) {
        double[] v = new double[n * n];
        for (int i = 0; i < n; i++) {
            v[i * n + i] = 1;
        }
        return Tensor.of(Shape.of(n, n), v, dtype, device);
    }

    private static int square(Tensor a) {
        Shape s = a.shape();
        if (s.rank() < 2 || s.dim(-1) != s.dim(-2)) {
            throw new IllegalArgumentException("needs square matrices [..., n, n], got " + s);
        }
        return s.dim(-1);
    }

    private static int batch(Tensor a) {
        int n = a.shape().dim(-1) * a.shape().dim(-2);
        return Math.toIntExact(a.size() / n);
    }

    private static int rhs(Tensor a, Tensor b, int n) {
        Shape s = b.shape();
        if (s.rank() != a.shape().rank() || s.dim(-2) != n) {
            throw new IllegalArgumentException("right-hand sides " + s + " do not fit matrices " + a.shape());
        }
        for (int i = 0; i < s.rank() - 2; i++) {
            if (s.dim(i) != a.shape().dim(i)) {
                throw new IllegalArgumentException("right-hand sides " + s + " do not fit matrices " + a.shape());
            }
        }
        return s.dim(-1);
    }
}
