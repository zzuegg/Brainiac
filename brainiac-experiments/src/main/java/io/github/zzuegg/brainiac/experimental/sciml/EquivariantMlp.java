package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Equivariant MLPs for arbitrary matrix groups (Finzi, Welling and Wilson, ICML 2021). The
 * linear maps {@code W} that commute with a group, {@code rho_out(g) W = W rho_in(g)}, form a
 * linear subspace, found numerically as a null space: for each Lie algebra generator
 * {@code A}, {@code drho_out(A) W - W drho_in(A) = 0}, and for each discrete generator
 * {@code h}, {@code rho_out(h) W rho_in(h)^-1 - W = 0}. Stacking these constraints on
 * {@code vec(W)} and taking the eigenvectors of {@code C^T C} with zero eigenvalue (the core
 * {@link Linalg#eigh}) gives a basis; a layer is a learned combination of it, equivariant by
 * construction for any weights. Representations combine by direct sums and tensor products
 * ({@link Representation#plus}, {@link Representation#times}).
 *
 * <p>The model here is the EMLP recipe for invariant targets from vectors: the input
 * representation {@code k V} is lifted bilinearly to {@code (kV) x (kV)}, an
 * {@link #basis equivariant linear map} takes it to {@code m} invariant scalars (with
 * multiplicity handled by a learned mixing of the basis, as EMLP does), and an ordinary MLP
 * acts on the invariants. It is a {@link Layer} from {@code [b, k * dim]} to
 * {@code [b, outputs]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2104.09459">arXiv:2104.09459</a>.
 */
public final class EquivariantMlp implements Layer {

    /**
     * A group representation given by its Lie algebra and discrete generators.
     *
     * @param size the dimension
     * @param algebra the Lie algebra generators {@code drho(A_i)}
     * @param discrete the discrete generators {@code rho(h_j)}
     */
    public record Representation(int size, List<double[][]> algebra, List<double[][]> discrete) {

        /**
         * The trivial representation (scalars) of the same group.
         *
         * @return a one-dimensional representation with zero generators
         */
        public Representation scalar() {
            List<double[][]> a = new ArrayList<>();
            for (int i = 0; i < algebra.size(); i++) {
                a.add(new double[][] {{0}});
            }
            List<double[][]> d = new ArrayList<>();
            for (int i = 0; i < discrete.size(); i++) {
                d.add(new double[][] {{1}});
            }
            return new Representation(1, a, d);
        }

        /**
         * The direct sum.
         *
         * @param other a representation of the same group
         * @return {@code this + other}, block diagonal
         */
        public Representation plus(Representation other) {
            List<double[][]> a = new ArrayList<>();
            for (int i = 0; i < algebra.size(); i++) {
                a.add(blockDiag(algebra.get(i), other.algebra().get(i)));
            }
            List<double[][]> d = new ArrayList<>();
            for (int i = 0; i < discrete.size(); i++) {
                d.add(blockDiag(discrete.get(i), other.discrete().get(i)));
            }
            return new Representation(size + other.size(), a, d);
        }

        /**
         * The tensor product: {@code A x I + I x B} on the algebra, {@code g x h} on discrete
         * elements.
         *
         * @param other a representation of the same group
         * @return {@code this x other}
         */
        public Representation times(Representation other) {
            List<double[][]> a = new ArrayList<>();
            for (int i = 0; i < algebra.size(); i++) {
                double[][] k1 = kron(algebra.get(i), identity(other.size()));
                double[][] k2 = kron(identity(size), other.algebra().get(i));
                for (int r = 0; r < k1.length; r++) {
                    for (int c = 0; c < k1.length; c++) {
                        k1[r][c] += k2[r][c];
                    }
                }
                a.add(k1);
            }
            List<double[][]> d = new ArrayList<>();
            for (int i = 0; i < discrete.size(); i++) {
                d.add(kron(discrete.get(i), other.discrete().get(i)));
            }
            return new Representation(size * other.size(), a, d);
        }

        /**
         * {@code k} copies of this representation.
         *
         * @param k the multiplicity
         * @return the direct sum of {@code k} copies
         */
        public Representation copies(int k) {
            Representation r = this;
            for (int i = 1; i < k; i++) {
                r = r.plus(this);
            }
            return r;
        }
    }

    /**
     * The vector representation of SO(3) or O(3) on {@code R^3}.
     *
     * @param reflections whether to include the parity {@code -I} (O(3))
     * @return the representation
     */
    public static Representation rotations3d(boolean reflections) {
        double[][] lx = {{0, 0, 0}, {0, 0, -1}, {0, 1, 0}};
        double[][] ly = {{0, 0, 1}, {0, 0, 0}, {-1, 0, 0}};
        double[][] lz = {{0, -1, 0}, {1, 0, 0}, {0, 0, 0}};
        List<double[][]> d = reflections
                ? List.<double[][]>of(new double[][] {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}})
                : List.<double[][]>of();
        return new Representation(3, List.of(lx, ly, lz), d);
    }

    /**
     * The vector representation of SO(2) on {@code R^2}.
     *
     * @return the representation
     */
    public static Representation rotations2d() {
        return new Representation(2, List.<double[][]>of(new double[][] {{0, -1}, {1, 0}}), List.<double[][]>of());
    }

    static double[][] identity(int n) {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            m[i][i] = 1;
        }
        return m;
    }

    static double[][] kron(double[][] a, double[][] b) {
        int p = a.length;
        int q = b.length;
        double[][] k = new double[p * q][p * q];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                for (int r = 0; r < q; r++) {
                    for (int c = 0; c < q; c++) {
                        k[i * q + r][j * q + c] = a[i][j] * b[r][c];
                    }
                }
            }
        }
        return k;
    }

    static double[][] blockDiag(double[][] a, double[][] b) {
        int p = a.length;
        int q = b.length;
        double[][] m = new double[p + q][p + q];
        for (int i = 0; i < p; i++) {
            System.arraycopy(a[i], 0, m[i], 0, p);
        }
        for (int i = 0; i < q; i++) {
            System.arraycopy(b[i], 0, m[p + i], p, q);
        }
        return m;
    }

    private static double[][] invert(double[][] m) {
        int n = m.length;
        double[] f = new double[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, f, i * n, n);
        }
        return Sindy.matrix(Linalg.inverse(Tensor.of(Shape.of(n, n), f, DType.FLOAT64, Device.cpu())));
    }

    /**
     * A basis of the equivariant linear maps from one representation to another.
     *
     * @param in the input representation
     * @param out the output representation
     * @param tolerance eigenvalues of {@code C^T C} below it count as zero, for example 1e-8
     * @return basis matrices {@code [k][out.size][in.size]}
     */
    public static double[][][] basis(Representation in, Representation out, double tolerance) {
        int n = in.size();
        int m = out.size();
        int dim = n * m;
        double[][] gram = new double[dim][dim];
        // vec(W) index: r * n + c. Constraint rows accumulate directly into C^T C.
        List<double[][]> constraints = new ArrayList<>();
        for (int g = 0; g < in.algebra().size(); g++) {
            constraints.add(operator(out.algebra().get(g), identity(n), identity(m), in.algebra().get(g), -1));
        }
        for (int g = 0; g < in.discrete().size(); g++) {
            double[][] inv = invert(in.discrete().get(g));
            double[][] c = operator(out.discrete().get(g), inv, identity(m), identity(n), 0);
            for (int i = 0; i < dim; i++) {
                c[i][i] -= 1;
            }
            constraints.add(c);
        }
        for (double[][] c : constraints) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    double s = 0;
                    for (int k = 0; k < dim; k++) {
                        s += c[k][i] * c[k][j];
                    }
                    gram[i][j] += s;
                }
            }
        }
        double[] flat = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            System.arraycopy(gram[i], 0, flat, i * dim, dim);
        }
        Tensor[] eig = Linalg.eigh(Tensor.of(Shape.of(dim, dim), flat, DType.FLOAT64, Device.cpu()));
        double[] values = eig[0].toDoubleArray();
        double[] vectors = eig[1].toDoubleArray();
        double scale = Math.max(1, values[dim - 1]);
        List<double[][]> out2 = new ArrayList<>();
        for (int k = 0; k < dim; k++) {
            if (values[k] > tolerance * scale) {
                continue;
            }
            double[][] w = new double[m][n];
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    w[r][c] = vectors[(r * n + c) * dim + k];
                }
            }
            out2.add(w);
        }
        return out2.toArray(new double[0][][]);
    }

    /**
     * The matrix of {@code W -> A W B + sign * C W D} on {@code vec(W)} (row-major), or of
     * {@code W -> A W B} when {@code sign} is 0.
     */
    private static double[][] operator(double[][] a, double[][] b, double[][] c, double[][] d, double sign) {
        int m = a.length;
        int n = b.length;
        double[][] op = new double[m * n][m * n];
        for (int r = 0; r < m; r++) {
            for (int col = 0; col < n; col++) {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        // (A W B)[r][col] = sum_ij A[r][i] W[i][j] B[j][col]
                        double v = a[r][i] * b[j][col];
                        if (sign != 0) {
                            v += sign * c[r][i] * d[j][col];
                        }
                        op[r * n + col][i * n + j] = v;
                    }
                }
            }
        }
        return op;
    }

    private final int vectors;
    private final int dim;
    private final Tensor invariantBasis;
    private final Var mixing;
    private final SimpleMlp head;

    private EquivariantMlp(Representation vector, int vectors, int invariants, int width, int outputs, long seed) {
        this.vectors = vectors;
        this.dim = vector.size();
        // invariant functionals on V x V: one per independent bilinear invariant
        double[][][] pair = basis(vector.times(vector), vector.scalar(), 1e-9);
        int k = vectors * vectors * pair.length;
        int full = vectors * dim;
        float[] b = new float[k * full * full];
        int idx = 0;
        for (int a = 0; a < vectors; a++) {
            for (int c = 0; c < vectors; c++) {
                for (double[][] p : pair) {
                    for (int i = 0; i < dim; i++) {
                        for (int j = 0; j < dim; j++) {
                            b[idx * full * full + (a * dim + i) * full + (c * dim + j)] = (float) p[0][i * dim + j];
                        }
                    }
                    idx++;
                }
            }
        }
        this.invariantBasis = Tensor.of(Shape.of(k, full * full), b);
        java.util.SplittableRandom r = new java.util.SplittableRandom(seed);
        this.mixing = Var.param(Tensor.normal(Shape.of(k, invariants), 0, (float) (1 / Math.sqrt(k)), r));
        this.head = SimpleMlp.of(seed + 1, v -> v.mul(v.sigmoid()), invariants, width, width, outputs);
    }

    /**
     * Creates an invariant EMLP over {@code vectors} copies of a representation.
     *
     * @param vector the per-vector representation, such as {@link #rotations3d}
     * @param vectors the input vectors {@code k}
     * @param invariants the invariant channels {@code m}
     * @param width the head width
     * @param outputs the outputs
     * @param seed the seed
     * @return the model
     */
    public static EquivariantMlp invariant(Representation vector, int vectors, int invariants, int width,
            int outputs, long seed) {
        return new EquivariantMlp(vector, vectors, invariants, width, outputs, seed);
    }

    @Override
    public Var forward(Var x) {
        int b = x.shape().dim(0);
        int full = vectors * dim;
        // bilinear lift x (x) x, then the equivariant (here invariant) linear map
        Var outer = x.reshape(b, full, 1).matmul(x.reshape(b, 1, full)).reshape(b, full * full);
        Var w = mixing.transpose().matmul(Var.of(invariantBasis));   // [m, full^2]
        Var inv = outer.matmul(w.transpose());
        return head.forward(inv);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(mixing));
        p.addAll(head.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return head.sizes()[head.sizes().length - 1];
    }

    @Override
    public String toString() {
        return "EquivariantMlp(vectors=" + vectors + ", dim=" + dim + ")";
    }
}
