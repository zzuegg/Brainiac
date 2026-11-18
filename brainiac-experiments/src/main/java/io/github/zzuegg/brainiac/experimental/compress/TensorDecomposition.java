package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Tensor decompositions for compressing layers, beyond the matrix SVD of {@code LowRankDense}:
 * <ul>
 * <li>{@link #cp}: the CP (canonical polyadic) decomposition by alternating least squares, a sum
 *     of {@code R} rank-one terms; a {@code [f, c, k, k]} convolution kernel then runs as four cheap
 *     convolutions {@code 1x1 (c -> R), k x 1, 1 x k (depthwise), 1x1 (R -> f)} (Lebedev et al. 2015,
 *     <a href="https://arxiv.org/abs/1412.6553">arXiv:1412.6553</a>);</li>
 * <li>{@link #tucker2}: Tucker-2 along the two channel modes by HOSVD, giving
 *     {@code 1x1 (c -> r_in), k x k core (r_in -> r_out), 1x1 (r_out -> f)} (Kim et al. 2016,
 *     <a href="https://arxiv.org/abs/1511.06530">arXiv:1511.06530</a>);</li>
 * <li>{@link #tensorTrain}: the TT-SVD of a dense matrix reshaped into a high-order tensor, a
 *     chain of small cores, which {@link TensorTrainDense} runs without rebuilding the matrix
 *     (Novikov et al. 2015, <a href="https://arxiv.org/abs/1509.06569">arXiv:1509.06569</a>).</li>
 * </ul>
 * Everything runs in double precision on the host with {@link ThinSvd}.
 */
public final class TensorDecomposition {

    private TensorDecomposition() {
    }

    /**
     * A CP decomposition {@code X ~ sum_r a1_r (x) a2_r (x) ... }.
     *
     * @param factors per mode, {@code [dim][rank]}
     */
    public record Cp(List<double[][]> factors) {

        /** @return the rebuilt tensor, row-major over the modes */
        public double[] rebuild() {
            int modes = factors.size();
            int[] dims = new int[modes];
            int size = 1;
            for (int m = 0; m < modes; m++) {
                dims[m] = factors.get(m).length;
                size *= dims[m];
            }
            int rank = factors.getFirst()[0].length;
            double[] x = new double[size];
            int[] idx = new int[modes];
            for (int p = 0; p < size; p++) {
                unravel(p, dims, idx);
                double s = 0;
                for (int r = 0; r < rank; r++) {
                    double t = 1;
                    for (int m = 0; m < modes; m++) {
                        t *= factors.get(m)[idx[m]][r];
                    }
                    s += t;
                }
                x[p] = s;
            }
            return x;
        }

        /** @return the number of stored values */
        public int parameters() {
            int n = 0;
            for (double[][] f : factors) {
                n += f.length * f[0].length;
            }
            return n;
        }
    }

    /**
     * CP by alternating least squares.
     *
     * @param x the tensor, row-major
     * @param dims its dimensions
     * @param rank the number of rank-one terms
     * @param iterations ALS sweeps
     * @param seed the seed of the random start
     * @return the decomposition
     */
    public static Cp cp(double[] x, int[] dims, int rank, int iterations, long seed) {
        int modes = dims.length;
        SplittableRandom r = new SplittableRandom(seed);
        List<double[][]> f = new ArrayList<>();
        for (int d : dims) {
            double[][] a = new double[d][rank];
            for (double[] row : a) {
                for (int k = 0; k < rank; k++) {
                    row[k] = r.nextGaussian();
                }
            }
            f.add(a);
        }
        int[] idx = new int[modes];
        for (int it = 0; it < iterations; it++) {
            for (int n = 0; n < modes; n++) {
                // V = hadamard of A_m^T A_m over m != n; M = X_(n) * khatri-rao
                double[][] v = new double[rank][rank];
                for (double[] row : v) {
                    java.util.Arrays.fill(row, 1);
                }
                for (int m = 0; m < modes; m++) {
                    if (m == n) {
                        continue;
                    }
                    double[][] a = f.get(m);
                    for (int p = 0; p < rank; p++) {
                        for (int q = 0; q < rank; q++) {
                            double s = 0;
                            for (double[] row : a) {
                                s += row[p] * row[q];
                            }
                            v[p][q] *= s;
                        }
                    }
                }
                double[][] mtt = new double[dims[n]][rank];
                for (int p = 0; p < x.length; p++) {
                    if (x[p] == 0) {
                        continue;
                    }
                    unravel(p, dims, idx);
                    for (int k = 0; k < rank; k++) {
                        double t = x[p];
                        for (int m = 0; m < modes; m++) {
                            if (m != n) {
                                t *= f.get(m)[idx[m]][k];
                            }
                        }
                        mtt[idx[n]][k] += t;
                    }
                }
                double trace = 0;
                for (int k = 0; k < rank; k++) {
                    trace += v[k][k];
                }
                for (int k = 0; k < rank; k++) {
                    v[k][k] += 1e-9 * trace / rank + 1e-12;
                }
                double[][] a = f.get(n);
                for (int i = 0; i < dims[n]; i++) {
                    a[i] = CurvatureProducts.solveSpd(v, mtt[i]);
                }
            }
        }
        return new Cp(f);
    }

    /**
     * A Tucker-2 decomposition of a convolution kernel: {@code W[f, c, ...] = sum U_out[f, a]
     * core[a, b, ...] U_in[c, b]}.
     *
     * @param out {@code U_out [f, r_out]}
     * @param core {@code [r_out, r_in, k...]}
     * @param in {@code U_in [c, r_in]}
     */
    public record Tucker2(Tensor out, Tensor core, Tensor in) {

        /** @return the number of stored values */
        public long parameters() {
            return out.shape().size() + core.shape().size() + in.shape().size();
        }

        /** @return the rebuilt kernel {@code [f, c, k...]} */
        public Tensor rebuild() {
            int f = out.shape().dim(0);
            int ro = out.shape().dim(1);
            int ri = in.shape().dim(1);
            int c = in.shape().dim(0);
            int k = (int) (core.shape().size() / ((long) ro * ri));
            float[] u = out.toArray();
            float[] g = core.toArray();
            float[] v = in.toArray();
            float[] w = new float[f * c * k];
            for (int i = 0; i < f; i++) {
                for (int j = 0; j < c; j++) {
                    for (int s = 0; s < k; s++) {
                        double sum = 0;
                        for (int a = 0; a < ro; a++) {
                            for (int b = 0; b < ri; b++) {
                                sum += u[i * ro + a] * g[(a * ri + b) * k + s] * v[j * ri + b];
                            }
                        }
                        w[(i * c + j) * k + s] = (float) sum;
                    }
                }
            }
            int[] d = core.shape().toArray();
            d[0] = f;
            d[1] = c;
            return Tensor.of(Shape.of(d), w);
        }
    }

    /**
     * Tucker-2 by HOSVD (truncated SVDs of the two channel unfoldings).
     *
     * @param kernel {@code [f, c, k...]}
     * @param rankOut {@code r_out <= f}
     * @param rankIn {@code r_in <= c}
     * @return the decomposition
     */
    public static Tucker2 tucker2(Tensor kernel, int rankOut, int rankIn) {
        int[] d = kernel.shape().toArray();
        int f = d[0];
        int c = d[1];
        int k = (int) (kernel.shape().size() / ((long) f * c));
        double[] w = kernel.toDoubleArray();
        double[][] mode0 = new double[f][c * k];
        double[][] mode1 = new double[c][f * k];
        for (int i = 0; i < f; i++) {
            for (int j = 0; j < c; j++) {
                for (int s = 0; s < k; s++) {
                    double v = w[(i * c + j) * k + s];
                    mode0[i][j * k + s] = v;
                    mode1[j][i * k + s] = v;
                }
            }
        }
        double[][] uo = ThinSvd.of(mode0).u();
        double[][] ui = ThinSvd.of(mode1).u();
        float[] out = new float[f * rankOut];
        float[] in = new float[c * rankIn];
        for (int i = 0; i < f; i++) {
            for (int a = 0; a < rankOut; a++) {
                out[i * rankOut + a] = (float) uo[i][a];
            }
        }
        for (int j = 0; j < c; j++) {
            for (int b = 0; b < rankIn; b++) {
                in[j * rankIn + b] = (float) ui[j][b];
            }
        }
        float[] core = new float[rankOut * rankIn * k];
        for (int a = 0; a < rankOut; a++) {
            for (int b = 0; b < rankIn; b++) {
                for (int s = 0; s < k; s++) {
                    double sum = 0;
                    for (int i = 0; i < f; i++) {
                        for (int j = 0; j < c; j++) {
                            sum += uo[i][a] * w[(i * c + j) * k + s] * ui[j][b];
                        }
                    }
                    core[(a * rankIn + b) * k + s] = (float) sum;
                }
            }
        }
        int[] cd = d.clone();
        cd[0] = rankOut;
        cd[1] = rankIn;
        return new Tucker2(Tensor.of(Shape.of(f, rankOut), out), Tensor.of(Shape.of(cd), core),
                Tensor.of(Shape.of(c, rankIn), in));
    }

    /**
     * TT-SVD of a matrix {@code W [in, out]} with {@code in = prod inFactors} and
     * {@code out = prod outFactors}: core {@code k} has shape {@code [r_{k-1}, i_k, j_k, r_k]}.
     *
     * @param matrix the matrix
     * @param inFactors the factorisation of the input size
     * @param outFactors the factorisation of the output size, same length
     * @param maxRank the largest TT rank
     * @return the cores
     */
    public static List<Tensor> tensorTrain(Tensor matrix, int[] inFactors, int[] outFactors, int maxRank) {
        int d = inFactors.length;
        int in = matrix.shape().dim(0);
        int out = matrix.shape().dim(1);
        double[] w = matrix.toDoubleArray();
        // permute W[i1..id, j1..jd] to A[i1, j1, i2, j2, ...]
        int[] dims = new int[2 * d];
        for (int k = 0; k < d; k++) {
            dims[2 * k] = inFactors[k];
            dims[2 * k + 1] = outFactors[k];
        }
        double[] a = new double[in * out];
        int[] idx = new int[2 * d];
        for (int p = 0; p < a.length; p++) {
            unravel(p, dims, idx);
            int i = 0;
            int j = 0;
            for (int k = 0; k < d; k++) {
                i = i * inFactors[k] + idx[2 * k];
                j = j * outFactors[k] + idx[2 * k + 1];
            }
            a[p] = w[i * out + j];
        }
        List<Tensor> cores = new ArrayList<>();
        int r = 1;
        double[] rest = a;
        int restSize = in * out;
        for (int k = 0; k < d - 1; k++) {
            int rows = r * inFactors[k] * outFactors[k];
            int cols = restSize / (inFactors[k] * outFactors[k]);
            double[][] c = new double[rows][cols];
            for (int p = 0; p < rows; p++) {
                System.arraycopy(rest, p * cols, c[p], 0, cols);
            }
            ThinSvd svd = ThinSvd.of(c);
            int nr = Math.min(maxRank, svd.rank());
            float[] core = new float[rows * nr];
            for (int p = 0; p < rows; p++) {
                for (int q = 0; q < nr; q++) {
                    core[p * nr + q] = (float) svd.u()[p][q];
                }
            }
            cores.add(Tensor.of(Shape.of(r, inFactors[k], outFactors[k], nr), core));
            double[] next = new double[nr * cols];
            for (int q = 0; q < nr; q++) {
                for (int t = 0; t < cols; t++) {
                    next[q * cols + t] = svd.s()[q] * svd.v()[t][q];
                }
            }
            rest = next;
            restSize = cols;
            r = nr;
        }
        float[] last = new float[rest.length];
        for (int p = 0; p < last.length; p++) {
            last[p] = (float) rest[p];
        }
        cores.add(Tensor.of(Shape.of(r, inFactors[d - 1], outFactors[d - 1], 1), last));
        return cores;
    }

    private static void unravel(int p, int[] dims, int[] idx) {
        for (int m = dims.length - 1; m >= 0; m--) {
            idx[m] = p % dims[m];
            p /= dims[m];
        }
    }
}
