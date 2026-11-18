package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The parallel-beam Radon transform of a square image: for each of {@code angles} directions
 * evenly spread over {@code [0, pi)} and each of {@code side} unit-spaced detector bins, the line
 * integral of the image (bilinearly interpolated, sampled every half pixel along the ray). The
 * operator is stored as a sparse matrix, so the adjoint ({@link #adjoint}, unfiltered
 * back-projection) is its exact transpose; both are also available as differentiable
 * {@link Var} operations ({@link #apply(Var)}, {@link #adjoint(Var)}) for unrolled networks.
 *
 * <p>{@link #fbp} is filtered back-projection (Kak &amp; Slaney, 1988): each projection is
 * convolved with the discrete ramp filter ({@link Filter#RAM_LAK}, or the smoother
 * {@link Filter#SHEPP_LOGAN}) and smeared back along its rays (pixel-driven, linear
 * interpolation), scaled by {@code pi / angles}. Exact at full sampling without noise; it
 * streaks with few angles and amplifies noise.
 */
public final class RadonTransform implements LinearOperator {

    /** Ramp filters for filtered back-projection. */
    public enum Filter {
        /** The plain ramp {@code |w|}. */
        RAM_LAK,
        /** The ramp times a sinc window. */
        SHEPP_LOGAN
    }

    private final int side;
    private final int angles;
    private final double[] cos;
    private final double[] sin;
    private final int[] rowStart;
    private final int[] cols;
    private final double[] vals;

    /**
     * Builds the operator.
     *
     * @param side the image side (also the number of detector bins)
     * @param angles the number of projection angles
     */
    public RadonTransform(int side, int angles) {
        this.side = side;
        this.angles = angles;
        this.cos = new double[angles];
        this.sin = new double[angles];
        int rows = angles * side;
        this.rowStart = new int[rows + 1];
        double half = (side - 1) / 2.0;
        double reach = side / Math.sqrt(2) + 1;
        int samples = (int) Math.ceil(2 * reach / 0.5);
        int[] colBuf = new int[rows * (samples + 4)];
        double[] valBuf = new double[colBuf.length];
        double[] acc = new double[side * side];
        int[] touched = new int[side * side];
        int[] mark = new int[side * side];
        java.util.Arrays.fill(mark, -1);
        int nnz = 0;
        for (int a = 0; a < angles; a++) {
            double th = Math.PI * a / angles;
            cos[a] = Math.cos(th);
            sin[a] = Math.sin(th);
            for (int j = 0; j < side; j++) {
                int row = a * side + j;
                int count = 0;
                double s = j - half;
                for (int k = 0; k <= samples; k++) {
                    double t = -reach + 0.5 * k;
                    double u = s * cos[a] - t * sin[a] + half;
                    double v = s * sin[a] + t * cos[a] + half;
                    int u0 = (int) Math.floor(u);
                    int v0 = (int) Math.floor(v);
                    double fu = u - u0;
                    double fv = v - v0;
                    for (int dv = 0; dv < 2; dv++) {
                        for (int du = 0; du < 2; du++) {
                            int cu = u0 + du;
                            int cv = v0 + dv;
                            if (cu < 0 || cv < 0 || cu >= side || cv >= side) {
                                continue;
                            }
                            double w = 0.5 * (du == 0 ? 1 - fu : fu) * (dv == 0 ? 1 - fv : fv);
                            if (w == 0) {
                                continue;
                            }
                            int p = cv * side + cu;
                            if (mark[p] != row) {
                                mark[p] = row;
                                acc[p] = 0;
                                touched[count++] = p;
                            }
                            acc[p] += w;
                        }
                    }
                }
                java.util.Arrays.sort(touched, 0, count);
                if (nnz + count > colBuf.length) {
                    colBuf = java.util.Arrays.copyOf(colBuf, 2 * (nnz + count));
                    valBuf = java.util.Arrays.copyOf(valBuf, colBuf.length);
                }
                for (int i = 0; i < count; i++) {
                    colBuf[nnz] = touched[i];
                    valBuf[nnz++] = acc[touched[i]];
                }
                rowStart[row + 1] = nnz;
            }
        }
        this.cols = java.util.Arrays.copyOf(colBuf, nnz);
        this.vals = java.util.Arrays.copyOf(valBuf, nnz);
    }

    /** @return the image side */
    public int side() {
        return side;
    }

    /** @return the number of angles */
    public int angles() {
        return angles;
    }

    @Override
    public int inputSize() {
        return side * side;
    }

    @Override
    public int outputSize() {
        return angles * side;
    }

    @Override
    public double[] apply(double[] x) {
        double[] y = new double[angles * side];
        for (int r = 0; r < y.length; r++) {
            double s = 0;
            for (int k = rowStart[r]; k < rowStart[r + 1]; k++) {
                s += vals[k] * x[cols[k]];
            }
            y[r] = s;
        }
        return y;
    }

    @Override
    public double[] adjoint(double[] y) {
        double[] x = new double[side * side];
        for (int r = 0; r < y.length; r++) {
            double v = y[r];
            if (v == 0) {
                continue;
            }
            for (int k = rowStart[r]; k < rowStart[r + 1]; k++) {
                x[cols[k]] += vals[k] * v;
            }
        }
        return x;
    }

    /** @return the row sums of the system matrix (ray lengths) */
    public double[] rowSums() {
        double[] s = new double[angles * side];
        for (int r = 0; r < s.length; r++) {
            for (int k = rowStart[r]; k < rowStart[r + 1]; k++) {
                s[r] += vals[k];
            }
        }
        return s;
    }

    /**
     * One row of the system matrix (for Kaczmarz sweeps).
     *
     * @param row the row
     * @return {@code {columns, values}} as parallel arrays packed in a double matrix
     */
    public double[][] row(int row) {
        int len = rowStart[row + 1] - rowStart[row];
        double[] c = new double[len];
        double[] v = new double[len];
        for (int k = 0; k < len; k++) {
            c[k] = cols[rowStart[row] + k];
            v[k] = vals[rowStart[row] + k];
        }
        return new double[][] {c, v};
    }

    /**
     * Ramp-filters every projection (spatial-domain discrete filter for unit spacing).
     *
     * @param sinogram {@code angles x side}
     * @param filter the filter
     * @return the filtered sinogram
     */
    public double[] filter(double[] sinogram, Filter filter) {
        int n = side;
        double[] h = new double[2 * n - 1];
        for (int k = -(n - 1); k <= n - 1; k++) {
            double v;
            if (filter == Filter.RAM_LAK) {
                v = k == 0 ? 0.25 : (k % 2 == 0 ? 0 : -1 / (Math.PI * Math.PI * k * k));
            } else {
                v = -2 / (Math.PI * Math.PI * (4.0 * k * k - 1));
            }
            h[k + n - 1] = v;
        }
        double[] out = new double[sinogram.length];
        for (int a = 0; a < angles; a++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    s += h[j - i + n - 1] * sinogram[a * n + i];
                }
                out[a * n + j] = s;
            }
        }
        return out;
    }

    /**
     * Pixel-driven back-projection with linear interpolation on the detector, scaled by
     * {@code pi / angles} (the inverse of {@link #apply} after ramp filtering).
     *
     * @param sinogram the (filtered) sinogram
     * @return the image
     */
    public double[] backProject(double[] sinogram) {
        double half = (side - 1) / 2.0;
        double[] x = new double[side * side];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                double u = c - half;
                double v = r - half;
                double s = 0;
                for (int a = 0; a < angles; a++) {
                    double t = u * cos[a] + v * sin[a] + half;
                    int j = (int) Math.floor(t);
                    double f = t - j;
                    if (j >= 0 && j < side) {
                        s += (1 - f) * sinogram[a * side + j];
                    }
                    if (j + 1 >= 0 && j + 1 < side) {
                        s += f * sinogram[a * side + j + 1];
                    }
                }
                x[r * side + c] = s * Math.PI / angles;
            }
        }
        return x;
    }

    /**
     * Filtered back-projection.
     *
     * @param sinogram the sinogram
     * @param filter the ramp filter
     * @return the reconstruction
     */
    public double[] fbp(double[] sinogram, Filter filter) {
        return backProject(filter(sinogram, filter));
    }

    /**
     * The transform as a differentiable operation on a batch.
     *
     * @param x images {@code [b, 1, side, side]}
     * @return sinograms {@code [b, 1, angles, side]}
     */
    public Var apply(Var x) {
        return Var.custom(in -> batch(in[0], true), (in, out, g) -> new Tensor[] {batch(g, false)}, x);
    }

    /**
     * The adjoint as a differentiable operation on a batch.
     *
     * @param y sinograms {@code [b, 1, angles, side]}
     * @return images {@code [b, 1, side, side]}
     */
    public Var adjoint(Var y) {
        return Var.custom(in -> batch(in[0], false), (in, out, g) -> new Tensor[] {batch(g, true)}, y);
    }

    /**
     * Filtered back-projection of a batch (no gradient).
     *
     * @param y sinograms {@code [b, 1, angles, side]}
     * @param filter the filter
     * @return images {@code [b, 1, side, side]}
     */
    public Tensor fbp(Tensor y, Filter filter) {
        int b = y.shape().dim(0);
        float[] v = y.toArray();
        int m = angles * side;
        float[] out = new float[b * side * side];
        for (int i = 0; i < b; i++) {
            double[] s = new double[m];
            for (int k = 0; k < m; k++) {
                s[k] = v[i * m + k];
            }
            double[] x = fbp(s, filter);
            for (int k = 0; k < x.length; k++) {
                out[i * x.length + k] = (float) x[k];
            }
        }
        return Tensor.of(Shape.of(b, 1, side, side), out);
    }

    private Tensor batch(Tensor t, boolean forward) {
        int b = t.shape().dim(0);
        int inSize = forward ? side * side : angles * side;
        int outSize = forward ? angles * side : side * side;
        float[] v = t.toArray();
        float[] out = new float[b * outSize];
        for (int i = 0; i < b; i++) {
            double[] in = new double[inSize];
            for (int k = 0; k < inSize; k++) {
                in[k] = v[i * inSize + k];
            }
            double[] o = forward ? apply(in) : adjoint(in);
            for (int k = 0; k < outSize; k++) {
                out[i * outSize + k] = (float) o[k];
            }
        }
        return forward ? Tensor.of(Shape.of(b, 1, angles, side), out) : Tensor.of(Shape.of(b, 1, side, side), out);
    }
}
