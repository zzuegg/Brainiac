package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * SVDQuant: W4A4 with the outliers absorbed by a 16-bit low-rank branch. At 4 bits both
 * weights and activations have outliers that a uniform grid cannot hold. SmoothQuant moves the
 * activation outliers into the weights ({@code x W = (x / s)(s W)}), which only makes the
 * weight outliers worse. SVDQuant then splits the smoothed weight {@code s W} into its top
 * {@code rank} singular directions {@code L1 L2}, kept in high precision, and a residual
 * {@code R = s W - L1 L2} whose outliers are gone and which quantizes well:
 *
 * <pre>
 * y = Q4(x / s) Q4(R) + (x / s) L1 L2
 * </pre>
 *
 * <p>The low-rank branch costs {@code rank (in + out)} high-precision values and two thin
 * products. Activations are rounded per token (symmetric, {@code bits} bits), the residual per
 * output column, as in {@link RotatedQuant}'s baselines, so the three are directly comparable.
 * Inference only.
 *
 * <p>Li et al., "SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion
 * Models" (ICLR 2025), <a href="https://arxiv.org/abs/2411.05007">arXiv:2411.05007</a>.
 */
public final class LowRankOutlierQuant implements Layer {

    private final int in;
    private final int out;
    private final int bits;
    private final double[] smooth;
    private final Tensor residual;
    private final Tensor left;
    private final Tensor right;
    private final Tensor bias;
    private final Activation activation;

    private LowRankOutlierQuant(Dense dense, int bits, int rank, Tensor calibration, float alpha) {
        Tensor w = dense.weight().value();
        this.in = w.shape().dim(0);
        this.out = w.shape().dim(1);
        this.bits = bits;
        this.activation = dense.activation();
        this.bias = dense.bias() == null ? null : dense.bias().value();
        this.smooth = SmoothQuant.scales(calibration, w, alpha);
        float[] m = w.toArray();
        double[][] a = new double[in][out];
        for (int r = 0; r < in; r++) {
            for (int c = 0; c < out; c++) {
                a[r][c] = m[r * out + c] * smooth[r];
            }
        }
        ThinSvd svd = ThinSvd.of(a);
        int k = Math.min(rank, svd.rank());
        float[] l = new float[in * Math.max(k, 1)];
        float[] rt = new float[Math.max(k, 1) * out];
        for (int j = 0; j < k; j++) {
            for (int r = 0; r < in; r++) {
                l[r * k + j] = (float) (svd.u()[r][j] * svd.s()[j]);
            }
            for (int c = 0; c < out; c++) {
                rt[j * out + c] = (float) svd.v()[c][j];
            }
        }
        float[] res = new float[in * out];
        for (int r = 0; r < in; r++) {
            for (int c = 0; c < out; c++) {
                double low = 0;
                for (int j = 0; j < k; j++) {
                    low += l[r * k + j] * rt[j * out + c];
                }
                res[r * out + c] = (float) (a[r][c] - low);
            }
        }
        this.left = k == 0 ? null : Tensor.of(Shape.of(in, k), l, w.device());
        this.right = k == 0 ? null : Tensor.of(Shape.of(k, out), rt, w.device());
        this.residual = Gptq.roundToNearest(Tensor.of(Shape.of(in, out), res, w.device()), bits);
    }

    /**
     * Builds the layer.
     *
     * @param dense the layer to quantize
     * @param bits bits for activations and the residual weight
     * @param rank the low-rank branch's rank (0 gives plain SmoothQuant W4A4)
     * @param calibration calibration inputs {@code [samples, in]}
     * @param alpha the smoothing migration strength, 0.5 in SmoothQuant
     * @return the quantized layer
     */
    public static LowRankOutlierQuant of(Dense dense, int bits, int rank, Tensor calibration, float alpha) {
        return new LowRankOutlierQuant(dense, bits, rank, calibration, alpha);
    }

    /** @return the high-precision values of the low-rank branch */
    public long lowRankValues() {
        return left == null ? 0 : left.size() + right.size();
    }

    @Override
    public Var forward(Var input) {
        Shape shape = input.shape();
        float[] a = input.value().toArray();
        int rows = a.length / in;
        int levels = (1 << (bits - 1)) - 1;
        float[] smoothed = new float[a.length];
        float[] quant = new float[a.length];
        for (int n = 0; n < rows; n++) {
            float amax = 0;
            for (int i = 0; i < in; i++) {
                smoothed[n * in + i] = (float) (a[n * in + i] / smooth[i]);
                amax = Math.max(amax, Math.abs(smoothed[n * in + i]));
            }
            float scale = amax == 0 ? 1 : amax / levels;
            for (int i = 0; i < in; i++) {
                quant[n * in + i] = Math.max(-levels, Math.min(levels, Math.round(smoothed[n * in + i] / scale)))
                        * scale;
            }
        }
        Tensor xq = Tensor.of(Shape.of(rows, in), quant, input.value().device());
        Tensor y = xq.matmul(residual);
        if (left != null) {
            Tensor xs = Tensor.of(Shape.of(rows, in), smoothed, input.value().device());
            y = y.add(xs.matmul(left).matmul(right));
        }
        if (bias != null) {
            y = y.add(bias);
        }
        int[] dims = shape.toArray();
        dims[dims.length - 1] = out;
        return activation.forward(Var.of(y)).reshape(dims);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "LowRankOutlierQuant[" + in + " -> " + out + ", W" + bits + "A" + bits + ", rank "
                + (left == null ? 0 : left.shape().dim(1)) + "]";
    }
}
