package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * GPTQ post-training weight quantization: quantizes a dense layer's weights one input
 * dimension at a time and, after each, updates the not-yet-quantized weights to cancel the
 * rounding error it caused, using the inverse Hessian {@code (2 X^T X)^-1} of the layer's
 * squared output error on calibration inputs {@code X}. This is the Optimal Brain Surgeon
 * update made fast with a Cholesky factor, and it keeps 3-4-bit LLMs close to full
 * precision where plain round-to-nearest falls apart.
 *
 * <p>Weights are Brainiac's {@code [inputs, outputs]} layout (a {@code Dense} weight), with
 * one symmetric scale per output channel, as {@code nn.Quantization.perChannel()} uses; the
 * result is the dequantized weight, ready to {@code assign} back (then
 * {@code model.quantize(...)} stores it on the same grid). With {@link #actOrder} the
 * inputs with the largest Hessian diagonal (most active features) are quantized first.
 *
 * <p>Paper: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
 * Pre-trained Transformers" (2022), <a href="https://arxiv.org/abs/2210.17323">arXiv:2210.17323</a>.
 */
public final class Gptq {

    private final int bits;
    private float damping = 0.01f;
    private boolean actOrder;

    /**
     * Creates the quantizer.
     *
     * @param bits the bits per weight, 2 to 8 typically
     */
    public Gptq(int bits) {
        QuantMath.levels(bits);
        this.bits = bits;
    }

    /**
     * Sets the Hessian damping, a share of the mean diagonal added to it for stability.
     *
     * @param share the share, 0.01 by default
     * @return this
     */
    public Gptq damping(float share) {
        this.damping = share;
        return this;
    }

    /**
     * Quantizes the inputs with the largest Hessian diagonal first ("act-order").
     *
     * @param on whether to reorder
     * @return this
     */
    public Gptq actOrder(boolean on) {
        this.actOrder = on;
        return this;
    }

    /**
     * Quantizes a weight matrix.
     *
     * @param weight the weight {@code [inputs, outputs]}
     * @param calibration calibration inputs to the layer {@code [samples, inputs]}
     * @return the dequantized weight, same shape, on the grid of per-channel symmetric
     *     {@code bits}-bit integers
     */
    public Tensor quantize(Tensor weight, Tensor calibration) {
        double[][] w = QuantMath.matrix(weight);
        double[][] x = QuantMath.matrix(calibration);
        int d = w.length;
        int out = w[0].length;
        int levels = QuantMath.levels(bits);
        double[] scales = QuantMath.columnScales(w, levels);
        double[][] h = QuantMath.gram(x);
        int[] order = new int[d];
        for (int i = 0; i < d; i++) {
            order[i] = i;
        }
        if (actOrder) {
            Integer[] boxed = new Integer[d];
            for (int i = 0; i < d; i++) {
                boxed[i] = i;
            }
            java.util.Arrays.sort(boxed, (a, b) -> Double.compare(h[b][b], h[a][a]));
            for (int i = 0; i < d; i++) {
                order[i] = boxed[i];
            }
        }
        // permute rows of w and rows/cols of h into processing order
        double[][] wp = new double[d][];
        double[][] hp = new double[d][d];
        double meanDiag = 0;
        for (int i = 0; i < d; i++) {
            wp[i] = w[order[i]].clone();
            for (int j = 0; j < d; j++) {
                hp[i][j] = 2 * h[order[i]][order[j]];
            }
            meanDiag += hp[i][i] / d;
        }
        for (int i = 0; i < d; i++) {
            if (hp[i][i] == 0) {
                hp[i][i] = 1;   // a dead input: any value is fine
            }
            hp[i][i] += damping * Math.max(meanDiag, 1e-8);
        }
        double[][] u = QuantMath.choleskyUpper(QuantMath.inverse(hp));
        double[][] q = new double[d][out];
        for (int i = 0; i < d; i++) {
            for (int c = 0; c < out; c++) {
                q[i][c] = QuantMath.quantize(wp[i][c], scales[c], levels);
            }
            for (int c = 0; c < out; c++) {
                double err = (wp[i][c] - q[i][c]) / u[i][i];
                for (int j = i + 1; j < d; j++) {
                    wp[j][c] -= err * u[i][j];
                }
            }
        }
        double[][] result = new double[d][];
        for (int i = 0; i < d; i++) {
            result[order[i]] = q[i];
        }
        return QuantMath.tensor(result, weight.device());
    }

    /**
     * Round-to-nearest with the same grid, the baseline GPTQ improves on.
     *
     * @param weight the weight {@code [inputs, outputs]}
     * @param bits the bits per weight
     * @return the dequantized weight
     */
    public static Tensor roundToNearest(Tensor weight, int bits) {
        return QuantMath.tensor(QuantMath.roundToNearest(QuantMath.matrix(weight), bits), weight.device());
    }

    /**
     * The mean squared difference of a layer's outputs under two weights.
     *
     * @param inputs the inputs {@code [samples, inputs]}
     * @param reference the reference weight
     * @param quantized the other weight
     * @return the mean squared output error
     */
    public static double outputError(Tensor inputs, Tensor reference, Tensor quantized) {
        return QuantMath.outputError(QuantMath.matrix(inputs), QuantMath.matrix(reference),
                QuantMath.matrix(quantized));
    }
}
