package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * SmoothQuant: makes activations quantizable for W8A8 (8-bit weights <em>and</em> 8-bit
 * activations) inference. LLM activations have a few outlier channels 100x larger than the
 * rest, so one per-tensor activation scale wastes almost all levels on them; weights have no
 * such outliers. Dividing activation channel {@code j} by
 * {@code s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)} and multiplying weight row {@code j} by
 * it leaves {@code X W} unchanged but migrates the difficulty from activations into weights,
 * where per-channel quantization absorbs it. The division is folded into the preceding
 * LayerNorm or linear layer, so it costs nothing at inference.
 *
 * <p>Weights are {@code [inputs, outputs]}; activations {@code [samples, inputs]}.
 *
 * <p>Paper: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for
 * Large Language Models" (2022), <a href="https://arxiv.org/abs/2211.10438">arXiv:2211.10438</a>.
 */
public final class SmoothQuant {

    private SmoothQuant() {
    }

    /**
     * The smoothing scales.
     *
     * @param activations calibration activations {@code [samples, inputs]}
     * @param weight the weight {@code [inputs, outputs]}
     * @param alpha the migration strength, 0.5 in the paper (higher moves more into weights)
     * @return one scale per input channel
     */
    public static double[] scales(Tensor activations, Tensor weight, float alpha) {
        double[][] x = QuantMath.matrix(activations);
        double[][] w = QuantMath.matrix(weight);
        int d = w.length;
        double[] s = new double[d];
        for (int j = 0; j < d; j++) {
            double ax = 0;
            for (double[] row : x) {
                ax = Math.max(ax, Math.abs(row[j]));
            }
            double aw = 0;
            for (double v : w[j]) {
                aw = Math.max(aw, Math.abs(v));
            }
            s[j] = ax == 0 || aw == 0 ? 1 : Math.pow(ax, alpha) / Math.pow(aw, 1 - alpha);
        }
        return s;
    }

    /**
     * The mean squared output error of simulated W8A8 (or other widths) inference, with
     * per-tensor symmetric activation quantization and per-channel weight quantization.
     *
     * @param activations the activations {@code [samples, inputs]}
     * @param weight the weight {@code [inputs, outputs]}
     * @param scales smoothing scales, or {@code null} for no smoothing
     * @param bits bits for both activations and weights
     * @return the mean squared error against the full-precision {@code X W}
     */
    public static double quantizedError(Tensor activations, Tensor weight, double[] scales, int bits) {
        double[][] x = QuantMath.matrix(activations);
        double[][] w = QuantMath.matrix(weight);
        int d = w.length;
        int out = w[0].length;
        double[][] xs = new double[x.length][d];
        double[][] ws = new double[d][out];
        for (int j = 0; j < d; j++) {
            double s = scales == null ? 1 : scales[j];
            for (int n = 0; n < x.length; n++) {
                xs[n][j] = x[n][j] / s;
            }
            for (int c = 0; c < out; c++) {
                ws[j][c] = w[j][c] * s;
            }
        }
        int levels = QuantMath.levels(bits);
        double amax = 0;
        for (double[] row : xs) {
            for (double v : row) {
                amax = Math.max(amax, Math.abs(v));
            }
        }
        double xscale = amax == 0 ? 1 : amax / levels;
        for (double[] row : xs) {
            for (int j = 0; j < d; j++) {
                row[j] = QuantMath.quantize(row[j], xscale, levels);
            }
        }
        double[][] wq = QuantMath.roundToNearest(ws, bits);
        double err = 0;
        for (int n = 0; n < x.length; n++) {
            for (int c = 0; c < out; c++) {
                double ref = 0;
                double got = 0;
                for (int j = 0; j < d; j++) {
                    ref += x[n][j] * w[j][c];
                    got += xs[n][j] * wq[j][c];
                }
                err += (ref - got) * (ref - got);
            }
        }
        return err / (x.length * (double) out);
    }
}
