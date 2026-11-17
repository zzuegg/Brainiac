package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * AWQ, activation-aware weight quantization: a small share of input channels carry much
 * larger activations, and the weights they multiply matter most. Scaling those weight rows
 * up by {@code s_i = mean|x_i|^alpha} before rounding (and the activations down by the same
 * factor, folded into the previous layer) gives them relatively finer quantization steps.
 * {@code alpha} is found by a grid search that minimizes the layer's output error on
 * calibration data. No back-propagation and no reconstruction, so it is fast and does not
 * overfit the calibration set.
 *
 * <p>Weights are {@code [inputs, outputs]} with per-output-channel symmetric scales.
 *
 * <p>Paper: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and
 * Acceleration" (2023), <a href="https://arxiv.org/abs/2306.00978">arXiv:2306.00978</a>.
 */
public final class Awq {

    /**
     * The search result.
     *
     * @param weight the effective dequantized weight {@code diag(1/s) Q(diag(s) W)}, which
     *     replaces {@code W} without changing the layer's input
     * @param scaled the quantized scaled weight {@code Q(diag(s) W)}, for when {@code 1/s}
     *     is folded into the previous layer instead
     * @param scales the per-input-channel scales {@code s}
     * @param alpha the best exponent
     * @param error the output error at that exponent
     */
    public record Result(Tensor weight, Tensor scaled, double[] scales, double alpha, double error) {
    }

    private Awq() {
    }

    /**
     * Searches the scaling exponent on a grid of 21 values in {@code [0, 1]}.
     *
     * @param weight the weight {@code [inputs, outputs]}
     * @param calibration calibration inputs {@code [samples, inputs]}
     * @param bits the bits per weight
     * @return the best scaling
     */
    public static Result quantize(Tensor weight, Tensor calibration, int bits) {
        double[][] w = QuantMath.matrix(weight);
        double[][] x = QuantMath.matrix(calibration);
        int d = w.length;
        int out = w[0].length;
        double[] mean = new double[d];
        for (double[] row : x) {
            for (int i = 0; i < d; i++) {
                mean[i] += Math.abs(row[i]) / x.length;
            }
        }
        Result best = null;
        for (int g = 0; g <= 20; g++) {
            double alpha = g / 20.0;
            double[] s = new double[d];
            double lo = Double.MAX_VALUE;
            double hi = 0;
            for (int i = 0; i < d; i++) {
                s[i] = Math.pow(Math.max(mean[i], 1e-8), alpha);
                lo = Math.min(lo, s[i]);
                hi = Math.max(hi, s[i]);
            }
            double norm = Math.sqrt(lo * hi);
            double[][] sw = new double[d][out];
            for (int i = 0; i < d; i++) {
                s[i] /= norm;
                for (int c = 0; c < out; c++) {
                    sw[i][c] = w[i][c] * s[i];
                }
            }
            double[][] q = QuantMath.roundToNearest(sw, bits);
            double[][] eff = new double[d][out];
            for (int i = 0; i < d; i++) {
                for (int c = 0; c < out; c++) {
                    eff[i][c] = q[i][c] / s[i];
                }
            }
            double err = QuantMath.outputError(x, w, eff);
            if (best == null || err < best.error()) {
                best = new Result(QuantMath.tensor(eff, weight.device()), QuantMath.tensor(q, weight.device()), s,
                        alpha, err);
            }
        }
        return best;
    }
}
