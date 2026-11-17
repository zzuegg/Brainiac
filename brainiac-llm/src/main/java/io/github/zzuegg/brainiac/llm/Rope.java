package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Rotary position embeddings as Llama-family models use them: each head's features are
 * split in two halves, and feature {@code i} of the first half and {@code i} of the second
 * are rotated together by {@code position * frequency(i)}.
 */
public final class Rope {

    /**
     * Rotary settings.
     *
     * @param dim the rotated features per head
     * @param theta the frequency base
     * @param type {@code "default"}, {@code "linear"} or {@code "llama3"}
     * @param factor the scaling factor (linear and llama3)
     * @param lowFreqFactor llama3's low-frequency factor
     * @param highFreqFactor llama3's high-frequency factor
     * @param originalMaxPositions llama3's original context length
     * @param freqFactors divisors per frequency (GGUF's {@code rope_freqs}), or {@code null}
     */
    public record Settings(int dim, double theta, String type, double factor, double lowFreqFactor,
            double highFreqFactor, int originalMaxPositions, float[] freqFactors) {

        /**
         * Returns these settings with per-frequency divisors.
         *
         * @param factors one divisor per frequency
         * @return the new settings
         */
        public Settings withFreqFactors(float[] factors) {
            return new Settings(dim, theta, type, factor, lowFreqFactor, highFreqFactor, originalMaxPositions,
                    factors);
        }
    }

    private final double[] inv;

    /**
     * Computes the frequencies.
     *
     * @param s the settings
     */
    public Rope(Settings s) {
        int half = s.dim() / 2;
        inv = new double[half];
        for (int i = 0; i < half; i++) {
            // as transformers computes it: in float32
            inv[i] = 1.0 / (float) Math.pow(s.theta(), (float) (2.0 * i) / s.dim());
        }
        switch (s.type()) {
            case "default" -> {
            }
            case "linear" -> {
                for (int i = 0; i < half; i++) {
                    inv[i] /= s.factor();
                }
            }
            case "llama3" -> {
                double low = s.originalMaxPositions() / s.lowFreqFactor();
                double high = s.originalMaxPositions() / s.highFreqFactor();
                for (int i = 0; i < half; i++) {
                    double wavelen = 2 * Math.PI / inv[i];
                    double f = wavelen > low ? inv[i] / s.factor() : inv[i];
                    if (!(wavelen < high) && !(wavelen > low)) {
                        double smooth = (s.originalMaxPositions() / wavelen - s.lowFreqFactor())
                                / (s.highFreqFactor() - s.lowFreqFactor());
                        f = (1 - smooth) * f / s.factor() + smooth * f;
                    }
                    inv[i] = f;
                }
            }
            default -> throw new IllegalArgumentException("rope scaling " + s.type() + " is not supported yet");
        }
        if (s.freqFactors() != null) {
            for (int i = 0; i < half; i++) {
                inv[i] /= s.freqFactors()[i];
            }
        }
    }

    /**
     * Rotates queries or keys.
     *
     * @param x {@code [batch, heads, steps, dim]}
     * @param positions the position of each step, {@code [batch][steps]}
     * @return the rotated values
     */
    public Var apply(Var x, int[][] positions) {
        Shape s = x.shape();
        int batch = s.dim(0);
        int steps = s.dim(2);
        int d = s.dim(3);
        int half = d / 2;
        Device device = x.value().device();
        DType type = x.value().dtype();
        Tensor[] cached = last;
        if (cached != null && lastPositions == positions && cached[0].dtype() == type
                && cached[0].device().equals(device) && cached[0].shape().equals(Shape.of(batch, 1, steps, 2, half))) {
            return rotate(x, cached[0], cached[1], half);   // every block of one step: the same angles
        }
        // per position, both halves: cos and cos, and -sin and sin
        float[] cos = new float[batch * steps * d];
        float[] sin = new float[batch * steps * d];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                int row = (b * steps + t) * d;
                for (int i = 0; i < half; i++) {
                    float angle = (float) positions[b][t] * (float) inv[i];
                    float cs = (float) Math.cos(angle);
                    float sn = (float) Math.sin(angle);
                    cos[row + i] = cs;
                    cos[row + half + i] = cs;
                    sin[row + i] = -sn;
                    sin[row + half + i] = sn;
                }
            }
        }
        Tensor c = Tensor.of(Shape.of(batch, 1, steps, 2, half), cos, device).to(type);
        Tensor sn = Tensor.of(Shape.of(batch, 1, steps, 2, half), sin, device).to(type);
        Tensor.evaluate(List.of(c, sn));
        last = new Tensor[] {c, sn};
        lastPositions = positions;
        return rotate(x, c, sn, half);
    }

    /** The angles of the last call, which the next call with the same positions array reuses. */
    private Tensor[] last;
    private int[][] lastPositions;

    /**
     * {@code [x1, x2] -> [x1 cos - x2 sin, x2 cos + x1 sin]} as {@code x * c + flip(x) * s} over
     * the two halves: element-wise only, so a GPU computes it in one pass.
     */
    private static Var rotate(Var x, Tensor c, Tensor sn, int half) {
        int[] s = x.shape().toArray();
        Var y = x.reshape(s[0], s[1], s[2], 2, half);
        return y.mul(c).add(y.flip(3).mul(sn)).reshape(s);
    }
}
