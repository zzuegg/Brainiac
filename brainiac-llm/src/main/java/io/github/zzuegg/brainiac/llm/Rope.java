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
     * @param type {@code "default"}, {@code "linear"}, {@code "llama3"}, {@code "yarn"} (Peng et
     *     al., 2023), {@code "dynamic"} (NTK-aware, the base grows once positions pass
     *     {@code originalMaxPositions}), {@code "bf16"} (the default frequencies, with cos and sin
     *     rounded to bfloat16 as Fish Speech's tables are) or {@code "longrope"} (Phi-3: the divisors are {@code
     *     freqFactors}, and cos and sin are scaled by {@code sqrt(1 + ln factor / ln original)})
     * @param factor the scaling factor (every type but default)
     * @param lowFreqFactor llama3's low-frequency factor
     * @param highFreqFactor llama3's high-frequency factor
     * @param originalMaxPositions the original context length (llama3, yarn and dynamic)
     * @param freqFactors divisors per frequency (GGUF's {@code rope_freqs}), or {@code null}
     * @param betaFast yarn: frequencies with more rotations than this over the original context
     *     are kept
     * @param betaSlow yarn: frequencies with fewer rotations than this are interpolated
     * @param attentionFactor yarn and longrope: the factor on cos and sin, or 0 for the type's default
     */
    public record Settings(int dim, double theta, String type, double factor, double lowFreqFactor,
            double highFreqFactor, int originalMaxPositions, float[] freqFactors, double betaFast,
            double betaSlow, double attentionFactor) {

        /**
         * Settings with yarn's default bands (32 and 1) and attention factor.
         *
         * @param dim the rotated features per head
         * @param theta the frequency base
         * @param type the scaling type
         * @param factor the scaling factor
         * @param lowFreqFactor llama3's low-frequency factor
         * @param highFreqFactor llama3's high-frequency factor
         * @param originalMaxPositions the original context length
         * @param freqFactors divisors per frequency, or {@code null}
         */
        public Settings(int dim, double theta, String type, double factor, double lowFreqFactor,
                double highFreqFactor, int originalMaxPositions, float[] freqFactors) {
            this(dim, theta, type, factor, lowFreqFactor, highFreqFactor, originalMaxPositions, freqFactors, 32, 1,
                    0);
        }

        /**
         * Returns these settings with per-frequency divisors.
         *
         * @param factors one divisor per frequency
         * @return the new settings
         */
        public Settings withFreqFactors(float[] factors) {
            return new Settings(dim, theta, type, factor, lowFreqFactor, highFreqFactor, originalMaxPositions,
                    factors, betaFast, betaSlow, attentionFactor);
        }

        /**
         * Returns these settings with yarn's bands and attention factor.
         *
         * @param fast the rotations above which frequencies are kept
         * @param slow the rotations below which frequencies are interpolated
         * @param attention the factor on cos and sin, or 0 for the default
         * @return the new settings
         */
        public Settings withYarn(double fast, double slow, double attention) {
            return new Settings(dim, theta, type, factor, lowFreqFactor, highFreqFactor, originalMaxPositions,
                    freqFactors, fast, slow, attention);
        }
    }

    private final Settings settings;
    /** Whether rotated pairs are neighbours ({@code x0, x1}) rather than halves. */
    private final boolean interleaved;
    private double[] inv;
    /** The factor on cos and sin: yarn's attention scaling, otherwise 1. */
    private final float magnitude;
    /** The context length the current dynamic frequencies were made for. */
    private int dynamicLength;

    /**
     * Computes the frequencies.
     *
     * @param s the settings
     */
    public Rope(Settings s) {
        this(s, false);
    }

    /**
     * Computes the frequencies, for features that pair up as halves or as neighbours.
     *
     * @param s the settings
     * @param interleaved whether feature {@code 2i} and {@code 2i + 1} rotate together (as
     *     DeepSeek's weights and llama.cpp's "normal" rotation hold them) rather than {@code i}
     *     and {@code i + dim / 2}
     */
    public Rope(Settings s, boolean interleaved) {
        this.interleaved = interleaved;
        this.settings = s;
        this.inv = frequencies(s, s.theta());
        double m = 1;
        if (s.type().equals("yarn")) {
            m = s.attentionFactor() > 0 ? s.attentionFactor() : s.factor() <= 1 ? 1 : 0.1 * Math.log(s.factor()) + 1;
        } else if (s.type().equals("longrope")) {
            m = s.attentionFactor() > 0 ? s.attentionFactor() : s.factor() <= 1 ? 1
                    : Math.sqrt(1 + Math.log(s.factor()) / Math.log(s.originalMaxPositions()));
        }
        this.magnitude = (float) m;
        this.dynamicLength = s.originalMaxPositions();
    }

    /** @return the inverse frequencies in use, one per rotated pair */
    public double[] frequencies() {
        return inv.clone();
    }

    /** @return the factor on cos and sin (yarn's attention scaling, otherwise 1) */
    public double magnitude() {
        return magnitude;
    }

    private static double[] frequencies(Settings s, double theta) {
        int half = s.dim() / 2;
        double[] inv = new double[half];
        for (int i = 0; i < half; i++) {
            // as transformers computes it: in float32
            inv[i] = 1.0 / (float) Math.pow(theta, (float) (2.0 * i) / s.dim());
        }
        switch (s.type()) {
            case "default", "dynamic", "longrope", "bf16" -> {
                // longrope (Phi-3): its per-pair divisors are the freqFactors
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
            case "yarn" -> {
                // as transformers: pairs with many rotations over the original context keep
                // their frequency, those with few are interpolated, a linear ramp between
                double low = Math.max(0, Math.floor(correctionDim(s.betaFast(), s.dim(), theta,
                        s.originalMaxPositions())));
                double high = Math.min(s.dim() - 1, Math.ceil(correctionDim(s.betaSlow(), s.dim(), theta,
                        s.originalMaxPositions())));
                if (low == high) {
                    high += 0.001;
                }
                for (int i = 0; i < half; i++) {
                    double ramp = Math.clamp((i - low) / (high - low), 0, 1);
                    double keep = 1 - ramp;
                    inv[i] = inv[i] / s.factor() * (1 - keep) + inv[i] * keep;
                }
            }
            default -> throw new IllegalArgumentException("rope scaling " + s.type() + " is not supported yet");
        }
        if (s.freqFactors() != null) {
            for (int i = 0; i < half; i++) {
                inv[i] /= s.freqFactors()[i];
            }
        }
        return inv;
    }

    /** The pair index whose wavelength fits {@code rotations} times into {@code positions}. */
    private static double correctionDim(double rotations, int dim, double theta, int positions) {
        return dim * Math.log(positions / (rotations * 2 * Math.PI)) / (2 * Math.log(theta));
    }

    /**
     * Dynamic NTK: once positions reach past the original context, the base grows to
     * {@code theta (factor L / original - (factor - 1))^(dim / (dim - 2))} for length {@code L},
     * as transformers does. Keys already cached keep the angles they were read with.
     */
    private void grow(int[][] positions) {
        int longest = 0;
        for (int[] row : positions) {
            for (int p : row) {
                longest = Math.max(longest, p + 1);
            }
        }
        if (longest <= dynamicLength) {
            return;
        }
        dynamicLength = longest;
        Settings s = settings;
        double base = s.theta() * Math.pow(s.factor() * longest / s.originalMaxPositions() - (s.factor() - 1),
                s.dim() / (s.dim() - 2.0));
        inv = frequencies(s, base);
        last = null;
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
        if (settings.type().equals("dynamic")) {
            grow(positions);
        }
        double[] inv = this.inv;
        Device device = x.value().device();
        DType type = x.value().dtype();
        Tensor[] cached = last;
        if (cached != null && lastPositions == positions && cached[0].dtype() == type
                && cached[0].device().equals(device) && cached[0].shape().equals(angles(batch, steps, half))) {
            return rotate(x, cached[0], cached[1], half, interleaved);   // every block of one step: the same angles
        }
        // per position, both halves: cos and cos, and -sin and sin
        float[] cos = new float[batch * steps * d];
        float[] sin = new float[batch * steps * d];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                int row = (b * steps + t) * d;
                for (int i = 0; i < half; i++) {
                    float angle = (float) positions[b][t] * (float) inv[i];
                    put(cos, sin, row, i, half, table((float) Math.cos(angle) * magnitude),
                            table((float) Math.sin(angle) * magnitude));
                }
            }
        }
        Tensor c = Tensor.of(angles(batch, steps, half), cos, device).to(type);
        Tensor sn = Tensor.of(angles(batch, steps, half), sin, device).to(type);
        Tensor.evaluate(List.of(c, sn));
        last = new Tensor[] {c, sn};
        lastPositions = positions;
        lastMulti = null;
        return rotate(x, c, sn, half, interleaved);
    }

    /**
     * Rotates queries or keys by 3D positions, as Qwen2-VL: the frequency pairs split in
     * {@code sections} (time, height, width), each rotating by its own position.
     *
     * @param x {@code [batch, heads, steps, dim]}
     * @param positions {@code [3][batch][steps]}
     * @param sections how many frequency pairs each of the three takes, summing to {@code dim / 2}
     * @return the rotated values
     */
    public Var apply(Var x, int[][][] positions, int[] sections) {
        Shape s = x.shape();
        int batch = s.dim(0);
        int steps = s.dim(2);
        int d = s.dim(3);
        int half = d / 2;
        Device device = x.value().device();
        DType type = x.value().dtype();
        Tensor[] cached = last;
        if (cached != null && lastMulti == positions && cached[0].dtype() == type
                && cached[0].device().equals(device) && cached[0].shape().equals(angles(batch, steps, half))) {
            return rotate(x, cached[0], cached[1], half, interleaved);
        }
        int[] sectionOf = new int[half];
        for (int i = 0, sec = 0, upto = sections[0]; i < half; i++) {
            while (i >= upto && sec < 2) {
                upto += sections[++sec];
            }
            sectionOf[i] = sec;
        }
        float[] cos = new float[batch * steps * d];
        float[] sin = new float[batch * steps * d];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                int row = (b * steps + t) * d;
                for (int i = 0; i < half; i++) {
                    float angle = (float) positions[sectionOf[i]][b][t] * (float) inv[i];
                    put(cos, sin, row, i, half, table((float) Math.cos(angle) * magnitude),
                            table((float) Math.sin(angle) * magnitude));
                }
            }
        }
        Tensor c = Tensor.of(angles(batch, steps, half), cos, device).to(type);
        Tensor sn = Tensor.of(angles(batch, steps, half), sin, device).to(type);
        Tensor.evaluate(List.of(c, sn));
        last = new Tensor[] {c, sn};
        lastPositions = null;
        lastMulti = positions;
        return rotate(x, c, sn, half, interleaved);
    }

    private int[][][] lastMulti;

    /** The shape of the angle tables: per step, the two halves or the pairs. */
    private Shape angles(int batch, int steps, int half) {
        return interleaved ? Shape.of(batch, 1, steps, half, 2) : Shape.of(batch, 1, steps, 2, half);
    }

    /** A table value: as computed, or rounded to bfloat16 (to nearest, ties to even) for type bf16. */
    private float table(float v) {
        if (!settings.type().equals("bf16")) {
            return v;
        }
        int bits = Float.floatToRawIntBits(v);
        bits += 0x7FFF + ((bits >>> 16) & 1);
        return Float.intBitsToFloat(bits & 0xFFFF0000);
    }

    /** Pair {@code i}'s cos and sin in a step's row: {@code cos, cos} and {@code -sin, sin}. */
    private void put(float[] cos, float[] sin, int row, int i, int half, float cs, float sn) {
        int a = interleaved ? row + 2 * i : row + i;
        int b = interleaved ? a + 1 : row + half + i;
        cos[a] = cs;
        cos[b] = cs;
        sin[a] = -sn;
        sin[b] = sn;
    }

    /** The angles of the last call, which the next call with the same positions array reuses. */
    private Tensor[] last;
    private int[][] lastPositions;

    /**
     * {@code [x1, x2] -> [x1 cos - x2 sin, x2 cos + x1 sin]} as {@code x * c + flip(x) * s} over
     * the two halves (or within each pair, interleaved): element-wise only, so a GPU computes
     * it in one pass.
     */
    private static Var rotate(Var x, Tensor c, Tensor sn, int half, boolean interleaved) {
        int[] s = x.shape().toArray();
        Var y = interleaved ? x.reshape(s[0], s[1], s[2], half, 2) : x.reshape(s[0], s[1], s[2], 2, half);
        return y.mul(c).add(y.flip(interleaved ? 4 : 3).mul(sn)).reshape(s);
    }
}
