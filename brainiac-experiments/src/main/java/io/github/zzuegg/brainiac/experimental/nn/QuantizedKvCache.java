package io.github.zzuegg.brainiac.experimental.nn;

import java.util.ArrayList;
import java.util.List;

/**
 * KIVI: a key/value cache kept in 2 (or 4) bits without any calibration. Keys and values have
 * different outlier structure, so they are quantized along different axes: keys per
 * <em>channel</em> (a few channels are large for every token, so a per-channel range absorbs
 * them) over groups of {@code group} tokens, values per <em>token</em> (each token's value
 * vector is mixed by the attention weights, so per-token error stays local). The latest
 * {@code residual} tokens stay in full precision, which keeps the tokens the next steps attend
 * to most exact and lets keys be quantized one whole group at a time. Each quantized number
 * is {@code bits} wide with an fp16 scale and zero point per group, so 2-bit KIVI holds the
 * cache in about an eighth of fp16's bytes and a sixteenth of float32's.
 *
 * <p>The cache serves one attention layer: {@code kvHeads} key/value heads of {@code headDim},
 * and queries with any multiple of {@code kvHeads} heads (grouped-query attention).
 * {@link #append} adds a token's keys and values (already rotated), {@link #attend} reads
 * them dequantized, and {@link #bytes} counts what it holds, metadata included.
 * {@link KeyAxis#TOKEN} quantizes keys per token instead, the ablation that shows why the
 * per-channel axis matters.
 *
 * <p>Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (ICML 2024),
 * <a href="https://arxiv.org/abs/2402.02750">arXiv:2402.02750</a>.
 */
public final class QuantizedKvCache {

    /** The axis keys are quantized along. */
    public enum KeyAxis {
        /** KIVI: one range per channel over a group of tokens. */
        CHANNEL,
        /** One range per token and head, as values are (the ablation). */
        TOKEN
    }

    /** Bytes of a stored scale or zero point (fp16). */
    private static final int META = 2;

    private final int kvHeads;
    private final int headDim;
    private final int width;
    private final int bits;
    private final int levels;
    private final int group;
    private final int residual;
    private KeyAxis keyAxis = KeyAxis.CHANNEL;
    /** Dequantized keys and values of the quantized part, oldest first. */
    private final List<float[]> keys = new ArrayList<>();
    private final List<float[]> values = new ArrayList<>();
    /** Full-precision tokens not yet quantized. */
    private final List<float[]> recentKeys = new ArrayList<>();
    private final List<float[]> recentValues = new ArrayList<>();

    /**
     * @param kvHeads key/value heads
     * @param headDim the size of each head
     * @param bits bits per quantized number, 2 or 4 in the paper
     * @param group tokens per key group (32 in the paper)
     * @param residual latest tokens kept in full precision (at least 0; the paper uses 32 to 128)
     */
    public QuantizedKvCache(int kvHeads, int headDim, int bits, int group, int residual) {
        if (kvHeads < 1 || headDim < 1 || bits < 1 || bits > 8 || group < 1 || residual < 0) {
            throw new IllegalArgumentException("bad cache shape");
        }
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.width = kvHeads * headDim;
        this.bits = bits;
        this.levels = (1 << bits) - 1;
        this.group = group;
        this.residual = residual;
    }

    /**
     * @param axis the axis keys are quantized along
     * @return this cache
     */
    public QuantizedKvCache keyAxis(KeyAxis axis) {
        this.keyAxis = java.util.Objects.requireNonNull(axis, "axis");
        return this;
    }

    /** @return the tokens held */
    public int size() {
        return keys.size() + recentKeys.size();
    }

    /** @return the tokens held in full precision */
    public int fullPrecisionTokens() {
        return recentKeys.size();
    }

    /**
     * Adds one token.
     *
     * @param key its keys, {@code kvHeads * headDim}
     * @param value its values, {@code kvHeads * headDim}
     */
    public void append(float[] key, float[] value) {
        if (key.length != width || value.length != width) {
            throw new IllegalArgumentException("expected " + width + " numbers");
        }
        recentKeys.add(key.clone());
        recentValues.add(value.clone());
        if (recentKeys.size() >= residual + group) {
            List<float[]> k = new ArrayList<>(recentKeys.subList(0, group));
            List<float[]> v = new ArrayList<>(recentValues.subList(0, group));
            recentKeys.subList(0, group).clear();
            recentValues.subList(0, group).clear();
            if (keyAxis == KeyAxis.CHANNEL) {
                keys.addAll(perChannel(k));
            } else {
                for (float[] row : k) {
                    keys.add(perToken(row));
                }
            }
            for (float[] row : v) {
                values.add(perToken(row));
            }
        }
    }

    /** Asymmetric quantization of each channel over the group's tokens, dequantized. */
    private List<float[]> perChannel(List<float[]> rows) {
        List<float[]> out = new ArrayList<>(rows.size());
        for (int i = 0; i < rows.size(); i++) {
            out.add(new float[width]);
        }
        for (int c = 0; c < width; c++) {
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (float[] r : rows) {
                lo = Math.min(lo, r[c]);
                hi = Math.max(hi, r[c]);
            }
            for (int i = 0; i < rows.size(); i++) {
                out.get(i)[c] = roundTrip(rows.get(i)[c], lo, hi);
            }
        }
        return out;
    }

    /** Asymmetric quantization of one token, one range per head, dequantized. */
    private float[] perToken(float[] row) {
        float[] out = new float[width];
        for (int h = 0; h < kvHeads; h++) {
            float lo = Float.POSITIVE_INFINITY;
            float hi = Float.NEGATIVE_INFINITY;
            for (int c = h * headDim; c < (h + 1) * headDim; c++) {
                lo = Math.min(lo, row[c]);
                hi = Math.max(hi, row[c]);
            }
            for (int c = h * headDim; c < (h + 1) * headDim; c++) {
                out[c] = roundTrip(row[c], lo, hi);
            }
        }
        return out;
    }

    private float roundTrip(float x, float lo, float hi) {
        float scale = (hi - lo) / levels;
        if (scale == 0) {
            return lo;
        }
        int q = Math.round((x - lo) / scale);
        return lo + Math.min(levels, Math.max(0, q)) * scale;
    }

    /**
     * Attention of one query over every cached token: {@code softmax(q k^T * scale) v} per head.
     *
     * @param query {@code heads * headDim}, heads a multiple of {@code kvHeads}
     * @param scale the score scale, usually {@code 1 / sqrt(headDim)}
     * @return {@code heads * headDim}
     */
    public float[] attend(float[] query, float scale) {
        int heads = query.length / headDim;
        int perKv = heads / kvHeads;
        int n = size();
        float[] out = new float[query.length];
        double[] w = new double[n];
        for (int h = 0; h < heads; h++) {
            int off = (h / perKv) * headDim;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                float[] k = j < keys.size() ? keys.get(j) : recentKeys.get(j - keys.size());
                double dot = 0;
                for (int c = 0; c < headDim; c++) {
                    dot += query[h * headDim + c] * k[off + c];
                }
                w[j] = dot * scale;
                max = Math.max(max, w[j]);
            }
            double sum = 0;
            for (int j = 0; j < n; j++) {
                w[j] = Math.exp(w[j] - max);
                sum += w[j];
            }
            for (int j = 0; j < n; j++) {
                float[] v = j < values.size() ? values.get(j) : recentValues.get(j - values.size());
                float a = (float) (w[j] / sum);
                for (int c = 0; c < headDim; c++) {
                    out[h * headDim + c] += a * v[off + c];
                }
            }
        }
        return out;
    }

    /**
     * The bytes held: packed {@code bits}-wide numbers plus an fp16 scale and zero point per
     * key group and channel (or per token and head) and per value token and head, plus float32
     * for the full-precision tokens.
     *
     * @return the bytes
     */
    public long bytes() {
        long quantized = keys.size();
        long numbers = 2 * quantized * width;
        long packed = (numbers * bits + 7) / 8;
        long keyMeta = keyAxis == KeyAxis.CHANNEL ? (quantized / group) * width * 2L * META
                : quantized * kvHeads * 2L * META;
        long valueMeta = quantized * kvHeads * 2L * META;
        long full = 2L * recentKeys.size() * width * Float.BYTES;
        return packed + keyMeta + valueMeta + full;
    }

    /**
     * @param tokens tokens held
     * @param kvHeads key/value heads
     * @param headDim the head size
     * @return the bytes float32 keys and values take
     */
    public static long float32Bytes(int tokens, int kvHeads, int headDim) {
        return 2L * tokens * kvHeads * headDim * Float.BYTES;
    }

    @Override
    public String toString() {
        return "QuantizedKvCache[" + bits + "-bit, " + size() + " tokens, " + fullPrecisionTokens()
                + " in full precision]";
    }
}
