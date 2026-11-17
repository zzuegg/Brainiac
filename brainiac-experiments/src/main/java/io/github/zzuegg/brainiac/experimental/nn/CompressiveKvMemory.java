package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * A key/value cache that folds what falls out of its window into a fixed-size memory
 * instead of dropping it. The window holds the most recent keys and values exactly; when it
 * is full, the oldest entry goes to a {@link Mode}:
 * <ul>
 * <li>{@link Mode#DROP}: discarded, as a sliding-window cache does (the baseline);</li>
 * <li>{@link Mode#LINEAR}: added to a linear-attention memory {@code M += phi(k)^T v},
 *     {@code z += phi(k)} with {@code phi = elu + 1}, read as {@code phi(q) M / phi(q) z}
 *     (Infini-attention, Munkhdalai et al. 2024,
 *     <a href="https://arxiv.org/abs/2404.07143">arXiv:2404.07143</a>);</li>
 * <li>{@link Mode#DELTA}: the same memory updated by the delta rule
 *     {@code M += phi(k)^T (v - phi(k) M / phi(k) z)}, which overwrites instead of piling up
 *     when a key returns (the normalizer {@code z} is written by the same rule, so a key stored
 *     again adds no weight, a small deviation from the paper that keeps reads unbiased);</li>
 * <li>{@link Mode#POOL}: every {@code rate} evicted entries are mean-pooled into one
 *     compressed key/value slot that stays attendable (the Compressive Transformer, Rae et
 *     al. 2019, <a href="https://arxiv.org/abs/1911.05507">arXiv:1911.05507</a>).</li>
 * </ul>
 * Memory never grows: {@code dk x dv} per head for the linear modes, {@code slots} entries
 * for pooling. {@link #attend} mixes window attention and the memory read with a gate
 * {@code beta}, as Infini-attention does with a learned one.
 *
 * <p>Tensors are {@code [heads, steps, size]}. Inference only; for training the gated memory
 * end to end use {@link InfiniAttention}.
 */
public final class CompressiveKvMemory {

    /** What happens to entries evicted from the window. */
    public enum Mode {
        /** Discard them. */
        DROP,
        /** Add them to a linear-attention memory. */
        LINEAR,
        /** Write them into a linear-attention memory with the delta rule. */
        DELTA,
        /** Mean-pool them into compressed slots. */
        POOL
    }

    /** The positive feature map {@code phi} of the linear memory. */
    public enum FeatureMap {
        /** {@code elu(x) + 1}, as in Infini-attention and linear transformers. */
        ELU_PLUS_ONE,
        /**
         * Deterministic parameter-free projection (Schlag et al. 2021,
         * <a href="https://arxiv.org/abs/2102.11174">arXiv:2102.11174</a>): {@code r = relu([x, -x])},
         * {@code phi_i = r_i * r_(i+1)}. Twice as many, sparser features that interfere less.
         */
        DPFP
    }

    private final int heads;
    private final int keySize;
    private final int valueSize;
    private final int window;
    private final Mode mode;
    private final Deque<float[][]> recent = new ArrayDeque<>();
    private float[][] m;
    private float[][] z;
    private FeatureMap featureMap = FeatureMap.ELU_PLUS_ONE;
    private int phiSize;
    private final List<float[][]> pending = new ArrayList<>();
    private final Deque<float[][]> slots = new ArrayDeque<>();
    private int rate = 4;
    private int maxSlots = 16;
    private float beta = 0.5f;
    private long folded;

    private CompressiveKvMemory(int heads, int keySize, int valueSize, int window, Mode mode) {
        if (heads < 1 || keySize < 1 || valueSize < 1 || window < 1) {
            throw new IllegalArgumentException("sizes and window must be positive");
        }
        this.heads = heads;
        this.keySize = keySize;
        this.valueSize = valueSize;
        this.window = window;
        this.mode = mode;
        allocate();
    }

    private void allocate() {
        phiSize = featureMap == FeatureMap.DPFP ? 2 * keySize : keySize;
        m = new float[heads][phiSize * valueSize];
        z = new float[heads][phiSize];
    }

    /**
     * Chooses the feature map; clears the memory.
     *
     * @param map the feature map
     * @return this memory
     */
    public CompressiveKvMemory featureMap(FeatureMap map) {
        this.featureMap = java.util.Objects.requireNonNull(map, "map");
        reset();
        allocate();
        return this;
    }

    /**
     * Creates an empty memory.
     *
     * @param heads the heads
     * @param keySize the key size per head
     * @param valueSize the value size per head
     * @param window the entries kept exactly
     * @param mode what happens to older entries
     * @return the memory
     */
    public static CompressiveKvMemory create(int heads, int keySize, int valueSize, int window, Mode mode) {
        return new CompressiveKvMemory(heads, keySize, valueSize, window, mode);
    }

    /**
     * @param evicted for {@link Mode#POOL}, the evicted entries pooled into one slot
     * @param slots for {@link Mode#POOL}, the most slots kept
     * @return this memory
     */
    public CompressiveKvMemory pooling(int evicted, int slots) {
        if (evicted < 1 || slots < 1) {
            throw new IllegalArgumentException("pooling sizes must be positive");
        }
        this.rate = evicted;
        this.maxSlots = slots;
        return this;
    }

    /**
     * @param gate the share of {@link #attend}'s result taken from the linear memory, in [0, 1]
     * @return this memory
     */
    public CompressiveKvMemory gate(float gate) {
        if (gate < 0 || gate > 1) {
            throw new IllegalArgumentException("gate must be in [0, 1]");
        }
        this.beta = gate;
        return this;
    }

    /** @return the mode */
    public Mode mode() {
        return mode;
    }

    /** @return the entries in the window */
    public int windowSize() {
        return recent.size();
    }

    /** @return how many entries left the window so far */
    public long evicted() {
        return folded;
    }

    /** @return the floats the memory holds besides the window, constant for the linear modes */
    public long memoryFloats() {
        return switch (mode) {
            case DROP -> 0;
            case LINEAR, DELTA -> (long) heads * phiSize * (valueSize + 1);
            case POOL -> (long) (slots.size() + pending.size()) * heads * (keySize + valueSize);
        };
    }

    /**
     * Appends entries.
     *
     * @param keys {@code [heads, steps, keySize]}
     * @param values {@code [heads, steps, valueSize]}
     */
    public void append(Tensor keys, Tensor values) {
        int steps = keys.shape().dim(1);
        float[] k = keys.toArray();
        float[] v = values.toArray();
        for (int t = 0; t < steps; t++) {
            float[][] entry = new float[2 * heads][];
            for (int h = 0; h < heads; h++) {
                int ko = (h * steps + t) * keySize;
                int vo = (h * steps + t) * valueSize;
                entry[h] = java.util.Arrays.copyOfRange(k, ko, ko + keySize);
                entry[heads + h] = java.util.Arrays.copyOfRange(v, vo, vo + valueSize);
            }
            recent.addLast(entry);
            if (recent.size() > window) {
                evict(recent.removeFirst());
            }
        }
    }

    private void evict(float[][] entry) {
        folded++;
        switch (mode) {
            case DROP -> { }
            case LINEAR, DELTA -> {
                for (int h = 0; h < heads; h++) {
                    write(h, entry[h], entry[heads + h]);
                }
            }
            case POOL -> {
                pending.add(entry);
                if (pending.size() == rate) {
                    float[][] slot = new float[2 * heads][];
                    for (int i = 0; i < 2 * heads; i++) {
                        slot[i] = new float[pending.get(0)[i].length];
                        for (float[][] e : pending) {
                            for (int j = 0; j < slot[i].length; j++) {
                                slot[i][j] += e[i][j] / rate;
                            }
                        }
                    }
                    pending.clear();
                    slots.addLast(slot);
                    if (slots.size() > maxSlots) {
                        slots.removeFirst();
                    }
                }
            }
        }
    }

    private void write(int h, float[] key, float[] value) {
        float[] phi = phi(key);
        float[] mh = m[h];
        float[] target = value;
        if (mode == Mode.DELTA) {
            float[] old = read(h, phi);
            target = new float[valueSize];
            for (int j = 0; j < valueSize; j++) {
                target[j] = value[j] - old[j];
            }
        }
        float zScale = 1;
        if (mode == Mode.DELTA) {
            // the normalizer is written with the delta rule too: a key already stored adds no weight
            double seen = 0;
            double self = 0;
            for (int i = 0; i < phiSize; i++) {
                seen += phi[i] * z[h][i];
                self += phi[i] * phi[i];
            }
            zScale = self > 0 ? (float) Math.max(0, 1 - seen / self) : 1;
        }
        for (int i = 0; i < phiSize; i++) {
            for (int j = 0; j < valueSize; j++) {
                mh[i * valueSize + j] += phi[i] * target[j];
            }
            z[h][i] += zScale * phi[i];
        }
    }

    private float[] phi(float[] x) {
        if (featureMap == FeatureMap.DPFP) {
            int n = 2 * x.length;
            float[] r = new float[n];
            for (int i = 0; i < x.length; i++) {
                r[i] = Math.max(0, x[i]);
                r[x.length + i] = Math.max(0, -x[i]);
            }
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                y[i] = r[i] * r[(i + 1) % n];
            }
            return y;
        }
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] > 0 ? x[i] + 1 : (float) Math.exp(x[i]);
        }
        return y;
    }

    private float[] read(int h, float[] phiQ) {
        float[] out = new float[valueSize];
        double norm = 0;
        for (int i = 0; i < phiSize; i++) {
            norm += phiQ[i] * z[h][i];
        }
        if (norm <= 1e-9) {
            return out;
        }
        for (int i = 0; i < phiSize; i++) {
            float p = phiQ[i];
            if (p == 0) {
                continue;
            }
            for (int j = 0; j < valueSize; j++) {
                out[j] += p * m[h][i * valueSize + j];
            }
        }
        for (int j = 0; j < valueSize; j++) {
            out[j] /= (float) norm;
        }
        return out;
    }

    /**
     * Reads only the linear memory, {@code phi(q) M / phi(q) z} (zero for the other modes).
     *
     * @param queries {@code [heads, steps, keySize]}
     * @return {@code [heads, steps, valueSize]}
     */
    public Tensor readMemory(Tensor queries) {
        int steps = queries.shape().dim(1);
        float[] q = queries.toArray();
        float[] out = new float[heads * steps * valueSize];
        if (mode == Mode.LINEAR || mode == Mode.DELTA) {
            for (int h = 0; h < heads; h++) {
                for (int t = 0; t < steps; t++) {
                    int qo = (h * steps + t) * keySize;
                    float[] r = read(h, phi(java.util.Arrays.copyOfRange(q, qo, qo + keySize)));
                    System.arraycopy(r, 0, out, (h * steps + t) * valueSize, valueSize);
                }
            }
        }
        return Tensor.of(Shape.of(heads, steps, valueSize), out);
    }

    /**
     * Attends to the window (and, for {@link Mode#POOL}, the compressed slots) with softmax
     * attention and, for the linear modes, mixes in the memory read:
     * {@code beta * memory + (1 - beta) * window} once anything was folded.
     *
     * @param queries {@code [heads, steps, keySize]}
     * @param scale the score scale, usually {@code 1 / sqrt(keySize)}
     * @return {@code [heads, steps, valueSize]}
     */
    public Tensor attend(Tensor queries, float scale) {
        int steps = queries.shape().dim(1);
        float[] q = queries.toArray();
        List<float[][]> entries = new ArrayList<>(slots);
        entries.addAll(recent);
        boolean linear = (mode == Mode.LINEAR || mode == Mode.DELTA) && folded > 0;
        float[] out = new float[heads * steps * valueSize];
        float[] mem = linear ? readMemory(queries).toArray() : null;
        int n = entries.size();
        double[] w = new double[n];
        for (int h = 0; h < heads; h++) {
            for (int t = 0; t < steps; t++) {
                int qo = (h * steps + t) * keySize;
                int oo = (h * steps + t) * valueSize;
                double max = Double.NEGATIVE_INFINITY;
                for (int e = 0; e < n; e++) {
                    w[e] = scale * KnnIndex.dot(q, qo, entries.get(e)[h], 0, keySize);
                    max = Math.max(max, w[e]);
                }
                double sum = 0;
                for (int e = 0; e < n; e++) {
                    w[e] = Math.exp(w[e] - max);
                    sum += w[e];
                }
                for (int e = 0; e < n; e++) {
                    float[] val = entries.get(e)[heads + h];
                    float a = (float) (w[e] / sum);
                    for (int j = 0; j < valueSize; j++) {
                        out[oo + j] += a * val[j];
                    }
                }
                if (linear) {
                    for (int j = 0; j < valueSize; j++) {
                        out[oo + j] = beta * mem[oo + j] + (1 - beta) * out[oo + j];
                    }
                }
            }
        }
        return Tensor.of(Shape.of(heads, steps, valueSize), out);
    }

    /** Forgets everything. */
    public void reset() {
        recent.clear();
        pending.clear();
        slots.clear();
        folded = 0;
        allocate();
    }

    @Override
    public String toString() {
        return "CompressiveKvMemory(" + mode + ", window " + window + ", " + folded + " folded)";
    }
}
