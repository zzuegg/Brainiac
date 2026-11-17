package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Causal self-attention that also reads an external, growing episodic memory of earlier
 * keys and values (the Memorizing Transformer, Wu et al. 2022,
 * <a href="https://arxiv.org/abs/2203.08913">arXiv:2203.08913</a>). Each query looks up its
 * {@code topK} nearest stored keys per head with an exact kNN search, attends over them
 * separately, and a learned per-head gate mixes the memory result with ordinary local
 * attention: {@code out = g * memory + (1 - g) * local}, {@code g = sigmoid(gate)}.
 *
 * <p>After each forward pass with {@link #memorize(boolean) memorize} on, the segment's keys
 * and values are appended to the memory (detached: no gradient flows into the past, as in
 * the paper). The memory outlives a {@link io.github.zzuegg.brainiac.nn.KvCache}'s window,
 * so a model can recall a fact read thousands of tokens ago; {@link #saveMemory} and
 * {@link #loadMemory} keep it across sessions. All batch rows share one memory, so feed one
 * document at a time.
 *
 * <p>Built on {@link MultiHeadAttention}: {@link #wrap} reuses an existing layer's
 * projections (training this layer trains them), or {@link #create} makes fresh ones.
 */
public final class KnnMemoryAttention implements Layer {

    private final int heads;
    private final int headSize;
    private final int valueSize;
    private final int features;
    private final int outputSize;
    private final boolean causal;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var bq;
    private final Var bk;
    private final Var bv;
    private final Var bo;
    private final Var gate;
    private final KnnIndex[] memory;
    private int topK;
    private int capacity = Integer.MAX_VALUE;
    private boolean memorize = true;

    private KnnMemoryAttention(int features, int heads, int headSize, int valueSize, int outputSize, boolean causal,
            List<Var> projections, int topK) {
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        this.valueSize = valueSize;
        this.outputSize = outputSize;
        this.causal = causal;
        this.wq = projections.get(0);
        this.wk = projections.get(1);
        this.wv = projections.get(2);
        this.wo = projections.get(3);
        boolean bias = projections.size() > 4;
        this.bq = bias ? projections.get(4) : null;
        this.bk = bias ? projections.get(5) : null;
        this.bv = bias ? projections.get(6) : null;
        this.bo = bias ? projections.get(7) : null;
        this.gate = Var.param(Tensor.zeros(Shape.of(1, heads, 1, 1)));
        this.memory = new KnnIndex[heads];
        for (int h = 0; h < heads; h++) {
            memory[h] = KnnIndex.create(headSize, valueSize, KnnIndex.Metric.DOT);
        }
        topK(topK);
    }

    /**
     * Adds a kNN memory to an existing self-attention layer, sharing its weights. The layer
     * must have equal query and key/value inputs, one key/value head per query head and no
     * ALiBi or key mask; a rotary setting is ignored (memory keys carry no positions).
     *
     * @param attention the layer whose projections to use
     * @param topK the memories each query reads per head
     * @return the memory layer
     */
    public static KnnMemoryAttention wrap(MultiHeadAttention attention, int topK) {
        MultiHeadAttention.Settings s = attention.settings();
        if (s.queryFeatures() != s.kvFeatures() || s.kvHeads() != s.heads() || s.alibi() || s.keyMask()) {
            throw new IllegalArgumentException(attention + " is not plain multi-head self-attention");
        }
        return new KnnMemoryAttention(s.queryFeatures(), s.heads(), s.headSize(), s.valueSize(), s.outputSize(),
                s.causal(), attention.parameters(), topK);
    }

    /**
     * Creates the layer with fresh projections.
     *
     * @param features the input and output features
     * @param heads the heads
     * @param headSize the key, query and value size per head
     * @param topK the memories each query reads per head
     * @param random the random source for the weights
     * @return the layer
     */
    public static KnnMemoryAttention create(int features, int heads, int headSize, int topK, SplittableRandom random) {
        MultiHeadAttention mha = new MultiHeadAttention(new MultiHeadAttention.Settings(features, features, heads, heads,
                headSize, headSize, features, false, true, 0, false, 0, 0, false), Init.XAVIER, random);
        return wrap(mha, topK);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, on inputs {@code [steps, features]}.
     *
     * @param heads the heads
     * @param headSize the key, query and value size per head
     * @param topK the memories each query reads per head
     * @return the spec
     */
    public static io.github.zzuegg.brainiac.nn.LayerSpec spec(int heads, int headSize, int topK) {
        return new io.github.zzuegg.brainiac.nn.LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return KnnMemoryAttention.create(input.dim(-1), heads, headSize, topK, random);
            }
        };
    }

    /**
     * @param k the memories each query reads per head
     * @return this layer
     */
    public KnnMemoryAttention topK(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("topK must be positive");
        }
        this.topK = k;
        return this;
    }

    /**
     * @param on whether forward passes append their keys and values to the memory
     * @return this layer
     */
    public KnnMemoryAttention memorize(boolean on) {
        this.memorize = on;
        return this;
    }

    /**
     * @param rows the most memories kept per head; the least recently used go first
     * @return this layer
     */
    public KnnMemoryAttention capacity(int rows) {
        if (rows < 1) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        this.capacity = rows;
        return this;
    }

    /** @return the memories held per head */
    public int memorySize() {
        return memory[0].size();
    }

    /** Forgets every memory, to start a new document. */
    public void clearMemory() {
        for (KnnIndex m : memory) {
            m.clear();
        }
    }

    /** @return the gate value per head, the share of the output taken from memory */
    public float[] gates() {
        return gate.value().sigmoid().toArray();
    }

    private Var project(Var x, Var w, Var b, int size) {
        int n = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var y = x.matmul(w);
        if (b != null) {
            y = y.add(b);
        }
        return y.reshape(n, t, heads, size).permute(0, 2, 1, 3);
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 3 || input.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got " + input.shape());
        }
        int n = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = project(input, wq, bq, headSize);
        Var k = project(input, wk, bk, headSize);
        Var v = project(input, wv, bv, valueSize);
        float scale = (float) (1 / Math.sqrt(headSize));
        Var local = Functional.attention(q, k, v, causal, scale);
        Var out = local;
        if (memory[0].size() > 0) {
            Var remote = readMemory(q, n, t, scale);
            Var g = gate.sigmoid();
            out = remote.mul(g).add(local.mul(g.neg().add(1f)));
        }
        if (memorize) {
            write(k.value(), v.value(), n, t);
        }
        Var y = out.permute(0, 2, 1, 3).reshape(n, t, heads * valueSize).matmul(wo);
        return bo != null ? y.add(bo) : y;
    }

    /** Attention over each query's own top-k memories: {@code [n, heads, t, dv]}. */
    private Var readMemory(Var q, int n, int t, float scale) {
        int kk = Math.min(topK, memory[0].size());
        float[] qs = q.value().toArray();
        float[] keys = new float[n * heads * t * kk * headSize];
        float[] vals = new float[n * heads * t * kk * valueSize];
        float[] query = new float[headSize];
        for (int b = 0; b < n; b++) {
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < t; i++) {
                    int row = (b * heads + h) * t + i;
                    System.arraycopy(qs, row * headSize, query, 0, headSize);
                    KnnIndex.Hit[] hits = memory[h].search(query, kk);
                    for (int j = 0; j < kk; j++) {
                        memory[h].touch(hits[j].row());
                        System.arraycopy(memory[h].key(hits[j].row()), 0, keys, (row * kk + j) * headSize, headSize);
                        System.arraycopy(memory[h].value(hits[j].row()), 0, vals, (row * kk + j) * valueSize,
                                valueSize);
                    }
                }
            }
        }
        var device = q.value().device();
        Tensor mk = Tensor.of(Shape.of(n, heads, t, headSize, kk), transposeLast(keys, n * heads * t, kk, headSize),
                device);
        Tensor mv = Tensor.of(Shape.of(n, heads, t, kk, valueSize), vals, device);
        Var scores = q.reshape(n, heads, t, 1, headSize).matmul(Var.of(mk)).mul(scale);
        Var w = Activation.SOFTMAX.forward(scores);
        return w.matmul(Var.of(mv)).reshape(n, heads, t, valueSize);
    }

    private static float[] transposeLast(float[] x, int blocks, int rows, int cols) {
        float[] y = new float[x.length];
        for (int b = 0; b < blocks; b++) {
            int o = b * rows * cols;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    y[o + c * rows + r] = x[o + r * cols + c];
                }
            }
        }
        return y;
    }

    private void write(Tensor keys, Tensor values, int n, int t) {
        float[] ks = keys.toArray();
        float[] vs = values.toArray();
        float[] key = new float[headSize];
        float[] val = new float[valueSize];
        for (int b = 0; b < n; b++) {
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < t; i++) {
                    int row = (b * heads + h) * t + i;
                    System.arraycopy(ks, row * headSize, key, 0, headSize);
                    System.arraycopy(vs, row * valueSize, val, 0, valueSize);
                    memory[h].add(key, val);
                }
                memory[h].pruneLeastRecentlyUsed(capacity);
            }
        }
    }

    /**
     * Saves the memory, so a later session can {@link #loadMemory load} it.
     *
     * @param path the file
     * @throws IOException if writing fails
     */
    public void saveMemory(Path path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path))) {
            DataOutputStream d = new DataOutputStream(out);
            d.writeInt(heads);
            d.flush();
            for (KnnIndex m : memory) {
                m.write(out);
            }
        }
    }

    /**
     * Replaces the memory with one {@link #saveMemory} wrote.
     *
     * @param path the file
     * @throws IOException if reading fails or the memory does not fit this layer
     */
    public void loadMemory(Path path) throws IOException {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path))) {
            int h = new DataInputStream(in).readInt();
            if (h != heads) {
                throw new IOException("memory has " + h + " heads, the layer " + heads);
            }
            for (int i = 0; i < heads; i++) {
                KnnIndex m = KnnIndex.read(in);
                if (m.keySize() != headSize || m.valueSize() != valueSize) {
                    throw new IOException("memory sizes do not fit the layer");
                }
                memory[i] = m;
            }
        }
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo));
        if (bq != null) {
            p.addAll(List.of(bq, bk, bv, bo));
        }
        p.add(gate);
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputSize;
    }

    @Override
    public String toString() {
        return "KnnMemoryAttention(" + heads + " heads, top " + topK + ", " + memorySize() + " memories)";
    }
}
