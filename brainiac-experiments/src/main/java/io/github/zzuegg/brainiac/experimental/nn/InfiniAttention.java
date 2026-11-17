package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Infini-attention (Munkhdalai et al. 2024, <a href="https://arxiv.org/abs/2404.07143">arXiv:2404.07143</a>):
 * causal attention over a long sequence processed in segments, where each segment attends
 * to itself exactly and reads everything before it from a compressive linear memory of
 * constant size. Per head, the memory is {@code M = sum phi(K)^T V} and
 * {@code z = sum phi(K)} over earlier segments ({@code phi = elu + 1}); a query reads
 * {@code phi(q) M / phi(q) z}, and a learned per-head gate mixes the read with local
 * attention. With {@link #delta(boolean) delta} the memory is written with the delta rule.
 *
 * <p>The whole sequence is one graph, so gradients flow back through the memory into earlier
 * segments (backpropagation through the memory). Memory cost per head is
 * {@code dk x (dv + 1)}, however long the sequence; attention cost is {@code O(T * segment)}.
 * Input {@code [batch, steps, features]}, output the same shape.
 */
public final class InfiniAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int segment;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var gate;
    private boolean delta;
    private boolean useMemory = true;

    /**
     * Creates the layer.
     *
     * @param features the input and output features
     * @param heads the heads
     * @param headSize the size per head
     * @param segment the steps per segment
     * @param random the random source for the weights
     */
    public InfiniAttention(int features, int heads, int headSize, int segment, SplittableRandom random) {
        if (features < 1 || heads < 1 || headSize < 1 || segment < 1) {
            throw new IllegalArgumentException("sizes must be positive");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        this.segment = segment;
        int hd = heads * headSize;
        wq = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wk = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wv = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wo = Var.param(Init.XAVIER.create(Shape.of(hd, features), hd, features, random));
        gate = Var.param(Tensor.zeros(Shape.of(1, heads, 1, 1)));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param heads the heads
     * @param headSize the size per head
     * @param segment the steps per segment
     * @return the spec
     */
    public static LayerSpec spec(int heads, int headSize, int segment) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new InfiniAttention(input.dim(-1), heads, headSize, segment, random);
            }
        };
    }

    /**
     * @param on write the memory with the delta rule instead of plain addition
     * @return this layer
     */
    public InfiniAttention delta(boolean on) {
        this.delta = on;
        return this;
    }

    /**
     * @param on whether segments read the memory; off, each segment sees only itself (for comparisons)
     * @return this layer
     */
    public InfiniAttention useMemory(boolean on) {
        this.useMemory = on;
        return this;
    }

    /**
     * {@code elu(x) + 1}, written as {@code relu(x) + exp(min(x, 0))}.
     *
     * @param x the input
     * @return positive features
     */
    static Var eluPlusOne(Var x) {
        return x.relu().add(x.minimum(0f).exp());
    }

    private Var split(Var x, Var w, int n, int t) {
        return x.matmul(w).reshape(n, t, heads, headSize).permute(0, 2, 1, 3);
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int steps = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got " + input.shape());
        }
        Var q = split(input, wq, n, steps);
        Var k = split(input, wk, n, steps);
        Var v = split(input, wv, n, steps);
        float scale = (float) (1 / Math.sqrt(headSize));
        Var memory = null;
        Var norm = null;
        Var g = gate.sigmoid();
        Var[] outs = new Var[(steps + segment - 1) / segment];
        for (int s = 0, i = 0; s < steps; s += segment, i++) {
            int e = Math.min(steps, s + segment);
            Var qs = q.slice(2, s, e);
            Var ks = k.slice(2, s, e);
            Var vs = v.slice(2, s, e);
            Var local = Functional.attention(qs, ks, vs, true, scale);
            Var out = local;
            if (useMemory && memory != null) {
                Var pq = eluPlusOne(qs);
                Var read = pq.matmul(memory).div(pq.matmul(norm).add(1e-6f));
                out = read.mul(g).add(local.mul(g.neg().add(1f)));
            }
            outs[i] = out;
            if (useMemory) {
                Var pk = eluPlusOne(ks);
                Var write = vs;
                if (delta && memory != null) {
                    write = vs.sub(pk.matmul(memory).div(pk.matmul(norm).add(1e-6f)));
                }
                Var dm = pk.transpose().matmul(write);
                Var dz = pk.sum(2, true).transpose();
                memory = memory == null ? dm : memory.add(dm);
                norm = norm == null ? dz : norm.add(dz);
            }
        }
        Var all = outs.length == 1 ? outs[0] : Var.concat(2, outs);
        return all.permute(0, 2, 1, 3).reshape(n, steps, heads * headSize).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo, gate);
    }

    @Override
    public String toString() {
        return "InfiniAttention(" + heads + " heads, segment " + segment + (delta ? ", delta" : "") + ")";
    }
}
