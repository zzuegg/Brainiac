package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A fast weight programmer: a slow network writes a fast weight matrix per head, step by
 * step, and reads it with a query (Schmidhuber 1992; Schlag, Irie and Schmidhuber 2021,
 * "Linear Transformers Are Secretly Fast Weight Programmers",
 * <a href="https://arxiv.org/abs/2102.11174">arXiv:2102.11174</a>). With
 * {@code k, v, q, beta} projected from the input and {@code phi = elu + 1} (sum-normalized):
 * <pre>
 * SUM:   W_t = W_(t-1) + v_t phi(k_t)^T                          (= causal linear attention)
 * DELTA: W_t = W_(t-1) + beta_t (v_t - W_(t-1) phi(k_t)) phi(k_t)^T
 * y_t = W_t phi(q_t)
 * </pre>
 * The {@link Rule#SUM sum rule} is exactly an unnormalized linear transformer, which only
 * ever adds associations; the {@link Rule#DELTA delta rule} corrects the value stored under a
 * key, so it can overwrite. Memory is {@code dk x dv} per head however long the input.
 * Input {@code [batch, steps, features]}, output the same shape.
 */
public final class FastWeightProgrammer implements Layer {

    /** How the fast weights are written. */
    public enum Rule {
        /** Add {@code v phi(k)^T}: a linear transformer. */
        SUM,
        /** Write {@code beta (v - W phi(k)) phi(k)^T}: the delta rule. */
        DELTA
    }

    private final int features;
    private final int heads;
    private final int headSize;
    private final Rule rule;
    private final Var wk;
    private final Var wv;
    private final Var wq;
    private final Var wb;
    private final Var wo;
    private boolean normalizeKeys = true;

    /**
     * Creates the layer.
     *
     * @param features the input and output features
     * @param heads the heads
     * @param headSize the key and value size per head
     * @param rule how fast weights are written
     * @param random the random source for the slow weights
     */
    public FastWeightProgrammer(int features, int heads, int headSize, Rule rule, SplittableRandom random) {
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        this.rule = rule;
        int hd = heads * headSize;
        wk = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wv = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wq = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wb = Var.param(Init.XAVIER.create(Shape.of(features, heads), features, heads, random));
        wo = Var.param(Init.XAVIER.create(Shape.of(hd, features), hd, features, random));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param heads the heads
     * @param headSize the key and value size per head
     * @param rule how fast weights are written
     * @return the spec
     */
    public static LayerSpec spec(int heads, int headSize, Rule rule) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new FastWeightProgrammer(input.dim(-1), heads, headSize, rule, random);
            }
        };
    }

    /**
     * @param on whether {@code phi(k)} and {@code phi(q)} are divided by their sum (on by
     *     default, as in the paper; off gives plain linear attention for the sum rule)
     * @return this layer
     */
    public FastWeightProgrammer normalizeKeys(boolean on) {
        this.normalizeKeys = on;
        return this;
    }

    /** @return the projections k, v, q, beta and output, in that order */
    @Override
    public List<Var> parameters() {
        return List.of(wk, wv, wq, wb, wo);
    }

    private Var heads(Var x, Var w, int n, int t) {
        return x.matmul(w).reshape(n, t, heads, headSize).permute(0, 2, 1, 3);
    }

    private Var phi(Var x) {
        Var p = InfiniAttention.eluPlusOne(x);
        return normalizeKeys ? p.div(p.sum(-1, true)) : p;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int t = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got " + input.shape());
        }
        Var k = phi(heads(input, wk, n, t));
        Var v = heads(input, wv, n, t);
        Var q = phi(heads(input, wq, n, t));
        Var beta = input.matmul(wb).sigmoid().permute(0, 2, 1).reshape(n, heads, t, 1);
        // fast weights W [n, heads, dv, dk], stored transposed as F = W^T [n, heads, dk, dv]
        Var fast = Var.of(Tensor.zeros(Shape.of(n, heads, headSize, headSize)));
        Var[] outs = new Var[t];
        for (int s = 0; s < t; s++) {
            Var ks = k.slice(2, s, s + 1);
            Var vs = v.slice(2, s, s + 1);
            Var write = vs;
            if (rule == Rule.DELTA) {
                write = vs.sub(ks.matmul(fast)).mul(beta.slice(2, s, s + 1));
            }
            fast = fast.add(ks.transpose().matmul(write));
            outs[s] = q.slice(2, s, s + 1).matmul(fast);
        }
        Var y = Var.concat(2, outs);
        return y.permute(0, 2, 1, 3).reshape(n, t, heads * headSize).matmul(wo);
    }

    @Override
    public String toString() {
        return "FastWeightProgrammer(" + heads + " heads, " + rule + ")";
    }
}
