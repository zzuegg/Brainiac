package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Attention-based deep multiple-instance learning (Ilse, Tomczak, Welling, ICML 2018,
 * <a href="https://arxiv.org/abs/1802.04712">arXiv:1802.04712</a>). Only bags of instances are
 * labelled, a bag being positive when at least one instance is. Every instance is embedded,
 * {@code h_k = relu(W x_k + b)}, the bag pooled, and the pooled vector classified. Pooling is
 * {@link Pooling#MEAN}, {@link Pooling#MAX}, {@link Pooling#ATTENTION}
 * ({@code a_k ~ exp(w^T tanh(V h_k))}) or {@link Pooling#GATED}
 * ({@code a_k ~ exp(w^T (tanh(V h_k) * sigmoid(U h_k)))}); the attention weights localise the
 * key instances without instance labels ({@link #attention()}).
 *
 * <p>A core {@link Layer}: input {@code [bags, instances, features]} (bags of equal size in one
 * batch, or one bag at a time), output {@code [bags, outputs]} logits, the same contract as
 * {@code experimental.nn.SetEncoder}, so the two swap in one line.
 */
public final class AttentionMil implements Layer {

    /** The bag pooling operator. */
    public enum Pooling {
        /** Mean of the instance embeddings. */
        MEAN,
        /** Element-wise max of the instance embeddings. */
        MAX,
        /** Softmax attention. */
        ATTENTION,
        /** Gated softmax attention. */
        GATED
    }

    private final int inputs;
    private final int embed;
    private final int outputs;
    private final Pooling pooling;
    private final Var w1;
    private final Var b1;
    private final Var v;
    private final Var u;
    private final Var wa;
    private final Var w2;
    private final Var b2;
    private Tensor lastAttention;

    /**
     * Creates the model.
     *
     * @param inputs features per instance
     * @param embed the instance embedding width
     * @param attention the attention hidden width
     * @param outputs output logits (1 for a binary bag label)
     * @param pooling the pooling operator
     * @param random the source of initial weights
     */
    public AttentionMil(int inputs, int embed, int attention, int outputs, Pooling pooling, RandomGenerator random) {
        this.inputs = inputs;
        this.embed = embed;
        this.outputs = outputs;
        this.pooling = pooling;
        this.w1 = matrix(random, inputs, embed);
        this.b1 = Var.param(Tensor.zeros(Shape.of(embed)));
        this.v = matrix(random, embed, attention);
        this.u = matrix(random, embed, attention);
        this.wa = matrix(random, attention, 1);
        this.w2 = matrix(random, embed, outputs);
        this.b2 = Var.param(Tensor.zeros(Shape.of(outputs)));
    }

    /**
     * A spec for model builders.
     *
     * @param embed the embedding width
     * @param attention the attention width
     * @param outputs output logits
     * @param pooling the pooling operator
     * @return the spec, for inputs {@code [bags, instances, features]}
     */
    public static LayerSpec spec(int embed, int attention, int outputs, Pooling pooling) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new AttentionMil(input.dim(-1), embed, attention, outputs, pooling, random);
            }
        };
    }

    private static Var matrix(RandomGenerator random, int in, int out) {
        float[] w = new float[in * out];
        double scale = Math.sqrt(2.0 / in);
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (random.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(Shape.of(in, out), w));
    }

    @Override
    public Var forward(Var input) {
        int bags = input.shape().dim(0);
        int k = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [bags, instances, " + inputs + "], got " + input.shape());
        }
        Var h = input.reshape(bags * k, inputs).matmul(w1).add(b1).relu();
        Var pooled;
        switch (pooling) {
            case MEAN -> {
                pooled = h.reshape(bags, k, embed).mean(1, false);
                lastAttention = Tensor.full(Shape.of(bags, k), 1f / k);
            }
            case MAX -> {
                pooled = h.reshape(bags, k, embed).max(1, false);
                lastAttention = null;
            }
            default -> {
                Var gate = h.matmul(v).tanh();
                if (pooling == Pooling.GATED) {
                    gate = gate.mul(h.matmul(u).sigmoid());
                }
                Var score = gate.matmul(wa).reshape(bags, k);
                Var a = score.sub(score.logsumexp(1, true)).exp();
                lastAttention = a.value();
                pooled = a.reshape(bags, k, 1).mul(h.reshape(bags, k, embed)).sum(1, false);
            }
        }
        return pooled.matmul(w2).add(b2);
    }

    /** @return the attention weights of the last forward pass, {@code [bags, instances]} (null for MAX) */
    public Tensor attention() {
        return lastAttention;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), outputs);
    }

    @Override
    public List<Var> parameters() {
        return switch (pooling) {
            case MEAN, MAX -> List.of(w1, b1, w2, b2);
            case ATTENTION -> List.of(w1, b1, v, wa, w2, b2);
            case GATED -> List.of(w1, b1, v, u, wa, w2, b2);
        };
    }

    @Override
    public String toString() {
        return "AttentionMil(" + inputs + " -> " + embed + ", " + pooling + " -> " + outputs + ")";
    }
}
