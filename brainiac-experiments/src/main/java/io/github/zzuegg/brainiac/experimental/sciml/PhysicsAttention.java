package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Attention for neural operators on meshes or point clouds, with cost linear in the number of
 * points {@code N}. One pre-norm transformer block, {@code x + attn(norm x)} then
 * {@code x + mlp(norm x)}, over point features {@code [b, N, C]}, with one of two attentions:
 *
 * <ul>
 *   <li>{@link Mode#TRANSOLVER}, Transolver's physics-attention (Wu et al., ICML 2024): each
 *       point is softly assigned to {@code M << N} learned slices,
 *       {@code w = softmax(x W_s / tau)}; each slice is averaged into a token,
 *       {@code z_m = sum_i w_im x_i / sum_i w_im}; full attention runs among the {@code M}
 *       tokens; the result is scattered back, {@code x_i' = sum_m w_im z_m'}.
 *   <li>{@link Mode#LANO}, the linear attention neural operator (2025), which recasts
 *       Transolver as linear attention through {@code M} learned agent tokens {@code A}:
 *       agents gather {@code softmax(A K^T) V} from all points, then points read
 *       {@code softmax(Q A^T)} from the agents.
 * </ul>
 *
 * <p>Neither depends on a grid, so the same weights work on regular grids and on irregular,
 * randomly sampled points (pass the coordinates as features).
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2402.02366">arXiv:2402.02366</a>,
 * <a href="https://arxiv.org/abs/2510.16816">arXiv:2510.16816</a>.
 */
public final class PhysicsAttention implements Layer {

    /** The attention variant. */
    public enum Mode {
        /** Slice, attend among slice tokens, deslice. */
        TRANSOLVER,
        /** Linear attention through learned agent tokens. */
        LANO
    }

    private final Mode mode;
    private final int channels;
    private final int slices;
    private final float temperature;
    private final Var slice;
    private final Var agents;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var ff1;
    private final Var ff1b;
    private final Var ff2;

    private PhysicsAttention(Mode mode, int channels, int slices, float temperature, SplittableRandom random) {
        this.mode = mode;
        this.channels = channels;
        this.slices = slices;
        this.temperature = temperature;
        this.slice = weight(channels, slices, random);
        this.agents = Var.param(Tensor.normal(Shape.of(slices, channels), 0, 1f, random));
        this.wq = weight(channels, channels, random);
        this.wk = weight(channels, channels, random);
        this.wv = weight(channels, channels, random);
        this.wo = weight(channels, channels, random);
        this.ff1 = weight(channels, 2 * channels, random);
        this.ff1b = Var.param(Tensor.zeros(Shape.of(2 * channels)));
        this.ff2 = weight(2 * channels, channels, random);
    }

    private static Var weight(int in, int out, SplittableRandom random) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), random));
    }

    /**
     * Creates a block.
     *
     * @param mode the attention variant
     * @param channels the feature width {@code C}
     * @param slices the slices or agent tokens {@code M}
     * @param seed the seed
     * @return the block
     */
    public static PhysicsAttention of(Mode mode, int channels, int slices, long seed) {
        return new PhysicsAttention(mode, channels, slices, 0.5f, new SplittableRandom(seed));
    }

    /**
     * Describes a block for model builders.
     *
     * @param mode the attention variant
     * @param slices the slices or agent tokens
     * @return the spec
     */
    public static LayerSpec spec(Mode mode, int slices) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PhysicsAttention(mode, input.dim(-1), slices, 0.5f, random);
            }
        };
    }

    /**
     * The softmax over an axis.
     *
     * @param x scores
     * @param axis the axis
     * @return probabilities
     */
    static Var softmax(Var x, int axis) {
        return x.sub(x.logsumexp(axis, true)).exp();
    }

    /**
     * Zero-mean, unit-variance normalisation over the last axis (no affine part).
     *
     * @param x features
     * @return normalised features
     */
    static Var norm(Var x) {
        int last = x.shape().rank() - 1;
        Var c = x.sub(x.mean(last, true));
        return c.div(c.square().mean(last, true).add(1e-5f).sqrt());
    }

    /**
     * The attention part alone (no residual or MLP).
     *
     * @param x features {@code [b, N, C]}
     * @return {@code [b, N, C]}
     */
    public Var attention(Var x) {
        int b = x.shape().dim(0);
        float scale = (float) (1 / Math.sqrt(channels));
        if (mode == Mode.TRANSOLVER) {
            Var w = softmax(x.matmul(slice).div(temperature), 2);                // [b, N, M]
            Var wt = w.permute(0, 2, 1);                                          // [b, M, N]
            Var tokens = wt.matmul(x).div(w.sum(1, true).permute(0, 2, 1).add(1e-5f));   // [b, M, C]
            Var q = tokens.matmul(wq);
            Var k = tokens.matmul(wk);
            Var v = tokens.matmul(wv);
            Var att = softmax(q.matmul(k.permute(0, 2, 1)).mul(scale), 2).matmul(v);    // [b, M, C]
            return w.matmul(att).matmul(wo);
        }
        Var a = agents.reshape(1, slices, channels).expand(Shape.of(b, slices, channels));
        Var q = x.matmul(wq);
        Var k = x.matmul(wk);
        Var v = x.matmul(wv);
        Var gather = softmax(a.matmul(k.permute(0, 2, 1)).mul(scale), 2).matmul(v);    // [b, M, C]
        Var read = softmax(q.matmul(a.permute(0, 2, 1)).mul(scale), 2);                // [b, N, M]
        return read.matmul(gather).matmul(wo);
    }

    @Override
    public Var forward(Var x) {
        Var h = x.add(attention(norm(x)));
        Var f = Activation.GELU.forward(norm(h).matmul(ff1).add(ff1b)).matmul(ff2);
        return h.add(f);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo, ff1, ff1b, ff2));
        p.add(mode == Mode.TRANSOLVER ? slice : agents);
        return p;
    }

    @Override
    public String toString() {
        return "PhysicsAttention(" + mode + ", C=" + channels + ", M=" + slices + ")";
    }
}
