package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The Flamingo bridge (Alayrac et al., NeurIPS 2022): a perceiver resampler compresses a
 * frozen vision encoder's patch features into a fixed number of visual tokens, and new
 * gated cross-attention layers inserted between the blocks of a frozen language model let
 * the text attend to them:
 *
 * <pre>
 * x = x + tanh(a_attn) * CrossAttention(LN(x), visual)
 * x = x + tanh(a_ffn)  * FFN(LN(x))
 * </pre>
 *
 * <p>with both gates initialised at zero, so at the start the language model computes
 * exactly what it did before, and the visual pathway is blended in as training opens the
 * gates. The resampler is a stack of core {@link TransformerBlock}s with cross-attention:
 * learned latent queries attend among themselves and to the (projected) patch features.
 * The cross-attention layers are core {@link MultiHeadAttention}s; {@link #hook} plugs them
 * into a {@link FusionDecoder} after chosen blocks.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2204.14198">arXiv:2204.14198</a>.
 */
public final class GatedCrossAttentionAdapter {

    private final int dim;
    private final Var inProject;
    private final Var latents;
    private final List<MultiInputLayer> resampler = new ArrayList<>();
    private final boolean[] after;
    private final List<Gated> layers = new ArrayList<>();

    private static final class Gated {
        final LayerNorm attnNorm;
        final MultiInputLayer attention;
        final Var attnGate;
        final LayerNorm ffnNorm;
        final Var up;
        final Var down;
        final Var ffnGate;

        Gated(int dim, int heads, SplittableRandom r) {
            attnNorm = LayerNorm.of(dim);
            attention = MultiHeadAttention.of(heads).create(List.of(Shape.of(8, dim), Shape.of(8, dim)), r);
            attnGate = Var.param(Tensor.zeros(Shape.of(1)));
            ffnNorm = LayerNorm.of(dim);
            up = Var.param(Tensor.normal(Shape.of(dim, 2 * dim), 0, (float) Math.sqrt(2.0 / dim), r));
            down = Var.param(Tensor.normal(Shape.of(2 * dim, dim), 0, (float) Math.sqrt(1.0 / (2 * dim)), r));
            ffnGate = Var.param(Tensor.zeros(Shape.of(1)));
        }

        Var apply(Var x, Var visual) {
            Var a = attention.forward(List.of(attnNorm.forward(x), visual), false);
            x = x.add(a.mul(attnGate.tanh()));
            Var f = Activation.GELU.forward(ffnNorm.forward(x).matmul(up)).matmul(down);
            return x.add(f.mul(ffnGate.tanh()));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(attnNorm.parameters());
            p.addAll(attention.parameters());
            p.add(attnGate);
            p.addAll(ffnNorm.parameters());
            p.addAll(List.of(up, down, ffnGate));
            return p;
        }
    }

    /**
     * Creates the adapter.
     *
     * @param visionDim the patch feature width
     * @param dim the language model width
     * @param latents the resampler's visual tokens
     * @param resamplerLayers resampler blocks
     * @param lmLayers the language model's blocks
     * @param every insert a gated layer after every {@code every}-th block (1 = all)
     * @param heads attention heads
     * @param seed the seed
     */
    public GatedCrossAttentionAdapter(int visionDim, int dim, int latents, int resamplerLayers, int lmLayers,
            int every, int heads, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.dim = dim;
        this.inProject = Var.param(Tensor.normal(Shape.of(visionDim, dim), 0, (float) Math.sqrt(1.0 / visionDim), r));
        this.latents = Var.param(Tensor.normal(Shape.of(1, latents, dim), 0, 0.3f, r));
        for (int i = 0; i < resamplerLayers; i++) {
            resampler.add(TransformerBlock.of(heads).ffnSize(2 * dim).crossAttention(true)
                    .create(List.of(Shape.of(8, dim), Shape.of(8, dim)), r));
        }
        this.after = new boolean[lmLayers];
        for (int i = 0; i < lmLayers; i++) {
            after[i] = (i + 1) % every == 0;
            layers.add(after[i] ? new Gated(dim, heads, r) : null);
        }
    }

    /**
     * The perceiver resampler: a fixed number of visual tokens from any number of patches.
     *
     * @param patches {@code [batch, patches, visionDim]}
     * @return {@code [batch, latents, dim]}
     */
    public Var resample(Var patches) {
        int b = patches.shape().dim(0);
        Var memory = patches.matmul(inProject);
        Var x = latents.expand(Shape.of(b, latents.shape().dim(1), dim));
        for (MultiInputLayer l : resampler) {
            x = l.forward(List.of(x, memory), false);
        }
        return x;
    }

    /**
     * The hook that runs the gated cross-attention layers inside a frozen decoder.
     *
     * @param visual the resampled visual tokens {@code [batch, latents, dim]}
     * @return the hook
     */
    public FusionDecoder.BlockHook hook(Var visual) {
        return (block, x) -> block < after.length && after[block] ? layers.get(block).apply(x, visual) : x;
    }

    /** @return the current gate values {@code tanh(a)}, attention then feed-forward, per inserted layer */
    public double[] gates() {
        List<Double> g = new ArrayList<>();
        for (Gated l : layers) {
            if (l != null) {
                g.add(Math.tanh(l.attnGate.value().item()));
                g.add(Math.tanh(l.ffnGate.value().item()));
            }
        }
        return g.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /** @return the trainable values (projection, latents, resampler, gated layers) */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inProject, latents));
        for (MultiInputLayer l : resampler) {
            p.addAll(l.parameters());
        }
        for (Gated l : layers) {
            if (l != null) {
                p.addAll(l.parameters());
            }
        }
        return p;
    }

    @Override
    public String toString() {
        return "GatedCrossAttentionAdapter(" + latents.shape().dim(1) + " latents)";
    }
}
