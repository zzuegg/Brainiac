package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * iRoPE, the interleaved position scheme of Llama 4: in a causal stack, every
 * {@code nopeEvery}-th layer drops the position encoding (NoPE) and attends globally, with
 * Scalable-Softmax-style temperature scaling of its logits; the other layers use RoPE,
 * optionally over a local sliding window. The RoPE layers carry precise relative order
 * where they are trained; the NoPE layers carry long-range retrieval that does not depend on
 * distances seen in training, and the {@code log n} temperature keeps their attention from
 * flattening as the context grows. Together they extrapolate further than an all-RoPE or
 * all-NoPE stack.
 *
 * <p>{@code nopeEvery = 0} gives an all-RoPE stack and {@code nopeEvery = 1} an all-NoPE
 * one, the two baselines. Every layer is a {@link ResidualMixerBlock} around a
 * {@link RefinedAttention}; input and output are {@code [batch, steps, features]}.
 *
 * <p>Source: Meta AI, "The Llama 4 herd" (2025); the NoPE/RoPE trade-off is studied in Yang
 * et al., "Rope to Nope and Back Again: A New Hybrid Attention Strategy",
 * <a href="https://arxiv.org/abs/2501.18795">arXiv:2501.18795</a>.
 */
public final class InterleavedRopeStack implements Layer {

    private final List<ResidualMixerBlock> blocks = new ArrayList<>();
    private final boolean[] nope;

    /**
     * Creates the stack.
     *
     * @param features the width
     * @param heads the attention heads
     * @param depth the number of layers
     * @param nopeEvery every how many layers one is NoPE (0 for none, 1 for all)
     * @param temperature whether NoPE layers scale their logits by a learned {@code s log n}
     * @param ropeWindow the sliding window of the RoPE layers, or 0 for full attention
     * @param random the random source
     */
    public InterleavedRopeStack(int features, int heads, int depth, int nopeEvery, boolean temperature,
            int ropeWindow, SplittableRandom random) {
        this.nope = new boolean[depth];
        for (int l = 0; l < depth; l++) {
            nope[l] = nopeEvery > 0 && (l + 1) % nopeEvery == 0;
            RefinedAttention.Spec attn = RefinedAttention.of(heads).causal(true);
            attn = nope[l] ? attn.scalableSoftmax(temperature) : attn.rotary(10000f).window(ropeWindow);
            blocks.add(ResidualMixerBlock.of(attn).create(io.github.zzuegg.brainiac.tensor.Shape.of(1, features),
                    random));
        }
    }

    /**
     * Describes the stack for {@code Model.sequential().add(...)}.
     *
     * @param heads the attention heads
     * @param depth the number of layers
     * @param nopeEvery every how many layers one is NoPE (0 for none, 1 for all)
     * @param temperature whether NoPE layers use the {@code log n} temperature
     * @param ropeWindow the RoPE layers' sliding window, or 0
     * @return the spec
     */
    public static LayerSpec of(int heads, int depth, int nopeEvery, boolean temperature, int ropeWindow) {
        return MixerOps.sameShape((in, r) -> new InterleavedRopeStack(in.dim(-1), heads, depth, nopeEvery,
                temperature, ropeWindow, r));
    }

    /** @return for each layer, whether it is NoPE */
    public boolean[] nopeLayers() {
        return nope.clone();
    }

    @Override
    public Var forward(Var input) {
        Var x = input;
        for (ResidualMixerBlock b : blocks) {
            x = b.forward(x);
        }
        return x;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        blocks.forEach(b -> p.addAll(b.parameters()));
        return p;
    }

    @Override
    public String toString() {
        return "InterleavedRopeStack(" + blocks.size() + " layers, nope=" + java.util.Arrays.toString(nope) + ")";
    }
}
