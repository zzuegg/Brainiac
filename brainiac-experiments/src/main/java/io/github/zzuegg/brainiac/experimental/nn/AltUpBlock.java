package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Alternating updates (AltUp), as used in Gemma 3n: the residual stream is {@code K} times
 * wider than the transformer blocks ({@code K} slices of width {@code d}), but each block
 * runs on one slice only, so the representation grows without growing the compute. Per
 * block, with the active slice {@code j = layer mod K}:
 *
 * <ol>
 *   <li><b>predict</b>: every slice is re-estimated as a learned mix of all slices,
 *       {@code x^_i = sum_m P_im x_m} ({@code P} starts at the identity);</li>
 *   <li><b>compute</b>: the block runs on the active slice, {@code y~ = block(x^_j)};</li>
 *   <li><b>correct</b>: every slice moves by a learned share of the block's update,
 *       {@code y_i = x^_i + g_i (y~ - x^_j)}.</li>
 * </ol>
 *
 * <p>Input and output are {@code [batch, steps, K d]}; the wrapped block maps
 * {@code [batch, steps, d]} to the same (with its own residual).
 *
 * <p>Paper: Baykal et al., "Alternating Updates for Efficient Transformers",
 * <a href="https://arxiv.org/abs/2301.13310">arXiv:2301.13310</a>; Gemma 3n
 * (<a href="https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/">developer guide</a>).
 */
public final class AltUpBlock implements Layer {

    private final Layer block;
    private final int slices;
    private final int width;
    private final int active;
    private final Var predict;
    private final Var correct;

    /**
     * Wraps a block.
     *
     * @param block the block on one slice, {@code [batch, steps, width]}
     * @param slices the number of slices {@code K}
     * @param width the slice (block) width {@code d}
     * @param active the slice this block computes on
     * @param random the random source
     */
    public AltUpBlock(Layer block, int slices, int width, int active, RandomGenerator random) {
        if (slices < 1 || active < 0 || active >= slices) {
            throw new IllegalArgumentException("need slices >= 1 and 0 <= active < slices");
        }
        this.block = block;
        this.slices = slices;
        this.width = width;
        this.active = active;
        this.predict = TalkingHeadsAttention.identityMixing(slices, 0.01f, random);
        this.correct = MixerOps.filled(1f, slices, 1);
    }

    /**
     * Describes one block for {@code Model.sequential().add(...)}: the input width is
     * {@code K d}, the wrapped block is built for width {@code d}.
     *
     * @param block the block spec
     * @param slices {@code K}
     * @param active the active slice
     * @return the spec
     */
    public static LayerSpec of(LayerSpec block, int slices, int active) {
        return MixerOps.sameShape((in, r) -> {
            int d = in.dim(-1) / slices;
            return new AltUpBlock(block.create(Shape.of(in.dim(0), d), r), slices, d, active, r);
        });
    }

    /**
     * {@code depth} AltUp blocks, the active slice cycling {@code 0, 1, ..., K-1, 0, ...}.
     *
     * @param block the block spec
     * @param slices {@code K}
     * @param depth the number of blocks
     * @return the spec
     */
    public static LayerSpec stack(LayerSpec block, int slices, int depth) {
        LayerSpec[] specs = new LayerSpec[depth];
        for (int l = 0; l < depth; l++) {
            specs[l] = of(block, slices, l % slices);
        }
        return LayerChain.of(specs);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        int b = s.dim(0);
        int t = s.dim(1);
        if (s.rank() != 3 || s.dim(2) != slices * width) {
            throw new IllegalArgumentException("AltUpBlock expects [batch, steps, " + slices * width + "], got " + s);
        }
        Var x = input.reshape(b, t, slices, width);
        Var predicted = x.transpose().matmul(predict).transpose();               // [b, t, K, d]: sum_m P_mi x_m
        Var activeSlice = predicted.slice(2, active, active + 1).reshape(b, t, width);
        Var update = block.forward(activeSlice, training).sub(activeSlice).reshape(b, t, 1, width);
        return predicted.add(update.mul(correct)).reshape(b, t, slices * width);
    }

    /**
     * Splits the computed slice off the stream, for a readout that only uses it.
     *
     * @param stream {@code [batch, steps, K d]}
     * @param slices {@code K}
     * @param slice which slice
     * @return {@code [batch, steps, d]}
     */
    public static Var slice(Var stream, int slices, int slice) {
        Shape s = stream.shape();
        int d = s.dim(2) / slices;
        return stream.reshape(s.dim(0), s.dim(1), slices, d).slice(2, slice, slice + 1).reshape(s.dim(0), s.dim(1), d);
    }

    /** @return the wrapped block */
    public Layer block() {
        return block;
    }

    /** @return the predict matrix {@code [K, K]} */
    public Tensor predictMatrix() {
        return predict.value();
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new java.util.ArrayList<>(block.parameters());
        p.add(predict);
        p.add(correct);
        return p;
    }

    @Override
    public Var auxiliaryLoss() {
        return block.auxiliaryLoss();
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "AltUpBlock(K=" + slices + ", active=" + active + ", " + block + ")";
    }
}
