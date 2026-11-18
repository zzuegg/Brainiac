package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The Conformer block, the standard speech encoder layer: multi-head self-attention and a
 * convolution module sandwiched between two half-step feed-forward modules (macaron
 * style), each with a pre-norm residual, then a final layer norm:
 * <pre>
 * x = x + FF(x) / 2;  x = x + MHSA(LN(x));  x = x + Conv(LN(x));  x = x + FF(x) / 2;  y = LN(x)
 * </pre>
 * The convolution module is a pointwise expansion with a GLU, a depthwise convolution
 * along time ({@code kernel} taps, "same" padding), a layer norm (in place of the paper's
 * batch norm), SiLU and a pointwise projection: attention models global context, the
 * convolution local acoustic patterns. Self-attention is the core
 * {@link MultiHeadAttention}. Input and output are {@code [batch, frames, width]}.
 *
 * <p>Paper: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech
 * Recognition" (Interspeech 2020), <a href="https://arxiv.org/abs/2005.08100">arXiv:2005.08100</a>.
 */
public final class ConformerBlock implements Layer {

    private final int width;
    private final int kernel;
    private final Layer attention;
    private final LayerNorm[] norms = new LayerNorm[6];   // ff1, mhsa, conv, conv-inner, ff2, final
    private final Var[] ff1;
    private final Var[] ff2;
    private final Var pointwiseIn;
    private final Var depthwise;
    private final Var pointwiseOut;

    /**
     * Creates the block.
     *
     * @param width the model width
     * @param heads attention heads
     * @param kernel depthwise convolution taps (odd), such as 7
     * @param frames a typical sequence length (for the attention layer's shape)
     * @param random the random source
     */
    public ConformerBlock(int width, int heads, int kernel, int frames, SplittableRandom random) {
        this.width = width;
        this.kernel = kernel;
        this.attention = MultiHeadAttention.of(heads).create(Shape.of(frames, width), random);
        for (int i = 0; i < norms.length; i++) {
            norms[i] = LayerNorm.of(width);
        }
        ff1 = new Var[] {MixerOps.weight(random, width, 2 * width), MixerOps.weight(random, 2 * width, width)};
        ff2 = new Var[] {MixerOps.weight(random, width, 2 * width), MixerOps.weight(random, 2 * width, width)};
        pointwiseIn = MixerOps.weight(random, width, 2 * width);
        depthwise = MixerOps.uniform(random, (float) (1 / Math.sqrt(kernel)), kernel, width);
        pointwiseOut = MixerOps.weight(random, width, width);
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)}.
     *
     * @param heads attention heads
     * @param kernel depthwise convolution taps
     * @return the spec
     */
    public static LayerSpec of(int heads, int kernel) {
        return MixerOps.sameShape((in, r) -> new ConformerBlock(in.dim(-1), heads, kernel, in.dim(-2), r));
    }

    private static Var feedForward(Var x, Var[] w) {
        return x.matmul(w[0]).activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0).matmul(w[1]);
    }

    /** Depthwise convolution along time with "same" zero padding: {@code [b, t, c]}. */
    private Var depthwiseConv(Var x) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int half = kernel / 2;
        Var zeros = Var.of(Tensor.zeros(Shape.of(b, half, width)));
        Var padded = Var.concat(1, zeros, x, zeros);
        Var out = null;
        for (int k = 0; k < kernel; k++) {
            Var term = padded.slice(1, k, k + t).mul(depthwise.slice(0, k, k + 1));
            out = out == null ? term : out.add(term);
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var x = input.add(feedForward(norms[0].forward(input), ff1).mul(0.5f));
        x = x.add(attention.forward(norms[1].forward(x), training));
        Var c = norms[2].forward(x).matmul(pointwiseIn);
        Var glu = c.slice(2, 0, width).mul(c.slice(2, width, 2 * width).sigmoid());
        Var conv = norms[3].forward(depthwiseConv(glu))
                .activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0).matmul(pointwiseOut);
        x = x.add(conv);
        x = x.add(feedForward(norms[4].forward(x), ff2).mul(0.5f));
        return norms[5].forward(x);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(attention.parameters());
        for (LayerNorm n : norms) {
            p.addAll(n.parameters());
        }
        p.addAll(List.of(ff1));
        p.addAll(List.of(ff2));
        p.addAll(List.of(pointwiseIn, depthwise, pointwiseOut));
        return p;
    }

    @Override
    public String toString() {
        return "ConformerBlock(" + width + ", kernel " + kernel + ")";
    }
}
