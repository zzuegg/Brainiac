package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The Perceiver encoder {@code [batch, m, inputs] -> [batch, latents, width]}: a small learned
 * latent array cross-attends to a large input array (one {@link LatentCrossAttention}), then
 * processes itself with {@code blocks} core {@link TransformerBlock}s in latent space. Attention
 * cost is {@code O(m n + blocks n^2)} for {@code n} latents instead of a transformer's
 * {@code O(blocks m^2)}, so inputs can be raw pixels, audio samples or long token sequences,
 * with modality-agnostic position features concatenated to them.
 *
 * <p>Perceiver IO adds a decoder: {@link #decode} lets arbitrary output queries (one per output
 * position, built from position features) cross-attend to the latents, so one model emits
 * outputs of any size and structure.
 *
 * <p>Papers: Jaegle et al., "Perceiver: General Perception with Iterative Attention" (ICML 2021),
 * <a href="https://arxiv.org/abs/2103.03206">arXiv:2103.03206</a>; Jaegle et al., "Perceiver IO"
 * (ICLR 2022), <a href="https://arxiv.org/abs/2107.14795">arXiv:2107.14795</a>.
 */
public final class PerceiverEncoder implements Layer {

    private final int inputs;
    private final int latents;
    private final int width;
    private final LatentCrossAttention cross;
    private final List<Layer> self = new ArrayList<>();
    private final LatentCrossAttention decoder;

    /**
     * Creates the encoder.
     *
     * @param inputs the features per input element (including position features)
     * @param latents the latent array length
     * @param width the latent width
     * @param heads the attention heads; must divide {@code width}
     * @param blocks the latent self-attention blocks
     * @param queryFeatures the width of Perceiver IO output queries, 0 for no decoder
     * @param random the source of initial weights
     */
    public PerceiverEncoder(int inputs, int latents, int width, int heads, int blocks, int queryFeatures,
            RandomGenerator random) {
        this.inputs = inputs;
        this.latents = latents;
        this.width = width;
        this.cross = new LatentCrossAttention(width, inputs, heads, latents, random);
        java.util.SplittableRandom r = new java.util.SplittableRandom(random.nextLong());
        for (int i = 0; i < blocks; i++) {
            self.add(TransformerBlock.of(heads).ffnSize(2 * width).create(Shape.of(latents, width), r));
        }
        this.decoder = queryFeatures > 0 ? new LatentCrossAttention(queryFeatures, width, queryHeads(queryFeatures,
                heads), 0, random) : null;
    }

    private static int queryHeads(int features, int heads) {
        int h = Math.min(heads, features);
        while (features % h != 0) {
            h--;
        }
        return h;
    }

    /**
     * Describes an encoder (no decoder) for model builders on {@code [m, inputs]}.
     *
     * @param latents the latent array length
     * @param width the latent width
     * @param heads the attention heads
     * @param blocks the latent self-attention blocks
     * @return the spec
     */
    public static LayerSpec spec(int latents, int width, int heads, int blocks) {
        return ClassicOps.spec(s -> Shape.of(latents, width),
                (s, r) -> new PerceiverEncoder(s.dim(1), latents, width, heads, blocks, 0, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, m, " + inputs + "], got " + input.shape());
        }
        Var z = cross.forward(List.of(input), training);
        for (Layer l : self) {
            z = l.forward(z, training);
        }
        return z;
    }

    /**
     * Perceiver IO decoding: output queries cross-attend to the latents.
     *
     * @param latentArray {@code [batch, latents, width]} from {@link #forward(Var, boolean)}
     * @param queries {@code [batch, outputs, queryFeatures]}
     * @param training whether training
     * @return {@code [batch, outputs, queryFeatures]}
     */
    public Var decode(Var latentArray, Var queries, boolean training) {
        if (decoder == null) {
            throw new IllegalStateException("this encoder was built without a decoder");
        }
        return decoder.attend(queries, latentArray, training);
    }

    /**
     * Multiply-adds per example of the encoder's attention and projections, for an input of
     * {@code m} elements: the cross-attention ({@code m} keys and values, {@code n} queries) plus
     * the latent blocks.
     *
     * @param m the input elements
     * @return the count
     */
    public long multiplyAdds(int m) {
        long n = latents;
        long d = width;
        long crossCost = 2L * m * inputs * d + n * d * d + 2L * n * m * d + n * d * d + 4L * n * d * d;
        long block = 4L * n * d * d + 2L * n * n * d + 4L * n * d * d;
        return crossCost + self.size() * block;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(latents, width);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(cross.parameters());
        for (Layer l : self) {
            p.addAll(l.parameters());
        }
        if (decoder != null) {
            p.addAll(decoder.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "PerceiverEncoder(" + inputs + " -> " + latents + " x " + width + ", " + self.size() + " blocks)";
    }
}
