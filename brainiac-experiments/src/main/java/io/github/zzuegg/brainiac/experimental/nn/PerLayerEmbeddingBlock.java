package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Per-layer embeddings (PLE) from Gemma 3n: besides the usual input embedding, every block
 * has its own small token embedding table {@code E_l [vocab, p]}. After the block runs, its
 * output selects (gates) the token's per-layer embedding and adds it back, projected up:
 *
 * <pre>
 * y = block(h);   y = y + rms((gelu(y W_gate) o E_l[token]) W_up)
 * </pre>
 *
 * <p>The tables are only ever looked up, never multiplied, so they need no fast memory: in
 * Gemma 3n they live off the accelerator (CPU or flash) and only the looked-up rows are
 * streamed in. That splits the parameters into "hot" ones that cost compute and memory
 * bandwidth every step ({@link #hotParameters()}) and "cold" lookup tables
 * ({@link #coldParameters()}), so a model can grow its total size without growing its
 * active compute.
 *
 * <p>A {@link MultiInputLayer}: inputs are the hidden states {@code [batch, steps, features]}
 * and the token ids {@code [batch, steps]} (as floats).
 *
 * <p>Source: Google, "Introducing Gemma 3n: the developer guide",
 * <a href="https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/">developers.googleblog.com</a>.
 */
public final class PerLayerEmbeddingBlock implements MultiInputLayer {

    private final int features;
    private final int perLayer;
    private final Layer block;
    private final Var table;
    private final Var wGate;
    private final Var wUp;
    private final Var gain;

    /**
     * Wraps a block.
     *
     * @param block the block, {@code [batch, steps, features]} to the same shape
     * @param features the width
     * @param vocab the vocabulary
     * @param perLayer the per-layer embedding width {@code p}
     * @param random the random source
     */
    public PerLayerEmbeddingBlock(Layer block, int features, int vocab, int perLayer, RandomGenerator random) {
        this.features = features;
        this.perLayer = perLayer;
        this.block = block;
        this.table = Var.param(Tensor.normal(Shape.of(vocab, perLayer), 0, 1, random));
        this.wGate = MixerOps.weight(random, features, perLayer);
        this.wUp = MixerOps.weight(random, perLayer, features);
        this.gain = MixerOps.filled(0.1f, features);
    }

    /**
     * Describes the block for a {@code Node} graph: inputs {@code [steps, features]} and
     * {@code [steps]}.
     *
     * @param block the wrapped block
     * @param vocab the vocabulary
     * @param perLayer the per-layer embedding width
     * @return the spec
     */
    public static MultiInputLayer.Spec of(LayerSpec block, int vocab, int perLayer) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return inputs.getFirst();
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                Shape h = inputs.getFirst();
                return new PerLayerEmbeddingBlock(block.create(h, random), h.dim(-1), vocab, perLayer, random);
            }
        };
    }

    /**
     * Runs the block and adds the gated per-layer embedding.
     *
     * @param hidden {@code [batch, steps, features]}
     * @param tokens {@code [batch, steps]} token ids
     * @param training whether the block runs in training mode
     * @return {@code [batch, steps, features]}
     */
    public Var forward(Var hidden, Tensor tokens, boolean training) {
        MixerOps.requireSequence(hidden, features, "PerLayerEmbeddingBlock");
        int b = hidden.shape().dim(0);
        int t = hidden.shape().dim(1);
        Var y = block.forward(hidden, training);
        Var e = table.gather(0, tokens.reshape(b * t)).reshape(b, t, perLayer);
        Var g = Activation.GELU.forward(y.matmul(wGate)).mul(e).matmul(wUp);
        return y.add(RefinementOps.rms(g, gain));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("PerLayerEmbeddingBlock takes hidden states and token ids");
        }
        return forward(inputs.get(0), inputs.get(1).value(), training);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.getFirst();
    }

    /** @return the parameters used in matrix products every step: the block, the gate and up projections */
    public List<Var> hotParameters() {
        List<Var> p = new ArrayList<>(block.parameters());
        p.addAll(List.of(wGate, wUp, gain));
        return p;
    }

    /** @return the per-layer lookup table, which can live in slow memory */
    public List<Var> coldParameters() {
        return List.of(table);
    }

    /** @return the wrapped block */
    public Layer block() {
        return block;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = hotParameters();
        p.add(table);
        return p;
    }

    @Override
    public Var auxiliaryLoss() {
        return block.auxiliaryLoss();
    }

    @Override
    public String toString() {
        return "PerLayerEmbeddingBlock(" + block + ", p=" + perLayer + ")";
    }
}
