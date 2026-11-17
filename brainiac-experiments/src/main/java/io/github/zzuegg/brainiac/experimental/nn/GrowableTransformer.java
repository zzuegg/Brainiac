package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small decoder-only language model (token and learned position embeddings, a stack of
 * {@link GrowableTransformerBlock}s, a final RMS norm and an output projection) that can
 * grow in every direction while it trains:
 *
 * <ul>
 *   <li>FFN width and attention heads per block, function-preserving
 *       ({@link #widenFfn}, {@link #addHeadGroup});</li>
 *   <li>depth, function-preserving, by inserting a block that starts as the identity
 *       ({@link #addBlock});</li>
 *   <li>progressive stacking: doubling the depth by copying the trained blocks on top
 *       ({@link #stack()}), which is not function-preserving but transfers what the
 *       lower blocks learned (Gong et al. 2019,
 *       <a href="https://proceedings.mlr.press/v97/gong19a.html">ICML</a>).</li>
 * </ul>
 *
 * <p>Takes token ids {@code [batch, time]} (as float values) and returns logits
 * {@code [batch, time, vocabulary]}. After any growth call {@link #parameters()} returns
 * new variables, so the optimizer's state must be rebuilt (see {@code GrowingTrainer}).
 */
public final class GrowableTransformer implements Layer {

    private final int vocabulary;
    private final int maxLength;
    private final int model;
    private Var embedding;
    private Var positions;
    private Var finalGamma;
    private Var unembed;
    private final List<GrowableTransformerBlock> blocks = new ArrayList<>();

    private GrowableTransformer(int vocabulary, int maxLength, int model) {
        this.vocabulary = vocabulary;
        this.maxLength = maxLength;
        this.model = model;
    }

    /**
     * Creates a model.
     *
     * @param vocabulary the number of token ids
     * @param maxLength the longest sequence
     * @param block the design of every starting block (its model width is the model's)
     * @param depth the number of starting blocks
     * @param random the random source
     * @return the model
     */
    public static GrowableTransformer of(int vocabulary, int maxLength, GrowableTransformerBlock.Config block,
            int depth, SplittableRandom random) {
        GrowableTransformer t = new GrowableTransformer(vocabulary, maxLength, block.model());
        int d = block.model();
        t.embedding = Var.param(gaussian(Shape.of(vocabulary, d), 0.5, random));
        t.positions = Var.param(gaussian(Shape.of(maxLength, d), 0.1, random));
        t.finalGamma = Var.param(Tensor.ones(Shape.of(d)));
        t.unembed = Var.param(gaussian(Shape.of(d, vocabulary), 1 / Math.sqrt(d), random));
        for (int i = 0; i < depth; i++) {
            t.blocks.add(GrowableTransformerBlock.of(block, random));
        }
        return t;
    }

    /**
     * A spec for {@code Model.sequential().input(time).add(...)}.
     *
     * @param vocabulary the number of token ids
     * @param block the block design
     * @param depth the number of blocks
     * @return the spec
     */
    public static LayerSpec spec(int vocabulary, GrowableTransformerBlock.Config block, int depth) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] dims = java.util.Arrays.copyOf(input.toArray(), input.rank() + 1);
                dims[input.rank()] = vocabulary;
                return Shape.of(dims);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return of(vocabulary, input.dim(-1), block, depth, random);
            }
        };
    }

    private static Tensor gaussian(Shape s, double scale, SplittableRandom r) {
        float[] v = new float[Math.toIntExact(s.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * scale);
        }
        return Tensor.of(s, v);
    }

    @Override
    public Var forward(Var ids) {
        Shape s = ids.shape();
        int b = s.dim(0);
        int t = s.dim(1);
        if (t > maxLength) {
            throw new IllegalArgumentException("sequence of " + t + " exceeds " + maxLength);
        }
        Var x = embedding.gather(0, ids.value().reshape(b * t)).reshape(b, t, model)
                .add(positions.slice(0, 0, t));
        for (GrowableTransformerBlock block : blocks) {
            x = block.forward(x);
        }
        Var flat = x.reshape(b * t, model);
        flat = flat.div(flat.square().mean(1, true).add(1e-5f).sqrt()).mul(finalGamma);
        return flat.matmul(unembed).reshape(b, t, vocabulary);
    }

    /**
     * The mean next-token cross-entropy of logits against target ids.
     *
     * @param logits {@code [batch, time, vocabulary]}
     * @param targets {@code [batch, time]} ids, negative to skip
     * @return the scalar loss
     */
    public static Var loss(Var logits, Tensor targets) {
        Shape s = logits.shape();
        int rows = s.dim(0) * s.dim(1);
        return Functional.crossEntropy(logits.reshape(rows, s.dim(2)), targets.reshape(rows));
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>(List.of(embedding, positions));
        for (GrowableTransformerBlock b : blocks) {
            out.addAll(b.parameters());
        }
        out.add(finalGamma);
        out.add(unembed);
        return out;
    }

    @Override
    public List<Var> buffers() {
        List<Var> out = new ArrayList<>();
        blocks.forEach(b -> out.addAll(b.buffers()));
        return out;
    }

    @Override
    public Shape outputShape(Shape input) {
        int[] dims = java.util.Arrays.copyOf(input.toArray(), input.rank() + 1);
        dims[input.rank()] = vocabulary;
        return Shape.of(dims);
    }

    /** @return the blocks, bottom first */
    public List<GrowableTransformerBlock> blocks() {
        return List.copyOf(blocks);
    }

    /** @return the number of trainable values */
    public long parameterCount() {
        long n = 0;
        for (Var p : parameters()) {
            n += p.value().size();
        }
        return n;
    }

    /**
     * Widens a block's FFN, keeping the function.
     *
     * @param block the block index
     * @param width the new width
     * @param random the random source
     */
    public void widenFfn(int block, int width, SplittableRandom random) {
        blocks.get(block).widenFfn(width, random);
    }

    /**
     * Adds a key/value head group to a block, keeping the function.
     *
     * @param block the block index
     * @param random the random source
     */
    public void addHeadGroup(int block, SplittableRandom random) {
        blocks.get(block).addHeadGroup(random);
    }

    /**
     * Inserts a block that starts as the identity, keeping the function.
     *
     * @param index where to insert it (0 is the bottom)
     * @param config its design
     * @param random the random source
     */
    public void addBlock(int index, GrowableTransformerBlock.Config config, SplittableRandom random) {
        blocks.add(index, GrowableTransformerBlock.identity(config, random));
    }

    /** Progressive stacking: doubles the depth by copying the current blocks on top. */
    public void stack() {
        List<GrowableTransformerBlock> copies = blocks.stream().map(GrowableTransformerBlock::copy).toList();
        blocks.addAll(copies);
    }

    @Override
    public String toString() {
        return "GrowableTransformer(vocab " + vocabulary + ", width " + model + ", " + blocks + ")";
    }
}
