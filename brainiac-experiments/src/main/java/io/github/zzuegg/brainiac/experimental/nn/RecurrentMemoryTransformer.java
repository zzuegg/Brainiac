package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The Recurrent Memory Transformer (Bulatov, Kuratov and Burtsev 2022,
 * <a href="https://arxiv.org/abs/2207.06881">arXiv:2207.06881</a>): a stack of ordinary
 * transformer blocks run segment by segment over a long input, with {@code m} learned memory
 * tokens carried from one segment to the next. Each segment is processed as
 * {@code [read memory; segment; write memory]}; the outputs at the write positions become the
 * next segment's read memory. The first segment reads the learned initial memory.
 *
 * <p>Memory makes the transformer a segment-level recurrent network: information crosses
 * segment boundaries only through the {@code m} tokens, so attention cost stays
 * {@code O(segment^2)} for any length. Training backpropagates through time across segments;
 * {@link #bptt(int)} truncates it to the last few. Blocks are core
 * {@link TransformerBlock}s built from a {@link TransformerBlock.Spec}, plus learned position
 * embeddings within a segment. Input {@code [batch, steps, features]}, output the same shape
 * (memory positions removed).
 */
public final class RecurrentMemoryTransformer implements Layer {

    private final int features;
    private final int segment;
    private final int memoryTokens;
    private final List<Layer> blocks = new ArrayList<>();
    private final Var memory;
    private final Var positions;
    private int bptt = Integer.MAX_VALUE;
    private Tensor lastMemory;

    /**
     * Creates the layer.
     *
     * @param features the features per step
     * @param segment the steps per segment
     * @param memoryTokens the memory tokens carried between segments; 0 makes segments independent
     * @param block the transformer block to stack
     * @param depth how many blocks
     * @param random the random source for the weights
     */
    public RecurrentMemoryTransformer(int features, int segment, int memoryTokens, TransformerBlock.Spec block,
            int depth, SplittableRandom random) {
        if (features < 1 || segment < 1 || memoryTokens < 0 || depth < 1) {
            throw new IllegalArgumentException("bad sizes");
        }
        this.features = features;
        this.segment = segment;
        this.memoryTokens = memoryTokens;
        int length = segment + 2 * memoryTokens;
        for (int i = 0; i < depth; i++) {
            blocks.add(block.create(Shape.of(length, features), random));
        }
        memory = Var.param(Tensor.normal(Shape.of(1, Math.max(1, memoryTokens), features), 0, 0.5f, random));
        positions = Var.param(Tensor.normal(Shape.of(1, length, features), 0, 0.1f, random));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param segment the steps per segment
     * @param memoryTokens the memory tokens
     * @param block the transformer block to stack
     * @param depth how many blocks
     * @return the spec
     */
    public static LayerSpec spec(int segment, int memoryTokens, TransformerBlock.Spec block, int depth) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new RecurrentMemoryTransformer(input.dim(-1), segment, memoryTokens, block, depth, random);
            }
        };
    }

    /**
     * @param segments how many segments gradients flow back through the memory; older memory is detached
     * @return this layer
     */
    public RecurrentMemoryTransformer bptt(int segments) {
        if (segments < 1) {
            throw new IllegalArgumentException("bptt must be at least 1");
        }
        this.bptt = segments;
        return this;
    }

    /** @return the memory after the last segment of the last forward pass, {@code [batch, m, features]}, or null */
    public Tensor lastMemory() {
        return lastMemory;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int steps = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != features || steps % segment != 0) {
            throw new IllegalArgumentException(this + " expects [batch, k * " + segment + ", " + features + "], got "
                    + input.shape());
        }
        int m = memoryTokens;
        Var mem = m == 0 ? null : memory.expand(Shape.of(n, m, features));
        Var writeInit = m == 0 ? null : Var.of(Tensor.zeros(Shape.of(n, m, features)));
        int segments = steps / segment;
        Var[] outs = new Var[segments];
        for (int s = 0; s < segments; s++) {
            if (mem != null && segments - s > bptt && s > 0) {
                mem = mem.detach();
            }
            Var seg = input.slice(1, s * segment, (s + 1) * segment);
            Var x = m == 0 ? seg : Var.concat(1, mem, seg, writeInit);
            x = x.add(positions);
            for (Layer b : blocks) {
                x = b.forward(x);
            }
            outs[s] = x.slice(1, m, m + segment);
            if (m > 0) {
                mem = x.slice(1, m + segment, 2 * m + segment);
            }
        }
        lastMemory = mem == null ? null : mem.value();
        return segments == 1 ? outs[0] : Var.concat(1, outs);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Layer b : blocks) {
            p.addAll(b.parameters());
        }
        if (memoryTokens > 0) {
            p.add(memory);
        }
        p.add(positions);
        return p;
    }

    @Override
    public String toString() {
        return "RecurrentMemoryTransformer(" + blocks.size() + " blocks, segment " + segment + ", " + memoryTokens
                + " memory tokens)";
    }
}
