package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The FT-Transformer (Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
 * 2021, <a href="https://arxiv.org/abs/2106.11959">arXiv:2106.11959</a>): a
 * {@link FeatureTokenizer} turns every column into a token, a {@code [CLS]} token is prepended,
 * a stack of core pre-norm {@link TransformerBlock}s lets the columns attend to each other, and a
 * head (layer norm, ReLU, linear) reads the {@code [CLS]} output.
 *
 * <p>Input {@code [batch, columns]}, output {@code [batch, outputs]}.
 */
public final class FtTransformer implements Layer {

    private final FeatureTokenizer tokenizer;
    private final List<MultiInputLayer> blocks = new ArrayList<>();
    private final LayerNorm norm;
    private final Var head;
    private final Var headBias;

    /**
     * Creates the model.
     *
     * @param tokenizer the column tokenizer; must have a {@code [CLS]} token
     * @param blocks transformer blocks
     * @param heads attention heads, dividing the token width
     * @param outputs the output width
     * @param random the initialization source
     */
    public FtTransformer(FeatureTokenizer tokenizer, int blocks, int heads, int outputs, SplittableRandom random) {
        this.tokenizer = tokenizer;
        int d = tokenizer.dim();
        Shape tokens = Shape.of(tokenizer.tokens(), d);
        for (int i = 0; i < blocks; i++) {
            this.blocks.add((MultiInputLayer) TransformerBlock.of(heads).ffnSize(2 * d).create(tokens, random));
        }
        this.norm = LayerNorm.of(d);
        this.head = ClassicOps.matrix(random, d, outputs);
        this.headBias = TabularOps.zeros(outputs);
    }

    /**
     * @param blocks transformer blocks
     * @param dim the token width
     * @param heads attention heads
     * @param outputs the output width
     * @return a spec with a linear tokenizer, for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int blocks, int dim, int heads, int outputs) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, outputs),
                (s, r) -> new FtTransformer(FeatureTokenizer.linear(s.dim(-1), dim, r), blocks, heads, outputs, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var x = tokenizer.forward(input, training);
        for (MultiInputLayer block : blocks) {
            x = block.forward(List.of(x), training);
        }
        int b = x.shape().dim(0);
        Var cls = x.slice(1, 0, 1).reshape(b, tokenizer.dim());
        return norm.forward(cls).relu().matmul(head).add(headBias);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(tokenizer.parameters());
        for (MultiInputLayer block : blocks) {
            p.addAll(block.parameters());
        }
        p.addAll(norm.parameters());
        p.add(head);
        p.add(headBias);
        return p;
    }

    @Override
    public String toString() {
        return "FtTransformer(" + tokenizer + ", " + blocks.size() + " blocks)";
    }
}
