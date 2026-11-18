package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small bidirectional text encoder for retrieval: token embeddings plus learned positions
 * through core {@link TransformerBlock}s, giving one vector per token. Token 0 is padding; its
 * outputs are excluded from {@link #meanPool}. The dense ({@link DenseRetriever}), late-interaction
 * ({@link LateInteractionRetriever}) and learned-sparse ({@link SparseLexicalEncoder}) retrievers
 * are built on it.
 */
public final class TokenEncoder {

    private final int vocabulary;
    private final int dim;
    private final Var embedding;
    private final Var positions;
    private final List<Layer> blocks = new ArrayList<>();

    /**
     * Creates an encoder.
     *
     * @param vocabulary the vocabulary size, including padding token 0
     * @param maxLength the longest input
     * @param dim the width
     * @param heads attention heads
     * @param layers transformer blocks (0 for a bag of embeddings)
     * @param random the initialization source
     */
    public TokenEncoder(int vocabulary, int maxLength, int dim, int heads, int layers, SplittableRandom random) {
        this.vocabulary = vocabulary;
        this.dim = dim;
        embedding = RecsysOps.normal(random, 0.3f, vocabulary, dim);
        positions = RecsysOps.normal(random, 0.05f, maxLength, dim);
        for (int l = 0; l < layers; l++) {
            blocks.add(TransformerBlock.of(heads).ffnSize(2 * dim).create(Shape.of(maxLength, dim), random));
        }
    }

    /** @return the width */
    public int dim() {
        return dim;
    }

    /** @return the vocabulary size */
    public int vocabulary() {
        return vocabulary;
    }

    /** @return the {@code [vocabulary, dim]} token embeddings */
    public Var embeddings() {
        return embedding;
    }

    /**
     * Encodes token sequences of equal length.
     *
     * @param tokens {@code [batch][length]} token ids
     * @param training whether dropout-style layers train
     * @return {@code [batch, length, dim]}
     */
    public Var encode(int[][] tokens, boolean training) {
        int b = tokens.length;
        int len = tokens[0].length;
        Var x = embedding.gather(0, RecsysOps.ids(tokens).reshape(b * len)).reshape(b, len, dim)
                .add(positions.slice(0, 0, len));
        for (Layer block : blocks) {
            x = block.forward(x, training);
        }
        return x;
    }

    /**
     * @param tokens {@code [batch][length]} token ids
     * @return {@code [batch, length, 1]}: 1 for real tokens, 0 for padding
     */
    public static Tensor mask(int[][] tokens) {
        int b = tokens.length;
        int len = tokens[0].length;
        float[] m = new float[b * len];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                m[i * len + j] = tokens[i][j] != 0 ? 1 : 0;
            }
        }
        return Tensor.of(Shape.of(b, len, 1), m);
    }

    /**
     * The mean of the real tokens' vectors.
     *
     * @param hidden {@code [batch, length, dim]}
     * @param tokens the token ids that produced it
     * @return {@code [batch, dim]}
     */
    public static Var meanPool(Var hidden, int[][] tokens) {
        Tensor m = mask(tokens);
        return hidden.mul(m).sum(1, false).div(m.sum(1, false).add(1e-6f));
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, positions));
        for (Layer block : blocks) {
            p.addAll(block.parameters());
        }
        return p;
    }
}
