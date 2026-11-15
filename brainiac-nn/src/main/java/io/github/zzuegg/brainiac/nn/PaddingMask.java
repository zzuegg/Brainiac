package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Turns token ids into a padding mask: 1 where the id is a real token and 0 where it is
 * the padding id. Feed it to a {@link TransformerBlock.Spec#keyMask(boolean) masked
 * transformer}, {@link MultiHeadAttention.Spec#keyMask(boolean) masked attention} or a
 * {@link MaskedPool} so padded steps are ignored.
 *
 * <pre>{@code
 * Node ids = Node.input(20);
 * Node mask = ids.layer(new PaddingMask(0));
 * Node x = ids.add(Embedding.of(5000, 64).paddingIndex(0)).positionalEncoding()
 *         .merge(TransformerBlock.of(4).keyMask(true), mask)
 *         .merge(MaskedPool.of(SequencePool.Kind.MEAN), mask);
 * }</pre>
 */
public final class PaddingMask implements Layer {

    private final int paddingIndex;

    /**
     * Creates the layer.
     *
     * @param paddingIndex the token id used for padding
     */
    public PaddingMask(int paddingIndex) {
        this.paddingIndex = paddingIndex;
    }

    /** @return the token id used for padding */
    public int paddingIndex() {
        return paddingIndex;
    }

    @Override
    public Var forward(Var input) {
        Tensor ids = input.value();
        return Var.of(ids.equalTo(Tensor.full(Shape.of(), paddingIndex, ids.dtype(), ids.device())).neg().add(1));
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "PaddingMask(" + paddingIndex + ")";
    }

    static Config write(PaddingMask m) {
        return Config.builder().put("paddingIndex", m.paddingIndex).build();
    }

    static PaddingMask read(Config c) {
        return new PaddingMask(c.getInt("paddingIndex"));
    }
}
