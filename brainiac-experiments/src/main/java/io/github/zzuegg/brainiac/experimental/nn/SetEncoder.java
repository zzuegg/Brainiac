package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A permutation-invariant set encoder {@code [batch, elements, inputs] -> [batch, outputs]}:
 *
 * <ul>
 *   <li>{@link Kind#DEEP_SETS}: {@code rho(sum_i phi(x_i))} with two-layer ReLU MLPs; every
 *       permutation-invariant function of a set can be written this way, and sum pooling
 *       lets it count.</li>
 *   <li>{@link Kind#SET_TRANSFORMER}: an input projection, two induced set attention blocks
 *       (ISAB: {@code m} learned inducing points attend to the set, then the set attends back
 *       to them, {@code O(n m)} instead of {@code O(n^2)}), and pooling by multihead attention
 *       (PMA, one learned seed attends over the set). Elements interact, so it models
 *       pairwise structure such as clusters that sum pooling cannot.</li>
 * </ul>
 *
 * <p>The ISAB and PMA blocks are {@link LatentCrossAttention}s.
 *
 * <p>Papers: Zaheer et al., "Deep Sets" (NeurIPS 2017),
 * <a href="https://arxiv.org/abs/1703.06114">arXiv:1703.06114</a>; Lee et al., "Set Transformer"
 * (ICML 2019), <a href="https://arxiv.org/abs/1810.00825">arXiv:1810.00825</a>.
 */
public final class SetEncoder implements Layer {

    /** The encoder family. */
    public enum Kind {
        /** sum-pooled per-element MLP */
        DEEP_SETS,
        /** induced set attention and attention pooling */
        SET_TRANSFORMER
    }

    private final int inputs;
    private final int width;
    private final int outputs;
    private final Kind kind;
    private final List<Var> weights = new ArrayList<>();
    private final List<LatentCrossAttention[]> isabs = new ArrayList<>();
    private final LatentCrossAttention pma;

    /**
     * Creates the encoder.
     *
     * @param inputs the features per element
     * @param width the hidden width
     * @param outputs the output features
     * @param kind the family
     * @param inducing the inducing points per ISAB (Set Transformer)
     * @param heads the attention heads (Set Transformer); must divide {@code width}
     * @param random the source of initial weights
     */
    public SetEncoder(int inputs, int width, int outputs, Kind kind, int inducing, int heads,
            RandomGenerator random) {
        this.inputs = inputs;
        this.width = width;
        this.outputs = outputs;
        this.kind = kind;
        if (kind == Kind.DEEP_SETS) {
            int[][] dims = {{inputs, width}, {width, width}, {width, width}, {width, outputs}};
            for (int[] d : dims) {
                weights.add(ClassicOps.matrix(random, d[0], d[1]));
                weights.add(ClassicOps.zeros(d[1]));
            }
            pma = null;
        } else {
            weights.add(ClassicOps.matrix(random, inputs, width));
            weights.add(ClassicOps.zeros(width));
            for (int i = 0; i < 2; i++) {
                isabs.add(new LatentCrossAttention[] {new LatentCrossAttention(width, width, heads, inducing, random),
                    new LatentCrossAttention(width, width, heads, 0, random)});
            }
            pma = new LatentCrossAttention(width, width, heads, 1, random);
            weights.add(ClassicOps.matrix(random, width, outputs));
            weights.add(ClassicOps.zeros(outputs));
        }
    }

    /**
     * Describes the encoder for model builders on {@code [elements, inputs]}.
     *
     * @param width the hidden width
     * @param outputs the output features
     * @param kind the family
     * @return the spec
     */
    public static LayerSpec spec(int width, int outputs, Kind kind) {
        return ClassicOps.spec(s -> Shape.of(outputs),
                (s, r) -> new SetEncoder(s.dim(1), width, outputs, kind, 8, 4, r));
    }

    private Var dense(Var x, int i) {
        return x.matmul(weights.get(2 * i)).add(weights.get(2 * i + 1));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, elements, " + inputs + "], got "
                    + input.shape());
        }
        int b = input.shape().dim(0);
        if (kind == Kind.DEEP_SETS) {
            Var phi = dense(dense(input, 0).relu(), 1).relu();
            Var pooled = phi.sum(1, false);
            return dense(dense(pooled, 2).relu(), 3);
        }
        Var x = dense(input, 0);
        for (LatentCrossAttention[] isab : isabs) {
            Var h = isab[0].forward(List.of(x), training);
            x = isab[1].attend(x, h, training);
        }
        Var pooled = pma.forward(List.of(x), training).reshape(b, width);
        return dense(pooled, 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(outputs);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        for (LatentCrossAttention[] isab : isabs) {
            p.addAll(isab[0].parameters());
            p.addAll(isab[1].parameters());
        }
        if (pma != null) {
            p.addAll(pma.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "SetEncoder(" + kind + ", " + inputs + " -> " + outputs + ")";
    }
}
