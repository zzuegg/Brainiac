package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Reduces a padded sequence {@code [steps, features]} to one vector {@code [features]},
 * looking only at the steps a {@code [steps]} mask marks with 1: the mean, sum or maximum
 * of the real steps, or the first or last real step. A sequence with no real steps gives
 * zeros. Wire it with {@code Node.merge(MaskedPool.of(kind), mask)}; see {@link PaddingMask}.
 */
public final class MaskedPool implements MultiInputLayer {

    private final SequencePool.Kind kind;

    /**
     * Creates the layer.
     *
     * @param kind how the real steps are combined
     */
    public MaskedPool(SequencePool.Kind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
    }

    /**
     * Describes the layer, for {@code Node.merge(spec, mask)}.
     *
     * @param kind how the real steps are combined
     * @return the spec
     */
    public static MultiInputLayer.Spec of(SequencePool.Kind kind) {
        Objects.requireNonNull(kind, "kind");
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                check(inputs);
                return Shape.of(inputs.get(0).dim(1));
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                check(inputs);
                return new MaskedPool(kind);
            }
        };
    }

    private static void check(List<Shape> inputs) {
        if (inputs.size() != 2 || inputs.get(0).rank() != 2 || inputs.get(1).size() != inputs.get(0).dim(0)) {
            throw new IllegalArgumentException("masked pooling takes a sequence [steps, features] and a mask [steps],"
                    + " got " + inputs);
        }
    }

    /** @return how the real steps are combined */
    public SequencePool.Kind kind() {
        return kind;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("masked pooling takes a sequence and a mask, got " + inputs.size()
                    + " inputs");
        }
        Var x = inputs.get(0);
        Shape s = x.shape();
        if (s.rank() != 3) {
            throw new IllegalArgumentException("masked pooling expects [batch, steps, features], got " + s);
        }
        int n = s.dim(0);
        int t = s.dim(1);
        Tensor mask = inputs.get(1).value().reshape(n, t).to(x.dtype());
        Tensor m = mask.reshape(n, t, 1);
        return switch (kind) {
            case SUM -> x.mul(m).sum(1, false);
            case MEAN -> x.mul(m).sum(1, false).mul(mask.sum(1, true).maximum(Tensor.full(Shape.of(), 1,
                    mask.dtype(), mask.device())).pow(-1));
            case MAX -> {
                Tensor any = mask.max(1, true);
                yield x.add(m.sub(1).mul(1e9f)).max(1, false).mul(any);
            }
            case FIRST, LAST -> x.mul(pick(mask, n, t, kind == SequencePool.Kind.FIRST).to(x.value().device()))
                    .sum(1, false);
        };
    }

    /** One-hot weights {@code [n, t, 1]} at the first or last real step of each sequence. */
    private static Tensor pick(Tensor mask, int n, int t, boolean first) {
        float[] m = mask.to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
        float[] w = new float[n * t];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < t; k++) {
                int j = first ? k : t - 1 - k;
                if (m[i * t + j] != 0) {
                    w[i * t + j] = 1;
                    break;
                }
            }
        }
        return Tensor.of(Shape.of(n, t, 1), w).to(mask.dtype()).to(mask.device());
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        check(inputs);
        return Shape.of(inputs.get(0).dim(1));
    }

    @Override
    public Shape outputShape(Shape input) {
        throw new IllegalStateException("masked pooling needs a mask; wire it with Node.merge(spec, mask)");
    }

    @Override
    public String toString() {
        return "MaskedPool(" + kind.name().toLowerCase(java.util.Locale.ROOT) + ")";
    }

    static Config write(MaskedPool p) {
        return Config.builder().put("kind", p.kind.name()).build();
    }

    static MaskedPool read(Config c) {
        return new MaskedPool(SequencePool.Kind.valueOf(c.getString("kind")));
    }
}
