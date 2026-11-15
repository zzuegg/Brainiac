package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Objects;

/**
 * Reduces a sequence {@code [steps, features]} to one vector {@code [features]}: the mean,
 * maximum or sum over steps, or the first or last step (as a classification token).
 */
public final class SequencePool implements Layer {

    /** How steps are combined. */
    public enum Kind {
        /** the mean over steps */
        MEAN,
        /** the maximum over steps */
        MAX,
        /** the sum over steps */
        SUM,
        /** the first step */
        FIRST,
        /** the last step */
        LAST
    }

    private final Kind kind;

    /**
     * Creates the layer.
     *
     * @param kind how steps are combined
     */
    public SequencePool(Kind kind) {
        this.kind = Objects.requireNonNull(kind, "kind");
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3) {
            throw new IllegalArgumentException("sequence pooling expects [batch, steps, features], got " + s);
        }
        int n = s.dim(0);
        int t = s.dim(1);
        int f = s.dim(2);
        return switch (kind) {
            case MEAN -> input.mean(1, false);
            case SUM -> input.sum(1, false);
            case MAX -> Functional.windowMax(input.permute(0, 2, 1).reshape(n * f, t)).reshape(n, f);
            case FIRST -> input.slice(1, 0, 1).reshape(n, f);
            case LAST -> input.slice(1, t - 1, t).reshape(n, f);
        };
    }

    @Override
    public Shape outputShape(Shape input) {
        if (input.rank() != 2) {
            throw new IllegalArgumentException("sequence pooling expects [steps, features], got " + input);
        }
        return Shape.of(input.dim(1));
    }

    @Override
    public String toString() {
        return "SequencePool(" + kind.name().toLowerCase(java.util.Locale.ROOT) + ")";
    }

    static Config write(SequencePool p) {
        return Config.builder().put("kind", p.kind.name()).build();
    }

    static SequencePool read(Config c) {
        return new SequencePool(Kind.valueOf(c.getString("kind")));
    }
}
