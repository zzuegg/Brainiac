package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/** A learned blend of several activations; see {@link Activation#mix}. */
final class MixedActivation implements Activation {

    private static final Activation SOFTMAX_OVER_CANDIDATES = Activation.softmax(0);

    private final Activation[] candidates;
    private final int width;
    private final Var logits;

    private MixedActivation(Activation[] candidates, int width, Var logits) {
        this.candidates = candidates;
        this.width = width;
        this.logits = logits;
    }

    static Activation of(Activation... candidates) {
        if (candidates.length < 2) {
            throw new IllegalArgumentException("mix needs at least two activations, got " + candidates.length);
        }
        for (Activation a : candidates) {
            Objects.requireNonNull(a, "activation");
        }
        return new MixedActivation(candidates.clone(), -1, null);
    }

    static Function<Config, Activation> factory() {
        return c -> {
            Activation[] candidates = new Activation[c.getInt("count")];
            for (int i = 0; i < candidates.length; i++) {
                candidates[i] = c.getActivation("a" + i);
            }
            return of(candidates);
        };
    }

    @Override
    public Activation bind(int width) {
        if (this.width == width) {
            return this;
        }
        if (this.width >= 0) {
            throw new IllegalArgumentException(this + " is bound to width " + this.width
                    + " and cannot be used with width " + width);
        }
        if (width < 1) {
            throw new IllegalArgumentException("mix needs a width of at least 1, got " + width);
        }
        Activation[] bound = new Activation[candidates.length];
        for (int i = 0; i < candidates.length; i++) {
            bound[i] = candidates[i].bind(width);
        }
        // equal logits: every unit starts as the plain average of the candidates
        return new MixedActivation(bound, width, Var.param(Tensor.zeros(Shape.of(candidates.length, width))));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (width < 0) {
            throw new IllegalStateException("mix has learnable weights and must be bound to a width first; use it"
                    + " in a dense(...) or activation(...) step, or call bind(width)");
        }
        Shape s = input.shape();
        if (s.dim(s.rank() - 1) != width) {
            throw new IllegalArgumentException(this + " is bound to width " + width + ", got input " + s);
        }
        Var weights = SOFTMAX_OVER_CANDIDATES.forward(logits);
        Var out = null;
        for (int k = 0; k < candidates.length; k++) {
            Var term = candidates[k].forward(input, training).mul(weights.slice(0, k, k + 1));
            out = out == null ? term : out.add(term);
        }
        return out;
    }

    /**
     * @return the current blend, shape {@code [candidates, width]}; each column sums to 1
     */
    Tensor weights() {
        return SOFTMAX_OVER_CANDIDATES.forward(Var.of(logits.value())).value();
    }

    @Override
    public List<Var> parameters() {
        return collect(Activation::parameters, logits);
    }

    @Override
    public List<Var> buffers() {
        return collect(Activation::buffers, null);
    }

    private List<Var> collect(Function<Activation, List<Var>> get, Var own) {
        Set<Var> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Var> out = new ArrayList<>();
        if (own != null) {
            seen.add(own);
            out.add(own);
        }
        for (Activation a : candidates) {
            for (Var v : get.apply(a)) {
                if (seen.add(v)) {
                    out.add(v);
                }
            }
        }
        return out;
    }

    @Override
    public Init preferredInit() {
        Init init = candidates[0].preferredInit();
        for (Activation a : candidates) {
            if (a.preferredInit() != init) {
                return Init.XAVIER;
            }
        }
        return init;
    }

    @Override
    public int boundWidth() {
        return width;
    }

    @Override
    public String name() {
        return "mix";
    }

    @Override
    public Config config() {
        Config.Builder b = Config.builder().put("count", candidates.length);
        for (int i = 0; i < candidates.length; i++) {
            b.put("a" + i, candidates[i]);
        }
        return b.build();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("mix(");
        for (int i = 0; i < candidates.length; i++) {
            sb.append(i > 0 ? ", " : "").append(candidates[i]);
        }
        return sb.append(')').toString();
    }
}
