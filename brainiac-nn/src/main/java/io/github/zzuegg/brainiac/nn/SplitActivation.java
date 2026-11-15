package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/** Different activations for different groups of units; see {@link Activation#split}. */
final class SplitActivation implements Activation {

    private final Activation[] parts;
    private final int[] fixedSizes;
    private final int width;
    private final int[] sizes;

    private SplitActivation(Activation[] parts, int[] fixedSizes, int width, int[] sizes) {
        this.parts = parts;
        this.fixedSizes = fixedSizes;
        this.width = width;
        this.sizes = sizes;
    }

    static Activation of(int[] fixedSizes, Activation... parts) {
        if (parts.length == 0) {
            throw new IllegalArgumentException("split needs at least one activation");
        }
        for (Activation a : parts) {
            Objects.requireNonNull(a, "activation");
        }
        if (fixedSizes != null) {
            if (fixedSizes.length != parts.length) {
                throw new IllegalArgumentException("split got " + fixedSizes.length + " sizes for " + parts.length
                        + " activations");
            }
            for (int s : fixedSizes) {
                if (s < 1) {
                    throw new IllegalArgumentException("split sizes must be at least 1, got " + Arrays.toString(fixedSizes));
                }
            }
            fixedSizes = fixedSizes.clone();
        }
        return new SplitActivation(parts.clone(), fixedSizes, -1, null);
    }

    static Function<Config, Activation> factory() {
        return c -> {
            Activation[] parts = new Activation[c.getInt("count")];
            for (int i = 0; i < parts.length; i++) {
                parts[i] = c.getActivation("a" + i);
            }
            return of(c.has("sizes") ? c.getInts("sizes") : null, parts);
        };
    }

    private int[] sizesFor(int width) {
        if (fixedSizes != null) {
            int total = Arrays.stream(fixedSizes).sum();
            if (total != width) {
                throw new IllegalArgumentException("split sizes " + Arrays.toString(fixedSizes) + " add up to " + total
                        + ", but the layer has " + width + " units");
            }
            return fixedSizes.clone();
        }
        if (width < parts.length) {
            throw new IllegalArgumentException("cannot split " + width + " units between " + parts.length + " activations");
        }
        int[] s = new int[parts.length];
        for (int i = 0; i < s.length; i++) {
            s[i] = width / parts.length + (i < width % parts.length ? 1 : 0);
        }
        return s;
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
        int[] s = sizesFor(width);
        Activation[] bound = new Activation[parts.length];
        for (int i = 0; i < parts.length; i++) {
            bound[i] = parts[i].bind(s[i]);
        }
        return new SplitActivation(bound, fixedSizes, width, s);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (width < 0) {
            throw new IllegalStateException("split must be bound to a width first; use it in a dense(...) or"
                    + " activation(...) step, or call bind(width)");
        }
        Shape s = input.shape();
        int axis = s.rank() - 1;
        if (s.dim(axis) != width) {
            throw new IllegalArgumentException(this + " is bound to width " + width + ", got input " + s);
        }
        Var[] outs = new Var[parts.length];
        int start = 0;
        for (int i = 0; i < parts.length; i++) {
            outs[i] = parts[i].forward(input.slice(axis, start, start + sizes[i]), training);
            start += sizes[i];
        }
        return outs.length == 1 ? outs[0] : Var.concat(axis, outs);
    }

    @Override
    public List<Var> parameters() {
        return collect(Activation::parameters);
    }

    @Override
    public List<Var> buffers() {
        return collect(Activation::buffers);
    }

    private List<Var> collect(Function<Activation, List<Var>> get) {
        Set<Var> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Var> out = new ArrayList<>();
        for (Activation a : parts) {
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
        Init init = parts[0].preferredInit();
        for (Activation a : parts) {
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
        return "split";
    }

    @Override
    public Config config() {
        Config.Builder b = Config.builder().put("count", parts.length);
        for (int i = 0; i < parts.length; i++) {
            b.put("a" + i, parts[i]);
        }
        if (fixedSizes != null) {
            b.put("sizes", fixedSizes);
        }
        return b.build();
    }

    /** @return the activations, one per group, bound if this split is bound */
    List<Activation> parts() {
        return List.of(parts);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("split(");
        for (int i = 0; i < parts.length; i++) {
            sb.append(i > 0 ? ", " : "").append(parts[i]);
            int[] s = sizes != null ? sizes : fixedSizes;
            if (s != null) {
                sb.append(" x").append(s[i]);
            }
        }
        return sb.append(')').toString();
    }
}
