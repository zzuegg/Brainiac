package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * A fixed choice of activation for every single unit; see {@link Activation#perUnit}.
 *
 * <p>Computed as {@code sum_k f_k(x) * mask_k}, one mask per distinct activation, so it is
 * only element-wise operations: a backend that fuses element-wise chains (the OpenCL one)
 * runs the whole choice as one kernel instead of one launch per activation.
 */
final class PerUnitActivation implements Activation {

    private final Activation[] distinct;
    private final int[] choice;
    private Tensor[] masks;

    private PerUnitActivation(Activation[] distinct, int[] choice) {
        this.distinct = distinct;
        this.choice = choice;
    }

    static Activation of(Activation... perUnit) {
        if (perUnit.length == 0) {
            throw new IllegalArgumentException("perUnit needs at least one unit");
        }
        List<Activation> distinct = new ArrayList<>();
        int[] choice = new int[perUnit.length];
        for (int u = 0; u < perUnit.length; u++) {
            Activation a = Objects.requireNonNull(perUnit[u], "activation");
            if (a.boundWidth() >= 0 || a.bind(1) != a || !a.parameters().isEmpty()) {
                throw new IllegalArgumentException("perUnit takes activations without parameters, got " + a);
            }
            int k = distinct.indexOf(a);
            if (k < 0) {
                k = distinct.size();
                distinct.add(a);
            }
            choice[u] = k;
        }
        if (distinct.size() == 1) {
            return distinct.get(0);
        }
        return new PerUnitActivation(distinct.toArray(Activation[]::new), choice);
    }

    static Function<Config, Activation> factory() {
        return c -> {
            Activation[] distinct = new Activation[c.getInt("count")];
            for (int i = 0; i < distinct.length; i++) {
                distinct[i] = c.getActivation("a" + i);
            }
            int[] choice = c.getInts("choice");
            Activation[] perUnit = new Activation[choice.length];
            for (int u = 0; u < choice.length; u++) {
                perUnit[u] = distinct[choice[u]];
            }
            return of(perUnit);
        };
    }

    /** @return the activation of every unit */
    Activation[] units() {
        Activation[] out = new Activation[choice.length];
        for (int u = 0; u < choice.length; u++) {
            out[u] = distinct[choice[u]];
        }
        return out;
    }

    @Override
    public Activation bind(int width) {
        if (width != choice.length) {
            throw new IllegalArgumentException(this + " has " + choice.length + " units, cannot be used with width "
                    + width);
        }
        return this;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.dim(s.rank() - 1) != choice.length) {
            throw new IllegalArgumentException(this + " has " + choice.length + " units, got input " + s);
        }
        Tensor[] m = masks(input.value().device(), input.value().dtype());
        Var out = null;
        for (int k = 0; k < distinct.length; k++) {
            Var term = distinct[k].forward(input, training).mul(m[k]);
            out = out == null ? term : out.add(term);
        }
        return out;
    }

    private synchronized Tensor[] masks(Device device, DType dtype) {
        if (masks == null || !masks[0].device().equals(device) || masks[0].dtype() != dtype) {
            Tensor[] t = new Tensor[distinct.length];
            for (int k = 0; k < distinct.length; k++) {
                float[] v = new float[choice.length];
                for (int u = 0; u < choice.length; u++) {
                    v[u] = choice[u] == k ? 1 : 0;
                }
                t[k] = Tensor.of(Shape.of(1, choice.length), v).to(dtype).to(device);
            }
            masks = t;
        }
        return masks;
    }

    @Override
    public Init preferredInit() {
        Init init = distinct[0].preferredInit();
        for (Activation a : distinct) {
            if (a.preferredInit() != init) {
                return Init.XAVIER;
            }
        }
        return init;
    }

    @Override
    public int boundWidth() {
        return -1;
    }

    @Override
    public String name() {
        return "per_unit";
    }

    @Override
    public Config config() {
        Config.Builder b = Config.builder().put("count", distinct.length).put("choice", choice.clone());
        for (int i = 0; i < distinct.length; i++) {
            b.put("a" + i, distinct[i]);
        }
        return b.build();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof PerUnitActivation p && Arrays.equals(units(), p.units());
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(units());
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("perUnit(");
        for (int k = 0; k < distinct.length; k++) {
            int n = 0;
            for (int c : choice) {
                n += c == k ? 1 : 0;
            }
            sb.append(k > 0 ? ", " : "").append(n).append(' ').append(distinct[k]);
        }
        return sb.append(')').toString();
    }
}
