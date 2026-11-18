package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * Reads one model's tensors from a weight file onto a device, under a name prefix, and puts
 * them into the parameters of brainiac layers. It keeps every tensor it placed, so that a
 * model can be freed at once, and which parameters it filled, so that a loader can check that
 * none was missed.
 */
final class Weights {

    private final WeightSource source;
    private final String prefix;
    private final UnaryOperator<String> rename;
    final Device device;
    final DType dtype;
    final List<Tensor> all;
    private final Set<Var> filled;
    /** Bits per weight of projection matrices, or 0 to keep them in {@link #dtype}. */
    private final int bits;

    Weights(WeightSource source, String prefix, Device device, DType dtype) {
        this(source, prefix, device, dtype, new ArrayList<>());
    }

    Weights(WeightSource source, String prefix, Device device, DType dtype, List<Tensor> all) {
        this(source, prefix, UnaryOperator.identity(), device, dtype, all,
                Collections.newSetFromMap(new IdentityHashMap<>()), 0);
    }

    private Weights(WeightSource source, String prefix, UnaryOperator<String> rename, Device device, DType dtype,
            List<Tensor> all, Set<Var> filled, int bits) {
        this.source = source;
        this.prefix = prefix;
        this.rename = rename;
        this.device = device;
        this.dtype = dtype;
        this.all = all;
        this.filled = filled;
        this.bits = bits;
    }

    /** The same file seen under {@code prefix + more}. */
    Weights at(String more) {
        return new Weights(source, prefix + more, rename, device, dtype, all, filled, bits);
    }

    /** The same file with names translated before they are looked up (for other checkpoint layouts). */
    Weights renamed(UnaryOperator<String> names) {
        return new Weights(source, prefix, names, device, dtype, all, filled, bits);
    }

    /**
     * The same file with projection matrices ({@link #matrix}, {@link #dense}) stored as
     * {@code bits}-bit levels with a scale per output and group of inputs, which multiply in
     * the model's type; 0 keeps them in the model's type.
     */
    Weights quantized(int bits) {
        if (bits != 0 && (bits < 2 || bits > 8)) {
            throw new IllegalArgumentException("weights quantize to 2 to 8 bits, got " + bits);
        }
        return new Weights(source, prefix, rename, device, dtype, all, filled, bits);
    }

    private String full(String name) {
        return rename.apply(prefix + name);
    }

    boolean has(String name) {
        return source.contains(full(name));
    }

    Shape shape(String name) {
        String n = full(name);
        if (!source.contains(n)) {
            throw new IllegalArgumentException("missing weight " + n);
        }
        return source.shape(n);
    }

    /** A tensor as stored, in the model's type. */
    Tensor get(String name) {
        String n = full(name);
        if (!source.contains(n)) {
            throw new IllegalArgumentException("missing weight " + n);
        }
        return source.tensor(n, device, dtype);
    }

    /** Puts {@code value} into a parameter, computed now; {@code from} (if another tensor) is freed. */
    void set(Var parameter, Tensor value, Tensor... from) {
        if (!parameter.shape().equals(value.shape())) {
            throw new IllegalArgumentException(prefix + ": weight of shape " + value.shape() + " for a parameter of "
                    + parameter.shape());
        }
        Tensor.evaluate(List.of(value));
        for (Tensor f : from) {
            if (f != null && f != value) {
                f.release();
            }
        }
        parameter.assign(value);
        all.add(value);
        filled.add(parameter);
    }

    /** Fills a parameter with zeros, for a bias the checkpoint does not have. */
    void zero(Var parameter) {
        set(parameter, Tensor.zeros(parameter.shape(), dtype, device));
    }

    /** Fills a bias from {@code name}, or with zeros if the checkpoint has none. */
    void bias(Var parameter, String name) {
        if (has(name)) {
            set(parameter, get(name));
        } else {
            zero(parameter);
        }
    }

    /** A projection {@code name.weight} {@code [out, in]} (or a 1x1 convolution) into a dense layer {@code [in, out]}. */
    void dense(Dense d, String name) {
        matrix(d.weight(), name + ".weight");
        bias(d.bias(), name + ".bias");
    }

    /** {@code name} {@code [out, in]} (or {@code [out, in, 1, 1]}) transposed into {@code parameter}. */
    void matrix(Var parameter, String name) {
        Tensor w = get(name);
        // a 1x1 convolution used as a projection is the same matrix; the reshape shares memory
        Tensor m = w.rank() == 4 ? w.reshape(w.shape().dim(0), w.shape().dim(1)) : w;
        if (bits == 0) {
            set(parameter, m.transpose(), w);
            return;
        }
        Tensor t = m.transpose();
        Tensor f = t.to(DType.FLOAT32);
        Tensor.evaluate(List.of(f));
        if (t != f) {
            t.release();
        }
        w.release();
        int rows = f.shape().dim(0);
        int group = bits <= 4 ? 32 : 64;
        set(parameter, f.quantized(bits, rows % group == 0 ? group : rows), f);
    }

    /** Rows {@code [start, end)} of a fused {@code [out, in]} projection, transposed. */
    void matrixRows(Var parameter, String name, int start, int end) {
        Tensor w = get(name);
        set(parameter, w.slice(0, start, end).transpose(), w);
    }

    /** Entries {@code [start, end)} of a fused bias. */
    void biasRows(Var parameter, String name, int start, int end) {
        Tensor b = get(name);
        set(parameter, b.slice(0, start, end), b);
    }

    /** A convolution {@code name.weight} {@code [out, in, k, k]} and its bias. */
    void conv(Conv c, String name) {
        set(c.weight(), get(name + ".weight"));
        bias(c.bias(), name + ".bias");
    }

    /** A projection that is a 1x1 convolution or a dense layer, whichever {@code layer} is. */
    void projection(Layer layer, String name) {
        if (layer instanceof Dense d) {
            dense(d, name);
        } else {
            Conv c = (Conv) layer;
            Tensor w = get(name + ".weight");
            // a linear projection's weight [out, in] as a 1x1 convolution [out, in, 1, 1]
            set(c.weight(), w.rank() == 2 ? w.reshape(w.shape().dim(0), w.shape().dim(1), 1, 1) : w);
            bias(c.bias(), name + ".bias");
        }
    }

    /** A norm's scale and shift, {@code name.weight} and {@code name.bias}. */
    void norm(Layer norm, String name) {
        List<Var> p = norm.parameters();
        set(p.get(0), get(name + ".weight"));
        set(p.get(1), get(name + ".bias"));
    }

    /**
     * An attention's projections, whose parameters are listed as {@code wq, wk, wv, wo, bq,
     * bk, bv, bo} starting at {@code first}; missing biases become zeros.
     */
    void attention(List<Var> p, int first, String q, String k, String v, String out) {
        String[] names = {q, k, v, out};
        for (int i = 0; i < 4; i++) {
            matrix(p.get(first + i), names[i] + ".weight");
            bias(p.get(first + 4 + i), names[i] + ".bias");
        }
    }

    /** Throws if a parameter of {@code layer} was not filled from the checkpoint. */
    void checkFilled(Layer layer, String what) {
        int missing = 0;
        for (Var p : layer.parameters()) {
            if (!filled.contains(p)) {
                missing++;
            }
        }
        if (missing > 0) {
            throw new IllegalStateException(missing + " parameters of the " + what + " were not in the checkpoint");
        }
    }
}
