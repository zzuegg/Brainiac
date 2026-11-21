package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.StyleNorm;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;

/**
 * Reads PyTorch modules' weights into Brainiac's layouts, keeping every tensor it creates so
 * the owner can release them: linear layers transposed, weight-normalized convolutions
 * folded, LSTMs and style norms as layers.
 */
final class Weights implements AutoCloseable {

    private final WeightSource source;
    private final Device device;
    private final DType dtype;
    private final List<Tensor> owned = new ArrayList<>();

    Weights(WeightSource source, Device device, DType dtype) {
        this.source = source;
        this.device = device;
        this.dtype = dtype;
    }

    Device device() {
        return device;
    }

    DType dtype() {
        return dtype;
    }

    boolean contains(String name) {
        return source.contains(name);
    }

    Shape shape(String name) {
        return source.shape(name);
    }

    /** A tensor as stored. */
    Var read(String name) {
        if (!source.contains(name)) {
            throw new IllegalArgumentException("the weights have no " + name);
        }
        return keep(source.tensor(name, device, dtype));
    }

    /** A tensor reshaped. */
    Var read(String name, int... shape) {
        return keep(Tensor.of(Shape.of(shape), floats(name), device).to(dtype));
    }

    /** The values of a tensor on the CPU. */
    float[] floats(String name) {
        Tensor t = source.tensor(name, Device.cpu(), DType.FLOAT32);
        float[] f = t.toArray();
        t.release();
        return f;
    }

    /** A bias, or {@code null} when there is none. */
    Var optional(String name) {
        return source.contains(name) ? read(name) : null;
    }

    /** A {@code torch.nn.Linear} weight as {@code [in, out]}. */
    Var linear(String name) {
        Shape s = source.shape(name);
        float[] w = floats(name);
        int out = s.dim(0);
        int in = s.dim(1);
        float[] t = new float[w.length];
        for (int o = 0; o < out; o++) {
            for (int i = 0; i < in; i++) {
                t[i * out + o] = w[o * in + i];
            }
        }
        return keep(Tensor.of(Shape.of(in, out), t, device).to(dtype));
    }

    /**
     * A convolution's weight, folding weight normalization ({@code g * v / |v|} over every
     * axis but the first) in either of PyTorch's spellings.
     */
    Var conv(String name) {
        if (source.contains(name + ".weight")) {
            return read(name + ".weight");
        }
        String g = source.contains(name + ".weight_g") ? name + ".weight_g" : name + ".parametrizations.weight.original0";
        String v = source.contains(name + ".weight_v") ? name + ".weight_v" : name + ".parametrizations.weight.original1";
        if (!source.contains(v)) {
            throw new IllegalArgumentException("the weights have no " + name + ".weight");
        }
        Shape shape = source.shape(v);
        float[] gv = floats(g);
        float[] vv = floats(v);
        int rows = shape.dim(0);
        int per = vv.length / rows;
        for (int o = 0; o < rows; o++) {
            double sum = 0;
            for (int j = 0; j < per; j++) {
                sum += (double) vv[o * per + j] * vv[o * per + j];
            }
            float s = (float) (gv[o] / Math.sqrt(sum));
            for (int j = 0; j < per; j++) {
                vv[o * per + j] *= s;
            }
        }
        return keep(Tensor.of(shape, vv, device).to(dtype));
    }

    /** A one-layer (optionally bidirectional) {@code torch.nn.LSTM}, batch first. */
    Recurrent lstm(String name) {
        return lstm(name, 0);
    }

    /** One layer of a (optionally bidirectional) {@code torch.nn.LSTM}, batch first. */
    Recurrent lstm(String name, int layer) {
        String l = "_l" + layer;
        boolean bi = source.contains(name + ".weight_ih" + l + "_reverse");
        int count = bi ? 2 : 1;
        Var[] wx = new Var[count];
        Var[] wh = new Var[count];
        Var[] b = new Var[count];
        int units = source.shape(name + ".weight_hh" + l).dim(1);
        int inputs = source.shape(name + ".weight_ih" + l).dim(1);
        for (int d = 0; d < count; d++) {
            String suffix = d == 0 ? l : l + "_reverse";
            wx[d] = linear(name + ".weight_ih" + suffix);
            wh[d] = linear(name + ".weight_hh" + suffix);
            float[] bi1 = floats(name + ".bias_ih" + suffix);
            float[] bh = floats(name + ".bias_hh" + suffix);
            for (int i = 0; i < bi1.length; i++) {
                bi1[i] += bh[i];
            }
            b[d] = keep(Tensor.of(Shape.of(bi1.length), bi1, device).to(dtype));
        }
        return Recurrent.of(new Recurrent.Settings(Recurrent.Cell.LSTM, inputs, units, true, false,
                bi ? Recurrent.Merge.CONCAT : null, ActivationOp.TANH, ActivationOp.SIGMOID, false, true, false),
                wx, wh, b);
    }

    /** A StyleTTS2 {@code AdaIN1d} or {@code AdaLayerNorm} (its {@code fc} projects the style). */
    StyleNorm styleNorm(String name, StyleNorm.Kind kind) {
        return new StyleNorm(kind, 1e-5f, linear(name + ".fc.weight"), read(name + ".fc.bias"));
    }

    private Var keep(Tensor t) {
        Tensor.evaluate(List.of(t));
        owned.add(t);
        return Var.of(t);
    }

    @Override
    public void close() {
        for (Tensor t : owned) {
            t.release();
        }
        owned.clear();
    }
}
