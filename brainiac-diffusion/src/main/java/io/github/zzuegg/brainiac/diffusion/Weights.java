package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Reads one model's tensors from a weight file onto a device, under a name prefix, and keeps
 * every tensor it made so that the forward passes never release them.
 */
final class Weights {

    private final WeightSource source;
    private final String prefix;
    private final UnaryOperator<String> rename;
    final Device device;
    final DType dtype;
    final List<Tensor> all;

    Weights(WeightSource source, String prefix, Device device, DType dtype) {
        this(source, prefix, UnaryOperator.identity(), device, dtype, new ArrayList<>());
    }

    Weights(WeightSource source, String prefix, Device device, DType dtype, List<Tensor> all) {
        this(source, prefix, UnaryOperator.identity(), device, dtype, all);
    }

    private Weights(WeightSource source, String prefix, UnaryOperator<String> rename, Device device, DType dtype,
            List<Tensor> all) {
        this.source = source;
        this.prefix = prefix;
        this.rename = rename;
        this.device = device;
        this.dtype = dtype;
        this.all = all;
    }

    /** The same file seen under {@code prefix + more}. */
    Weights at(String more) {
        return new Weights(source, prefix + more, rename, device, dtype, all);
    }

    /** The same file with names translated before they are looked up (for other checkpoint layouts). */
    Weights renamed(UnaryOperator<String> names) {
        return new Weights(source, prefix, names, device, dtype, all);
    }

    /** The same file read in another type, sharing the list of tensors. */
    Weights as(DType type) {
        return new Weights(source, prefix, rename, device, type, all);
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
        return keep(source.tensor(n, device, dtype));
    }

    Tensor keep(Tensor t) {
        all.add(t);
        return t;
    }

    /** Replaces a tensor made from weights by another one built from it, freeing the first. */
    Tensor replace(Tensor old, Tensor made) {
        Tensor.evaluate(List.of(made));
        all.remove(old);
        old.release();
        return keep(made);
    }

    /** {@code name.weight} {@code [out, in]} as {@code [in, out]} and {@code name.bias} if present. */
    Nn.Linear linear(String name) {
        Tensor w = get(name + ".weight");
        // a 1x1 convolution used as a projection is the same matrix; the reshape shares memory
        Tensor matrix = w.rank() == 4 ? w.reshape(w.shape().dim(0), w.shape().dim(1)) : w;
        Tensor t = replace(w, matrix.transpose());
        Tensor b = has(name + ".bias") ? get(name + ".bias") : null;
        return new Nn.Linear(t, b);
    }

    /** Rows {@code [start, end)} of a fused {@code [out, in]} projection (and its bias), as {@code [in, end - start]}. */
    Nn.Linear linearRows(String weight, String bias, int start, int end) {
        Tensor w = get(weight);
        Tensor t = keep(w.slice(0, start, end).transpose());
        Tensor b = bias == null ? null : get(bias);
        Tensor bt = b == null ? null : keep(b.slice(0, start, end));
        Tensor.evaluate(bt == null ? List.of(t) : List.of(t, bt));
        all.remove(w);
        w.release();
        if (b != null) {
            all.remove(b);
            b.release();
        }
        return new Nn.Linear(t, bt);
    }

    /** {@code name.weight} {@code [out, in]} split in two halves of the output, as GEGLU needs. */
    Nn.Linear[] linearHalves(String name) {
        Tensor w = get(name + ".weight");
        Tensor b = get(name + ".bias");
        int out = w.shape().dim(0) / 2;
        Nn.Linear[] halves = new Nn.Linear[2];
        for (int i = 0; i < 2; i++) {
            Tensor wi = keep(w.slice(0, i * out, (i + 1) * out).transpose());
            Tensor bi = keep(b.slice(0, i * out, (i + 1) * out));
            Tensor.evaluate(List.of(wi, bi));
            halves[i] = new Nn.Linear(wi, bi);
        }
        all.remove(w);
        all.remove(b);
        w.release();
        b.release();
        return halves;
    }

    /** A convolution {@code [out, in, k, k]} with its bias. */
    Nn.Conv2d conv(String name, int stride, int pad) {
        return conv(name, stride, pad, pad);
    }

    /** A convolution with different padding before and after the image. */
    Nn.Conv2d conv(String name, int stride, int pad, int padEnd) {
        Tensor w = get(name + ".weight");
        int out = w.shape().dim(0);
        int in = w.shape().dim(1);
        int k = w.shape().dim(2);
        Tensor cols = replace(w, w.reshape(out, in * k * k).transpose());
        Tensor b = has(name + ".bias") ? get(name + ".bias").reshape(out, 1, 1) : null;
        return new Nn.Conv2d(cols, b, in, out, k, stride, pad, padEnd);
    }

    /** A convolution with stride 1 that keeps the image size. */
    Nn.Conv2d conv(String name) {
        return conv(name, 1, shape(name + ".weight").dim(2) / 2);
    }

    /** A group norm's weight and bias. */
    Nn.GroupNorm groupNorm(String name, int groups, float eps) {
        return new Nn.GroupNorm(as(DType.FLOAT32).get(name + ".weight"), as(DType.FLOAT32).get(name + ".bias"),
                groups, eps);
    }

    /** A layer norm's weight and bias. */
    Nn.LayerNorm layerNorm(String name, float eps) {
        return new Nn.LayerNorm(as(DType.FLOAT32).get(name + ".weight"), as(DType.FLOAT32).get(name + ".bias"), eps);
    }
}
