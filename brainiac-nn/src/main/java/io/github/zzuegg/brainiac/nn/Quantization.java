package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * Stores weights as small whole numbers times a scale: symmetric linear quantization.
 *
 * <p>Each weight matrix is cut into groups, and every group gets one float scale, so that
 * its largest magnitude maps to the largest level: {@code w ≈ q * scale} with {@code q} a
 * whole number in {@code [-(2^(bits-1) - 1), 2^(bits-1) - 1]} (-127..127 for 8 bits,
 * -7..7 for 4). A group is a whole output channel by default ({@link #perChannel()}), one
 * run of {@code size} consecutive inputs of a channel ({@link #groups(int)}), or the whole
 * tensor ({@link #perTensor()}). Smaller groups follow the weights more closely and cost
 * one scale each.
 *
 * <pre>{@code
 * model.quantize(Quantization.int8());              // post-training, per channel
 * model.save(path);                                 // weights stored in 8 bits: ~4x smaller
 *
 * Trainer.of(model).quantizationAware(Quantization.int4().groups(32)).fit(data);
 * }</pre>
 *
 * <p>Quantized are the weight matrices and filters (every parameter with two or more
 * dimensions: dense, convolution, attention, recurrent and embedding weights); biases,
 * normalization parameters and the parameters of activations stay in floating point. The
 * channel is the output unit for dense-like {@code [inputs, units]} weights (the last
 * axis), the filter for convolutions and the token for embeddings (the first axis).
 *
 * <p>Quantizing a model rounds its weights in place, so it runs as before on any device
 * and backend, and saving it stores the weights in {@code bits} bits each plus the scales.
 */
public final class Quantization {

    private static final int PER_CHANNEL = 0;
    private static final int PER_TENSOR = -1;

    private final int bits;
    private final int groupSize;

    private Quantization(int bits, int groupSize) {
        this.bits = bits;
        this.groupSize = groupSize;
    }

    /**
     * Quantizes to a number of bits per weight, per output channel.
     *
     * @param bits 2 to 8
     * @return the quantization
     */
    public static Quantization bits(int bits) {
        if (bits < 2 || bits > 8) {
            throw new IllegalArgumentException("bits must be 2 to 8, got " + bits);
        }
        return new Quantization(bits, PER_CHANNEL);
    }

    /** @return 8 bits per weight (levels -127..127), per output channel */
    public static Quantization int8() {
        return bits(8);
    }

    /** @return 4 bits per weight (levels -7..7), per output channel */
    public static Quantization int4() {
        return bits(4);
    }

    /** @return the same bits with one scale per output channel (the default) */
    public Quantization perChannel() {
        return new Quantization(bits, PER_CHANNEL);
    }

    /** @return the same bits with one scale for each whole tensor */
    public Quantization perTensor() {
        return new Quantization(bits, PER_TENSOR);
    }

    /**
     * Uses one scale for every {@code size} consecutive inputs of an output channel, as
     * low-bit formats for language models do. A tensor whose channels do not split into
     * whole groups of that size falls back to one scale per channel.
     *
     * @param size the inputs per group, at least 2
     * @return the same bits with grouped scales
     */
    public Quantization groups(int size) {
        if (size < 2) {
            throw new IllegalArgumentException("group size must be at least 2, got " + size);
        }
        return new Quantization(bits, size);
    }

    /** @return the bits per weight */
    public int bits() {
        return bits;
    }

    /** @return the largest level, {@code 2^(bits-1) - 1} */
    public int levels() {
        return (1 << (bits - 1)) - 1;
    }

    /** @return the inputs per group; 0 for one scale per channel, -1 for one per tensor */
    public int groupSize() {
        return groupSize;
    }

    /**
     * A weight that quantization applies to.
     *
     * @param weight the parameter
     * @param axis its channel axis: {@code 0}, or {@code rank - 1}
     */
    public record Weight(Var weight, int axis) {
    }

    /**
     * Lists the weights of a model this quantization applies to, each once even when
     * layers share it.
     *
     * @param model the model
     * @return the weights, in model order
     */
    public static List<Weight> weights(Model model) {
        Set<Var> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Weight> out = new ArrayList<>();
        for (Layer layer : model.layers()) {
            if (layer instanceof Activation) {
                continue;
            }
            Set<Var> skip = Collections.newSetFromMap(new IdentityHashMap<>());
            if (layer instanceof Dense d) {
                skip.addAll(d.activation().parameters());
            } else if (layer instanceof Conv c) {
                skip.addAll(c.activation().parameters());
            }
            boolean firstAxis = layer instanceof Conv || layer instanceof Embedding;
            for (Var p : layer.parameters()) {
                int rank = p.shape().rank();
                if (rank >= 2 && !skip.contains(p) && seen.add(p)) {
                    out.add(new Weight(p, firstAxis ? 0 : rank - 1));
                }
            }
        }
        return out;
    }

    /**
     * Rounds a tensor to this quantization's grid, on its own device and in its own number
     * format: {@code round(w / scale) * scale} with one scale per group.
     *
     * @param value the weights
     * @param axis the channel axis, {@code 0} or the last
     * @return the rounded weights, same shape and type
     */
    public Tensor quantize(Tensor value, int axis) {
        Layout layout = layout(value.shape(), axis);
        Tensor w = layout.grouped(value);
        Tensor max = layout.perTensor ? w.abs().max() : w.abs().max(layout.channelFirst ? 2 : 1, true);
        float tiny = value.dtype() == DType.FLOAT16 ? 6.103515625e-5f : Float.MIN_NORMAL;
        Tensor scale = max.maximum(Tensor.full(Shape.of(1), tiny, value.dtype(), value.device())).div(levels());
        return w.div(scale).round().mul(scale).reshape(value.shape());
    }

    // ---------------------------------------------------------------- packed storage

    /**
     * Weights packed into whole numbers and scales, as model files store them.
     *
     * @param scales one per group
     * @param data the levels, {@code bits} bits each in two's complement, lowest bits first
     */
    record Packed(float[] scales, byte[] data) {
    }

    /** Packs weights (row-major) of a shape, computing scales as {@link #quantize} does. */
    Packed pack(float[] values, Shape shape, int axis) {
        Layout layout = layout(shape, axis);
        float[] max = new float[layout.scaleCount()];
        for (int i = 0; i < values.length; i++) {
            int g = layout.group(i);
            max[g] = Math.max(max[g], Math.abs(values[i]));
        }
        float[] scales = new float[max.length];
        for (int g = 0; g < max.length; g++) {
            scales[g] = Math.max(max[g], Float.MIN_NORMAL) / levels();
        }
        int limit = levels();
        int mask = (1 << bits) - 1;
        byte[] data = new byte[(int) (((long) values.length * bits + 7) / 8)];
        long bit = 0;
        for (int i = 0; i < values.length; i++) {
            int q = (int) Math.rint(values[i] / scales[layout.group(i)]);
            q = Math.max(-limit, Math.min(limit, q)) & mask;
            for (int b = 0; b < bits; b++, bit++) {
                if ((q >>> b & 1) != 0) {
                    data[(int) (bit >>> 3)] |= (byte) (1 << (bit & 7));
                }
            }
        }
        return new Packed(scales, data);
    }

    /** Unpacks weights: {@code q * scale} per value. */
    float[] unpack(Packed packed, Shape shape, int axis) {
        Layout layout = layout(shape, axis);
        int n = Math.toIntExact(shape.size());
        float[] out = new float[n];
        long bit = 0;
        for (int i = 0; i < n; i++) {
            int q = 0;
            for (int b = 0; b < bits; b++, bit++) {
                q |= (packed.data[(int) (bit >>> 3)] >>> (bit & 7) & 1) << b;
            }
            q = q << (32 - bits) >> (32 - bits);   // sign-extend
            out[i] = q * packed.scales[layout.group(i)];
        }
        return out;
    }

    /** @return the number of scales for weights of a shape */
    int scaleCount(Shape shape, int axis) {
        return layout(shape, axis).scaleCount();
    }

    /** @return the bytes the levels of {@code count} weights take */
    int dataBytes(long count) {
        return Math.toIntExact((count * bits + 7) / 8);
    }

    // ---------------------------------------------------------------- groups

    private Layout layout(Shape shape, int axis) {
        int rank = shape.rank();
        if (axis != 0 && axis != rank - 1) {
            throw new IllegalArgumentException("channel axis must be 0 or " + (rank - 1) + ", got " + axis);
        }
        long size = shape.size();
        if (groupSize == PER_TENSOR || rank < 2) {
            return new Layout(true, axis == 0, 1, Math.toIntExact(size), Math.toIntExact(size), shape);
        }
        int channels = shape.dim(axis);
        int rest = Math.toIntExact(size / channels);
        int g = groupSize > 0 && groupSize < rest && rest % groupSize == 0 ? groupSize : rest;
        return new Layout(false, axis == 0, channels, rest, g, shape);
    }

    /**
     * How a tensor splits into groups: viewed as {@code [channels, rest]} (channel axis
     * first) or {@code [rest, channels]} (channel axis last), with {@code rest} cut into
     * runs of {@code group}.
     */
    private record Layout(boolean perTensor, boolean channelFirst, int channels, int rest, int group, Shape shape) {

        int scaleCount() {
            return perTensor ? 1 : channels * (rest / group);
        }

        /** @return the scale index of the value at a row-major position */
        int group(int index) {
            if (perTensor) {
                return 0;
            }
            int perChannel = rest / group;
            if (channelFirst) {
                return index / rest * perChannel + index % rest / group;
            }
            return index / channels / group * channels + index % channels;
        }

        /** The tensor viewed with the group as its own axis. */
        Tensor grouped(Tensor t) {
            if (perTensor) {
                return t;
            }
            return channelFirst
                    ? t.reshape(channels, rest / group, group)
                    : t.reshape(rest / group, group, channels);
        }
    }

    @Override
    public String toString() {
        String scales = groupSize == PER_TENSOR ? "per tensor" : groupSize == PER_CHANNEL ? "per channel"
                : "groups of " + groupSize;
        return "int" + bits + " (" + scales + ")";
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Quantization q && q.bits == bits && q.groupSize == groupSize;
    }

    @Override
    public int hashCode() {
        return bits * 31 + groupSize;
    }
}
