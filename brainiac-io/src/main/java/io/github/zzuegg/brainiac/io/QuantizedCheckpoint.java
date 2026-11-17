package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hugging Face checkpoints quantized with GPTQ (AutoGPTQ, GPTQModel, optimum) or AWQ
 * (AutoAWQ's GEMM layout): each quantized linear layer's {@code qweight}, {@code qzeros},
 * {@code scales} and, for GPTQ, {@code g_idx} appear as one float {@code weight}
 * {@code [out, in]}, dequantized when read, so loaders see an ordinary checkpoint.
 * {@link WeightSource#open} applies it when {@code config.json} has a
 * {@code quantization_config} with {@code quant_method} {@code "gptq"} or {@code "awq"}.
 */
final class QuantizedCheckpoint implements WeightSource {

    private static final int[] AWQ_ORDER = {0, 2, 4, 6, 1, 3, 5, 7};

    private final WeightSource source;
    private final boolean awq;
    private final int bits;
    private final int groupSize;
    /** GPTQ's v1 checkpoints store zeros minus one. */
    private final boolean zeroPlusOne;
    /** Each synthesized weight's layer prefix, such as {@code model.layers.0.mlp.up_proj.}. */
    private final Map<String, String> layers = new LinkedHashMap<>();
    private final List<String> names = new ArrayList<>();

    QuantizedCheckpoint(WeightSource source, Map<String, Object> config) {
        this.source = source;
        String method = Json.string(config, "quant_method", "");
        awq = method.equals("awq");
        if (!awq && !method.equals("gptq")) {
            throw new IllegalArgumentException("quant_method " + method + " is not supported");
        }
        bits = (int) Json.integer(config, awq ? "w_bit" : "bits", awq ? Json.integer(config, "bits", 4) : 4);
        if (awq ? bits != 4 : bits != 2 && bits != 4 && bits != 8) {
            throw new IllegalArgumentException((awq ? "AWQ" : "GPTQ") + " with " + bits + " bits is not supported");
        }
        groupSize = (int) Json.integer(config, awq ? "q_group_size" : "group_size",
                Json.integer(config, "group_size", 128));
        String format = Json.string(config, "checkpoint_format", "gptq");
        zeroPlusOne = !awq && !format.equals("gptq_v2");
        if (awq && !Json.string(config, "version", "gemm").equalsIgnoreCase("gemm")) {
            throw new IllegalArgumentException("only AWQ's GEMM layout is supported, not " + config.get("version"));
        }
        for (String n : source.names()) {
            if (n.endsWith(".qweight")) {
                String prefix = n.substring(0, n.length() - "qweight".length());
                layers.put(prefix + "weight", prefix);
                names.add(prefix + "weight");
            } else if (!n.endsWith(".qzeros") && !n.endsWith(".scales") && !n.endsWith(".g_idx")) {
                names.add(n);
            }
        }
    }

    @Override
    public List<String> names() {
        return names;
    }

    @Override
    public boolean contains(String name) {
        return layers.containsKey(name) || source.contains(name) && names.contains(name);
    }

    @Override
    public Shape shape(String name) {
        String p = layers.get(name);
        if (p == null) {
            return source.shape(name);
        }
        Shape s = source.shape(p + "scales");   // [groups, out]
        Shape q = source.shape(p + "qweight");
        int in = awq ? q.dim(0) : q.dim(0) * 32 / bits;
        return Shape.of(s.dim(1), in);
    }

    @Override
    public String format(String name) {
        return layers.containsKey(name) ? source.format(layers.get(name) + "scales") : source.format(name);
    }

    @Override
    public DType naturalType(String name) {
        return layers.containsKey(name) ? source.naturalType(layers.get(name) + "scales") : source.naturalType(name);
    }

    @Override
    public Tensor tensor(String name, Device device, DType dtype) {
        String p = layers.get(name);
        if (p == null) {
            return source.tensor(name, device, dtype);
        }
        Device cpu = Device.cpu();
        int[] qweight = ints(source.tensor(p + "qweight", cpu, DType.FLOAT64));
        int[] qzeros = ints(source.tensor(p + "qzeros", cpu, DType.FLOAT64));
        float[] scales = source.tensor(p + "scales", cpu, DType.FLOAT32).toArray();
        Shape shape = shape(name);
        int out = shape.dim(0);
        int in = shape.dim(1);
        int[] gIdx = !awq && source.contains(p + "g_idx") ? ints(source.tensor(p + "g_idx", cpu, DType.FLOAT64)) : null;
        int per = 32 / bits;
        int mask = (1 << bits) - 1;
        float[] w = new float[out * in];
        for (int i = 0; i < in; i++) {
            int g = gIdx != null ? gIdx[i] : i / groupSize;
            for (int o = 0; o < out; o++) {
                int q;
                int z;
                if (awq) {
                    q = nibble(qweight[i * (out / 8) + o / 8], o % 8);
                    z = nibble(qzeros[g * (out / 8) + o / 8], o % 8);
                } else {
                    q = qweight[(i / per) * out + o] >>> (i % per) * bits & mask;
                    z = (qzeros[g * (out / per) + o / per] >>> (o % per) * bits & mask) + (zeroPlusOne ? 1 : 0);
                }
                w[o * in + i] = scales[g * out + o] * (q - z);
            }
        }
        return Tensor.of(shape, w, device).to(dtype);
    }

    /** The 4-bit value of output {@code k} (of 8) in an AWQ-packed int. */
    private static int nibble(int packed, int k) {
        for (int i = 0; i < 8; i++) {
            if (AWQ_ORDER[i] == k) {
                return packed >>> 4 * i & 0xF;
            }
        }
        throw new IllegalStateException();
    }

    private static int[] ints(Tensor t) {
        double[] d = t.toDoubleArray();
        int[] out = new int[d.length];
        for (int i = 0; i < d.length; i++) {
            out[i] = (int) (long) d[i];
        }
        return out;
    }

    @Override
    public void close() {
        source.close();
    }

    @Override
    public String toString() {
        return (awq ? "AWQ" : "GPTQ") + " " + bits + "-bit " + source;
    }
}
