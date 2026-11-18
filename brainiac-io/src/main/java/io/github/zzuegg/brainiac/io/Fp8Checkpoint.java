package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Hugging Face checkpoints stored in 8-bit floats ({@code quant_method "fp8"}, as the
 * {@code -FP8} releases of Qwen3 and DeepSeek): each linear layer's {@code weight} is
 * {@code F8_E4M3} with a {@code weight_scale_inv} that multiplies it back, either one
 * value per {@code weight_block_size} block (fine-grained, usually 128 x 128) or one per
 * tensor or output row. Weights are dequantized when read, so loaders see an ordinary
 * checkpoint; activation scales ({@code input_scale}) are dropped. {@link WeightSource#open}
 * applies it when {@code config.json} says so.
 */
final class Fp8Checkpoint implements WeightSource {

    private final WeightSource source;
    /** rows and columns per scale block, or {@code null} without blocks */
    private final int[] block;
    /** the weights that carry a scale, mapped to the scale's name */
    private final Map<String, String> scaled = new java.util.HashMap<>();
    private final List<String> names = new ArrayList<>();
    private final Set<String> dropped = new TreeSet<>();

    Fp8Checkpoint(WeightSource source, Map<String, Object> config) {
        this.source = source;
        String method = Json.string(config, "quant_method", "");
        if (!method.equals("fp8")) {
            throw new IllegalArgumentException("quant_method " + method + " is not fp8");
        }
        if (config.get("weight_block_size") instanceof List<?> b && b.size() == 2) {
            block = new int[] {((Number) b.get(0)).intValue(), ((Number) b.get(1)).intValue()};
        } else {
            block = null;
        }
        Set<String> all = new java.util.HashSet<>(source.names());
        for (String n : source.names()) {
            if (n.endsWith(".weight_scale_inv") || n.endsWith(".weight_scale") || n.endsWith(".input_scale")) {
                dropped.add(n);
                String w = n.substring(0, n.lastIndexOf('.')) + ".weight";
                if (!n.endsWith(".input_scale") && all.contains(w)) {
                    scaled.put(w, n);
                }
            } else {
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
        return source.contains(name) && !dropped.contains(name);
    }

    @Override
    public Shape shape(String name) {
        return source.shape(name);
    }

    @Override
    public String format(String name) {
        return source.format(name);
    }

    @Override
    public DType naturalType(String name) {
        String s = scaled.get(name);
        if (s == null) {
            return source.naturalType(name);
        }
        DType t = source.naturalType(s);
        // the scales are float32 in most releases: the layer runs in bf16 like the rest
        return t == DType.FLOAT32 ? DType.BFLOAT16 : t;
    }

    @Override
    public Tensor tensor(String name, Device device, DType dtype) {
        String s = scaled.get(name);
        if (s == null) {
            return source.tensor(name, device, dtype);
        }
        Device cpu = Device.cpu();
        Shape shape = source.shape(name);
        if (shape.rank() != 2) {
            throw new IllegalArgumentException(name + " is " + shape + "; fp8 weights are matrices");
        }
        int rows = shape.dim(0);
        int cols = shape.dim(1);
        float[] w = source.tensor(name, cpu, DType.FLOAT32).toArray();
        Shape ss = source.shape(s);
        float[] scale = source.tensor(s, cpu, DType.FLOAT32).toArray();
        if (scale.length == 1) {
            for (int i = 0; i < w.length; i++) {
                w[i] *= scale[0];
            }
        } else if (scale.length == rows && (ss.rank() == 1 || ss.rank() == 2 && ss.dim(1) == 1)) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    w[r * cols + c] *= scale[r];
                }
            }
        } else {
            int bm = block != null ? block[0] : Math.ceilDiv(rows, ss.dim(0));
            int bn = block != null ? block[1] : Math.ceilDiv(cols, ss.dim(1));
            int blocksAcross = Math.ceilDiv(cols, bn);
            if (scale.length != Math.ceilDiv(rows, bm) * blocksAcross) {
                throw new IllegalArgumentException(s + " is " + ss + ", which does not tile " + shape + " in "
                        + bm + " x " + bn + " blocks");
            }
            for (int r = 0; r < rows; r++) {
                int base = r / bm * blocksAcross;
                for (int c = 0; c < cols; c++) {
                    w[r * cols + c] *= scale[base + c / bn];
                }
            }
        }
        return Tensor.of(shape, w, device).to(dtype);
    }

    @Override
    public void close() {
        source.close();
    }

    @Override
    public String toString() {
        return "FP8" + (block != null ? " " + block[0] + "x" + block[1] + "-block" : "") + " " + source;
    }
}
