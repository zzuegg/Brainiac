package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Named weight tensors in files: one safetensors file, a model split into shards, or any
 * of those with its names mapped to another scheme.
 *
 * <pre>{@code
 * try (WeightSource w = WeightSource.open(Path.of("Qwen2.5-0.5B"))) {   // dir, index or file
 *     Tensor embed = w.tensor("model.embed_tokens.weight", Device.gpu(), DType.FLOAT16);
 * }
 * }</pre>
 *
 * <p>Values are read from memory-mapped files, so opening even a 40 GB model takes
 * almost no memory; each tensor is copied to its device only when asked for.
 */
public interface WeightSource extends AutoCloseable {

    /**
     * Returns the names of all tensors, in file order.
     *
     * @return the names
     */
    List<String> names();

    /**
     * Returns whether a tensor of this name exists.
     *
     * @param name the name
     * @return {@code true} if it does
     */
    default boolean contains(String name) {
        return names().contains(name);
    }

    /**
     * Returns a tensor's shape without reading it.
     *
     * @param name the name
     * @return its shape
     * @throws IllegalArgumentException if there is no such tensor
     */
    Shape shape(String name);

    /**
     * Returns the format a tensor is stored in, as the file names it ({@code "F32"},
     * {@code "BF16"}, {@code "Q4_K"} and so on).
     *
     * @param name the name
     * @return the stored format
     */
    String format(String name);

    /**
     * Returns the floating-point type closest to how a tensor is stored: the type itself
     * for float64, float32, float16 and bfloat16, otherwise float32.
     *
     * @param name the name
     * @return the type {@link #tensor(String, Device)} creates
     */
    DType naturalType(String name);

    /**
     * Reads a tensor onto a device in a given type. Values are converted on the device
     * when the stored type is one it supports.
     *
     * @param name the name
     * @param device where to put it
     * @param dtype its type
     * @return the tensor
     */
    Tensor tensor(String name, Device device, DType dtype);

    /**
     * Reads a tensor onto a device in its {@linkplain #naturalType natural type}.
     *
     * @param name the name
     * @param device where to put it
     * @return the tensor
     */
    default Tensor tensor(String name, Device device) {
        return tensor(name, device, naturalType(name));
    }

    /**
     * Reads every tensor onto a device in one type.
     *
     * @param device where to put them
     * @param dtype their type
     * @return the tensors by name, in file order
     */
    default Map<String, Tensor> loadAll(Device device, DType dtype) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (String n : names()) {
            out.put(n, tensor(n, device, dtype));
        }
        return out;
    }

    /**
     * Returns a view of these weights with every name passed through {@code rename}.
     * Names it maps to {@code null} are left out, so a mapping can also drop tensors.
     *
     * @param rename from a name in the files to the name to use
     * @return the renamed view; closing it closes this source
     */
    default WeightSource renamed(UnaryOperator<String> rename) {
        return Sources.renamed(this, rename);
    }

    /**
     * Closes the files. Tensors read before stay valid.
     */
    @Override
    void close();

    /**
     * Opens weights at a path: a {@code .safetensors} file, a sharded model's {@code
     * *.safetensors.index.json}, or a directory holding either (a Hugging Face model
     * download, with safetensors or a sharded or single {@code pytorch_model.bin}), a {@code
     * .gguf} file, or a PyTorch checkpoint ({@code .bin} read by {@link PyTorchCheckpoint};
     * {@code .pt}, {@code .pth} and {@code .ckpt}, which may nest dictionaries or hold a bare
     * tensor, by {@link TorchCheckpoint}). A folder whose {@code config.json} says it was
     * quantized with GPTQ or AWQ reads its layers as float weights, dequantized.
     *
     * @param path the file or directory
     * @return the weights
     * @throws UncheckedIOException if the files cannot be read
     */
    static WeightSource open(Path path) {
        Objects.requireNonNull(path, "path");
        WeightSource plain = openPlain(path);
        Path config = Files.isDirectory(path) ? path.resolve("config.json") : null;
        if (config != null && Files.isRegularFile(config)) {
            try {
                if (Json.object(Json.read(config)).get("quantization_config") instanceof Map<?, ?> q
                        && ("gptq".equals(q.get("quant_method")) || "awq".equals(q.get("quant_method")))) {
                    return new QuantizedCheckpoint(plain, Json.object(q));
                }
            } catch (IOException | RuntimeException e) {
                plain.close();
                throw e instanceof IOException io ? new UncheckedIOException(io) : (RuntimeException) e;
            }
        }
        return plain;
    }

    private static WeightSource openPlain(Path path) {
        try {
            if (Files.isDirectory(path)) {
                Path index = path.resolve("model.safetensors.index.json");
                if (Files.isRegularFile(index)) {
                    return open(index);
                }
                Path single = path.resolve("model.safetensors");
                if (Files.isRegularFile(single)) {
                    return SafeTensors.open(single);
                }
                List<Path> files;
                try (var list = Files.list(path)) {
                    files = list.filter(p -> p.getFileName().toString().endsWith(".safetensors")).sorted().toList();
                }
                if (files.isEmpty()) {
                    Path binIndex = path.resolve("pytorch_model.bin.index.json");
                    if (Files.isRegularFile(binIndex)) {
                        return open(binIndex);
                    }
                    Path bin = path.resolve("pytorch_model.bin");
                    if (Files.isRegularFile(bin)) {
                        return PyTorchCheckpoint.open(bin);
                    }
                    throw new IllegalArgumentException("no .safetensors or pytorch_model.bin files in " + path);
                }
                return of(files.stream().map(SafeTensors::open).map(WeightSource.class::cast).toList());
            }
            String file = path.getFileName().toString();
            if (file.endsWith(".index.json")) {
                Map<String, Object> index = Json.object(Json.read(path));
                Map<String, Object> map = Json.object(index.get("weight_map"));
                List<String> shards = new ArrayList<>();
                for (Object v : map.values()) {
                    if (!shards.contains((String) v)) {
                        shards.add((String) v);
                    }
                }
                Path dir = path.toAbsolutePath().getParent();
                List<WeightSource> opened = new ArrayList<>();
                for (String s : shards) {
                    opened.add(s.endsWith(".bin") ? PyTorchCheckpoint.open(dir.resolve(s)) : SafeTensors.open(dir.resolve(s)));
                }
                return of(opened);
            }
            if (file.endsWith(".gguf")) {
                return Gguf.open(path);
            }
            if (file.endsWith(".bin")) {
                return PyTorchCheckpoint.open(path);
            }
            if (file.endsWith(".pt") || file.endsWith(".pth") || file.endsWith(".ckpt")) {
                return TorchCheckpoint.open(path);
            }
            return SafeTensors.open(path);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Joins several sources, such as the shards of one model, into one.
     *
     * @param parts the sources; their names must not overlap
     * @return the joined source; closing it closes every part
     */
    static WeightSource of(List<? extends WeightSource> parts) {
        return Sources.joined(List.copyOf(parts));
    }
}
