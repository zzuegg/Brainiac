package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.OptionalLong;
import java.util.stream.Stream;

/**
 * Estimates from the weight files' headers how much device memory a model takes, before
 * loading it, so that loading can pick a precision that fits or fail with a clear message.
 */
final class MemoryFit {

    /** Room left for activations, the text encoders' outputs and the decoder, at the least. */
    static final long MIN_WORKING = 1L << 30;

    private MemoryFit() {
    }

    /**
     * The parameters of one component.
     *
     * @param matrices values in rank-2 projection weights, which quantize
     * @param other everything else: embeddings, norms, biases, convolutions
     */
    record Params(long matrices, long other) {

        static final Params NONE = new Params(0, 0);

        long total() {
            return matrices + other;
        }

        /**
         * @param dtype the load type
         * @param bits 0 for the load type, else the bits per quantized weight
         * @return the device bytes
         */
        long bytes(DType dtype, int bits) {
            if (bits == 0) {
                return total() * dtype.bytes();
            }
            // a byte (or half a byte) per level and a float scale per group of rows, as Weights packs them
            double perValue = (bits <= 4 ? 0.5 : 1) + 4.0 / (bits <= 4 ? 32 : 64);
            return (long) (matrices * perValue) + other * dtype.bytes();
        }
    }

    /**
     * Picks a diffusers component's weight file: the plain one, a sharded index, optionally
     * an {@code fp16} variant, else the first {@code .safetensors}.
     */
    static Path weightFile(Path dir, String base, boolean variants) throws IOException {
        String[] names = variants
                ? new String[] {base + ".safetensors", base + ".safetensors.index.json", base + ".fp16.safetensors",
                    base + ".fp16.safetensors.index.json"}
                : new String[] {base + ".safetensors", base + ".safetensors.index.json"};
        for (String name : names) {
            if (Files.isRegularFile(dir.resolve(name))) {
                return dir.resolve(name);
            }
        }
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(p -> p.getFileName().toString().endsWith(".safetensors")).sorted().findFirst()
                    .orElseThrow(() -> new IOException("no .safetensors weights in " + dir
                            + " (PyTorch .bin files are not supported)"));
        }
    }

    /** Counts a component's parameters from its headers, or none if the folder is missing. */
    static Params params(Path dir, String base, boolean variants) throws IOException {
        return Files.isDirectory(dir) ? params(weightFile(dir, base, variants)) : Params.NONE;
    }

    /** Counts the parameters of a weight file from its header. */
    static Params params(Path file) {
        long matrices = 0;
        long other = 0;
        try (WeightSource source = WeightSource.open(file)) {
            for (String name : source.names()) {
                long n = source.shape(name).size();
                String lower = name.toLowerCase(Locale.ROOT);
                boolean embedding = lower.contains("embed_tokens") || lower.contains("shared.")
                        || lower.contains("token_embedding") || lower.contains("position_embedding");
                if (source.shape(name).rank() == 2 && lower.endsWith("weight") && !embedding) {
                    matrices += n;
                } else {
                    other += n;
                }
            }
        }
        return new Params(matrices, other);
    }

    /** @return the bytes the device's backend may still take, or empty if it does not count them (the CPU) */
    static OptionalLong free(Device device) {
        return device.memory().map(m -> OptionalLong.of(Math.max(0, m.budget() - m.inUse())))
                .orElse(OptionalLong.empty());
    }

    /**
     * Fails if the weights alone leave less than {@link #MIN_WORKING} of the device's memory.
     *
     * @param what the model, for the message
     * @param weights the weights' bytes
     * @param device the device
     * @param hint what could help, for the message
     */
    static void require(String what, long weights, Device device, String hint) {
        OptionalLong free = free(device);
        if (free.isPresent() && weights + MIN_WORKING > free.getAsLong()) {
            throw new IllegalArgumentException(String.format(Locale.ROOT,
                    "%s does not fit on %s: its weights take about %s and generating needs at least %s more,"
                            + " but only %s is free%s", what, device, gb(weights), gb(MIN_WORKING),
                    gb(free.getAsLong()), hint.isEmpty() ? "" : " (" + hint + ")"));
        }
    }

    static String gb(long bytes) {
        return String.format(Locale.ROOT, "%.1f GB", bytes / 1e9);
    }
}
