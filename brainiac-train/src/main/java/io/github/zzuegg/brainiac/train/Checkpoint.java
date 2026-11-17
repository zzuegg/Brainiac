package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Everything needed to stop training and carry on later as if it had not stopped: the
 * weights, the optimizer's state, the weight averages and the step count, as a directory
 * of safetensors files ({@code model.safetensors}, {@code optimizer.safetensors}, {@code
 * ema.safetensors}) and {@code trainer.json}. {@link TrainingLoop#save} and {@link
 * TrainingLoop#resume} use it.
 */
public final class Checkpoint {

    private Checkpoint() {
    }

    /**
     * Writes a checkpoint. Files are written next to the old ones and then renamed, so a
     * crash while saving leaves the previous checkpoint readable.
     *
     * @param dir the directory, created if needed
     * @param parameters the weights by name
     * @param optimizer their optimizer
     * @param ema their averages, or {@code null}
     * @param step the steps taken
     * @param extra more values to keep, such as the data position; strings, numbers or booleans
     */
    public static void save(Path dir, Map<String, Var> parameters, Optimizer optimizer, Ema ema, int step,
            Map<String, Object> extra) {
        try {
            Files.createDirectories(dir);
            Map<String, Tensor> weights = new LinkedHashMap<>();
            parameters.forEach((k, v) -> weights.put(k, v.value()));
            List<Var> order = new ArrayList<>(parameters.values());
            write(dir, "model.safetensors", weights);
            write(dir, "optimizer.safetensors", optimizer.state(order));
            if (ema != null) {
                write(dir, "ema.safetensors", ema.state());
            }
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("step", step);
            info.put("optimizer", optimizer.toString());
            info.put("extra", extra == null ? Map.of() : extra);
            Path tmp = dir.resolve("trainer.json.tmp");
            Files.writeString(tmp, Json.write(info), StandardCharsets.UTF_8);
            Files.move(tmp, dir.resolve("trainer.json"), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void write(Path dir, String name, Map<String, Tensor> tensors) throws IOException {
        Path tmp = dir.resolve(name + ".tmp");
        SafeTensors.write(tmp, tensors);
        Files.move(tmp, dir.resolve(name), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                java.nio.file.StandardCopyOption.ATOMIC_MOVE);
    }

    /**
     * Reads a checkpoint back into existing weights, optimizer and averages.
     *
     * @param dir the directory {@link #save} wrote
     * @param parameters the weights by the same names; their values are replaced
     * @param optimizer gets its saved state
     * @param ema gets the saved averages, or {@code null}
     * @return the saved step count and extra values
     */
    public static Saved load(Path dir, Map<String, Var> parameters, Optimizer optimizer, Ema ema) {
        try {
            Map<String, Object> info = Json.object(Json.read(dir.resolve("trainer.json")));
            try (SafeTensors w = SafeTensors.open(dir.resolve("model.safetensors"))) {
                for (Map.Entry<String, Var> e : parameters.entrySet()) {
                    Tensor current = e.getValue().value();
                    if (!w.names().contains(e.getKey())) {
                        throw new IllegalArgumentException("the checkpoint has no weight " + e.getKey());
                    }
                    Tensor t = w.tensor(e.getKey(), current.device(), current.dtype());
                    if (!t.shape().equals(current.shape())) {
                        throw new IllegalArgumentException(e.getKey() + " has shape " + t.shape() + " in the checkpoint, "
                                + current.shape() + " in the model");
                    }
                    e.getValue().assign(t);
                }
            }
            List<Var> order = new ArrayList<>(parameters.values());
            optimizer.loadState(order, read(dir.resolve("optimizer.safetensors"), order));
            Path emaFile = dir.resolve("ema.safetensors");
            if (ema != null && Files.isRegularFile(emaFile)) {
                ema.loadState(read(emaFile, order));
            }
            int step = (int) Json.integer(info, "step");
            Object extra = info.get("extra");
            return new Saved(step, extra == null ? Map.of() : Json.object(extra));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Map<String, Tensor> read(Path file, List<Var> order) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        var device = order.isEmpty() ? io.github.zzuegg.brainiac.tensor.Device.getDefault()
                : order.getFirst().value().device();
        try (SafeTensors s = SafeTensors.open(file)) {
            for (String n : s.names()) {
                out.put(n, s.tensor(n, device, s.naturalType(n)));
            }
        }
        return out;
    }

    /**
     * What a checkpoint holds besides tensors.
     *
     * @param step the steps taken
     * @param extra the extra values saved with it
     */
    public record Saved(int step, Map<String, Object> extra) {
    }
}
