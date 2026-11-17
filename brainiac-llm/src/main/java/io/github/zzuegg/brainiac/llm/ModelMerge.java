package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.stream.Stream;

/**
 * Merges fine-tuned models of the same architecture into one, with the methods mergekit
 * offers: a weighted average, SLERP, task arithmetic, TIES and DARE.
 *
 * <pre>
 * ModelMerge.merge(new ModelMerge.Ties(new double[] {1, 1}, 0.5), Path.of("Qwen2.5-0.5B"),
 *         List.of(Path.of("qwen-math"), Path.of("qwen-code")), Path.of("qwen-merged"));
 * </pre>
 *
 * Each tensor is merged in float32 on the host and saved in the first model's type, one at a
 * time, so a merge needs memory for a few tensors and not for whole models.
 */
public final class ModelMerge {

    private ModelMerge() {
    }

    /** How the tensors are combined. */
    public sealed interface Method permits Linear, Slerp, TaskArithmetic, Ties, Dare {

        /** @return whether the method works on differences from a base model */
        default boolean needsBase() {
            return true;
        }
    }

    /**
     * The weighted mean of the models.
     *
     * @param weights one per model
     */
    public record Linear(double[] weights) implements Method {

        @Override
        public boolean needsBase() {
            return false;
        }
    }

    /**
     * Spherical interpolation between two models, per tensor, falling back to a straight line
     * when they point nearly the same way.
     *
     * @param t 0 for the first model, 1 for the second
     */
    public record Slerp(double t) implements Method {

        @Override
        public boolean needsBase() {
            return false;
        }
    }

    /**
     * The base plus the weighted sum of each model's difference from it (Ilharco et al., 2023).
     *
     * @param weights one per model
     */
    public record TaskArithmetic(double[] weights) implements Method {
    }

    /**
     * TIES (Yadav et al., 2023): keeps each difference's largest {@code density} share of
     * values, elects a sign per value by the weighted sum, and averages the differences that
     * agree with it.
     *
     * @param weights one per model
     * @param density the share of each difference kept, in (0, 1]
     */
    public record Ties(double[] weights, double density) implements Method {
    }

    /**
     * DARE (Yu et al., 2024): drops each difference value at random with probability
     * {@code 1 - density} and scales the rest by {@code 1 / density}, then adds the weighted
     * differences to the base.
     *
     * @param weights one per model
     * @param density the share of each difference kept, in (0, 1]
     * @param seed seeds the drops
     */
    public record Dare(double[] weights, double density, long seed) implements Method {
    }

    /**
     * Merges one tensor.
     *
     * @param method the method
     * @param name the tensor's name, which seeds DARE's drops
     * @param base the base model's tensor, or {@code null} for methods that need none
     * @param models the models' tensors, all of one shape
     * @return the merged values, in float32
     */
    public static float[] merge(Method method, String name, float[] base, List<float[]> models) {
        int n = models.getFirst().length;
        for (float[] m : models) {
            if (m.length != n || (base != null && base.length != n)) {
                throw new IllegalArgumentException(name + ": the models' tensors differ in size");
            }
        }
        return switch (method) {
            case Linear l -> linear(weights(l.weights(), models.size()), models);
            case Slerp s -> {
                if (models.size() != 2) {
                    throw new IllegalArgumentException("SLERP merges exactly two models");
                }
                yield slerp(s.t(), models.get(0), models.get(1));
            }
            case TaskArithmetic t -> {
                double[] w = weights(t.weights(), models.size());
                float[] out = base.clone();
                for (int i = 0; i < models.size(); i++) {
                    float[] m = models.get(i);
                    for (int j = 0; j < n; j++) {
                        out[j] += (float) (w[i] * (m[j] - base[j]));
                    }
                }
                yield out;
            }
            case Ties t -> ties(weights(t.weights(), models.size()), density(t.density()), base, models);
            case Dare d -> {
                double[] w = weights(d.weights(), models.size());
                double p = density(d.density());
                SplittableRandom r = new SplittableRandom(d.seed() ^ name.hashCode() * 0x9E3779B97F4A7C15L);
                float[] out = base.clone();
                for (int i = 0; i < models.size(); i++) {
                    float[] m = models.get(i);
                    for (int j = 0; j < n; j++) {
                        if (r.nextDouble() < p) {
                            out[j] += (float) (w[i] * (m[j] - base[j]) / p);
                        }
                    }
                }
                yield out;
            }
        };
    }

    private static double[] weights(double[] w, int models) {
        if (w.length != models) {
            throw new IllegalArgumentException(w.length + " weights for " + models + " models");
        }
        return w;
    }

    private static double density(double d) {
        if (!(d > 0 && d <= 1)) {
            throw new IllegalArgumentException("density must be in (0, 1], got " + d);
        }
        return d;
    }

    private static float[] linear(double[] w, List<float[]> models) {
        double total = Arrays.stream(w).sum();
        if (total == 0) {
            throw new IllegalArgumentException("the weights add up to 0");
        }
        float[] out = new float[models.getFirst().length];
        for (int j = 0; j < out.length; j++) {
            double s = 0;
            for (int i = 0; i < w.length; i++) {
                s += w[i] * models.get(i)[j];
            }
            out[j] = (float) (s / total);
        }
        return out;
    }

    private static float[] slerp(double t, float[] a, float[] b) {
        double na = 0;
        double nb = 0;
        double dot = 0;
        for (int j = 0; j < a.length; j++) {
            na += (double) a[j] * a[j];
            nb += (double) b[j] * b[j];
            dot += (double) a[j] * b[j];
        }
        double cos = na == 0 || nb == 0 ? 1 : dot / Math.sqrt(na * nb);
        double sa;
        double sb;
        if (Math.abs(cos) > 0.9995) {
            sa = 1 - t;
            sb = t;
        } else {
            double theta = Math.acos(cos);
            double sin = Math.sin(theta);
            sa = Math.sin((1 - t) * theta) / sin;
            sb = Math.sin(t * theta) / sin;
        }
        float[] out = new float[a.length];
        for (int j = 0; j < a.length; j++) {
            out[j] = (float) (sa * a[j] + sb * b[j]);
        }
        return out;
    }

    private static float[] ties(double[] w, double density, float[] base, List<float[]> models) {
        int n = base.length;
        int k = models.size();
        float[][] delta = new float[k][];
        for (int i = 0; i < k; i++) {
            float[] m = models.get(i);
            float[] d = new float[n];
            for (int j = 0; j < n; j++) {
                d[j] = m[j] - base[j];
            }
            int keep = (int) Math.ceil(density * n);
            if (keep < n) {
                float[] mags = new float[n];
                for (int j = 0; j < n; j++) {
                    mags[j] = Math.abs(d[j]);
                }
                Arrays.sort(mags);
                float threshold = mags[n - keep];
                int kept = 0;
                for (int j = 0; j < n; j++) {   // ties at the threshold kept in order
                    if (Math.abs(d[j]) > threshold) {
                        kept++;
                    }
                }
                int atThreshold = keep - kept;
                for (int j = 0; j < n; j++) {
                    float a = Math.abs(d[j]);
                    if (a < threshold || (a == threshold && atThreshold-- <= 0)) {
                        d[j] = 0;
                    }
                }
            }
            delta[i] = d;
        }
        float[] out = base.clone();
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < k; i++) {
                sum += w[i] * delta[i][j];
            }
            if (sum == 0) {
                continue;
            }
            double num = 0;
            double den = 0;
            for (int i = 0; i < k; i++) {
                double v = w[i] * delta[i][j];
                if (v != 0 && (v > 0) == (sum > 0)) {
                    num += v;
                    den += w[i];
                }
            }
            if (den != 0) {
                out[j] += (float) (num / den);
            }
        }
        return out;
    }

    /**
     * Merges model folders (or weight files) and writes the result as safetensors, split into
     * parts of about 2 GB with an index, with the first model's other files (configuration,
     * tokenizer) copied alongside.
     *
     * @param method the method
     * @param base the base model, or {@code null} for methods that need none
     * @param models the models to merge
     * @param out the folder to write
     */
    public static void merge(Method method, Path base, List<Path> models, Path out) {
        merge(method, base, models, out, 2L << 30);
    }

    static void merge(Method method, Path base, List<Path> models, Path out, long partBytes) {
        Objects.requireNonNull(method, "method");
        if (models.isEmpty()) {
            throw new IllegalArgumentException("no models to merge");
        }
        if (method.needsBase() && base == null) {
            throw new IllegalArgumentException(method.getClass().getSimpleName() + " needs a base model");
        }
        List<WeightSource> sources = new ArrayList<>();
        WeightSource baseSource = null;
        try {
            for (Path m : models) {
                sources.add(WeightSource.open(m));
            }
            baseSource = method.needsBase() ? WeightSource.open(base) : null;
            Files.createDirectories(out);
            write(method, baseSource, sources, out, partBytes);
            Path first = models.getFirst();
            if (Files.isDirectory(first)) {
                try (Stream<Path> files = Files.list(first)) {
                    for (Path f : files.toList()) {
                        String n = f.getFileName().toString();
                        if (Files.isRegularFile(f) && !n.endsWith(".safetensors") && !n.endsWith(".gguf")
                                && !n.equals("model.safetensors.index.json")) {
                            Files.copy(f, out.resolve(n), StandardCopyOption.REPLACE_EXISTING);
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            for (WeightSource s : sources) {
                close(s);
            }
            if (baseSource != null) {
                close(baseSource);
            }
        }
    }

    private static void close(WeightSource s) {
        try {
            s.close();
        } catch (Exception e) {
            // nothing to do
        }
    }

    private static void write(Method method, WeightSource base, List<WeightSource> models, Path out,
            long partBytes) throws IOException {
        WeightSource first = models.getFirst();
        Device cpu = Device.cpu();
        Map<String, Tensor> part = new LinkedHashMap<>();
        long bytes = 0;
        List<String> names = first.names();
        // write each part as soon as it is full, so only one part is in memory
        List<Path> written = new ArrayList<>();
        Map<String, Integer> where = new LinkedHashMap<>();
        for (String name : names) {
            for (WeightSource m : models) {
                if (!m.contains(name) || !m.shape(name).equals(first.shape(name))) {
                    throw new IllegalArgumentException("the models differ at " + name);
                }
            }
            DType type = first.naturalType(name);
            List<float[]> values = new ArrayList<>();
            for (WeightSource m : models) {
                values.add(read(m, name, cpu));
            }
            float[] b = base == null ? null : read(base, name, cpu);
            Tensor merged = Tensor.of(first.shape(name), merge(method, name, b, values), cpu).to(type);
            long size = merged.shape().size() * (long) merged.dtype().bytes();
            if (!part.isEmpty() && bytes + size > partBytes) {
                written.add(flush(part, out, written.size()));
                bytes = 0;
                part = new LinkedHashMap<>();
            }
            part.put(name, merged);
            where.put(name, written.size());
            bytes += size;
        }
        written.add(flush(part, out, written.size()));
        int total = written.size();
        Map<String, Object> weightMap = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : where.entrySet()) {
            weightMap.put(e.getKey(), partName(e.getValue(), total));
        }
        if (total == 1) {
            Files.move(written.getFirst(), out.resolve("model.safetensors"),
                    StandardCopyOption.REPLACE_EXISTING);
            Files.deleteIfExists(out.resolve("model.safetensors.index.json"));
            return;
        }
        for (int i = 0; i < total; i++) {
            Files.move(written.get(i), out.resolve(partName(i, total)), StandardCopyOption.REPLACE_EXISTING);
        }
        Files.writeString(out.resolve("model.safetensors.index.json"),
                Json.write(Map.of("metadata", Map.of(), "weight_map", weightMap)));
    }

    private static float[] read(WeightSource s, String name, Device cpu) {
        Tensor t = s.tensor(name, cpu, DType.FLOAT32);
        try {
            return t.toArray();
        } finally {
            t.release();
        }
    }

    private static Path flush(Map<String, Tensor> part, Path out, int index) {
        Path file = out.resolve("merge-part-" + index + ".tmp");
        SafeTensors.write(file, part, Map.of("format", "pt"));
        for (Tensor t : part.values()) {
            t.release();
        }
        return file;
    }

    private static String partName(int i, int total) {
        return String.format("model-%05d-of-%05d.safetensors", i + 1, total);
    }
}
