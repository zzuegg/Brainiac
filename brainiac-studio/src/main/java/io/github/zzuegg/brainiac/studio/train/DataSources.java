package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.data.CsvLoader;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.data.Images;
import io.github.zzuegg.brainiac.io.Json;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The data a run can train on: small built-in sets that need no download, and files in the
 * data folder's {@code datasets} folder (or anywhere on this machine): CSV tables, MNIST-style
 * IDX pairs and image folders with one subfolder per class.
 *
 * <p>A data source is described as JSON, for example {@code {"kind": "builtin", "name": "spirals"}},
 * {@code {"kind": "csv", "file": "iris.csv", "label": "species"}},
 * {@code {"kind": "idx", "images": "train-images-idx3-ubyte.gz", "labels": "train-labels-idx1-ubyte.gz"}}
 * or {@code {"kind": "images", "folder": "pets", "channels": 3, "height": 64, "width": 64}}.
 */
public final class DataSources {

    /** The most samples a built-in set or a capped file is read with. */
    static final int MAX_SAMPLES = 2_000_000;

    /**
     * A built-in set.
     *
     * @param name its name
     * @param what what it is, in a sentence
     * @param input the shape of one sample
     * @param targets how many target values a sample has
     * @param classes true if the targets are classes (one-hot)
     */
    public record Builtin(String name, String what, List<Integer> input, int targets, boolean classes) {
    }

    /** The built-in sets. */
    public static final List<Builtin> BUILTINS = List.of(
            new Builtin("xor", "400 points; is x times y positive? The classic test that needs a hidden layer.",
                    List.of(2), 1, false),
            new Builtin("spirals", "900 points on three interleaved spirals, three classes.", List.of(2), 3, true),
            new Builtin("circles", "600 points on two rings, inside or outside.", List.of(2), 2, true),
            new Builtin("blobs", "800 points in four clusters, four classes.", List.of(2), 4, true),
            new Builtin("sine", "1,000 values of sin(x) with a little noise, for regression.", List.of(1), 1, false),
            new Builtin("stripes", "2,000 colour images of 32 × 32 with horizontal, vertical or diagonal stripes or a "
                    + "checkerboard, four classes. Fits the builder's default 3 × 32 × 32 input.", List.of(3, 32, 32), 4, true),
            new Builtin("strokes", "1,500 grey images of 8 × 8 with a horizontal, vertical or diagonal stroke, three classes.",
                    List.of(1, 8, 8), 3, true));

    private final Path folder;

    /**
     * Uses a folder for data files.
     *
     * @param folder the {@code datasets} folder; relative file names are looked up here
     */
    public DataSources(Path folder) {
        this.folder = folder;
    }

    /**
     * Returns the folder data files are looked up in.
     *
     * @return the folder
     */
    public Path folder() {
        return folder;
    }

    /**
     * Loads a data source.
     *
     * @param spec its description
     * @return the dataset
     * @throws IOException if a file cannot be read
     */
    public Dataset load(Map<String, Object> spec) throws IOException {
        String kind = Json.string(spec, "kind", "builtin");
        return switch (kind) {
            case "builtin" -> builtin(Json.string(spec, "name", "spirals"));
            case "csv" -> csv(spec);
            case "idx" -> Dataset.idx(file(spec, "images"), file(spec, "labels"),
                    (int) Math.min(MAX_SAMPLES, Json.integer(spec, "limit", MAX_SAMPLES)));
            case "images" -> Images.folder(file(spec, "folder"), (int) Json.integer(spec, "channels", 3),
                    (int) Json.integer(spec, "height", 32), (int) Json.integer(spec, "width", 32));
            default -> throw new IllegalArgumentException("unknown data kind " + kind + "; use builtin, csv, idx or images");
        };
    }

    /**
     * Says what a data source is, in a few words, for the run list.
     *
     * @param spec its description
     * @return for example {@code spirals} or {@code iris.csv}
     */
    public static String label(Map<String, Object> spec) {
        return switch (Json.string(spec, "kind", "builtin")) {
            case "csv" -> fileName(Json.string(spec, "file", "?"));
            case "idx" -> fileName(Json.string(spec, "images", "?"));
            case "images" -> fileName(Json.string(spec, "folder", "?"));
            default -> Json.string(spec, "name", "spirals");
        };
    }

    private static String fileName(String path) {
        String p = path.replace('\\', '/');
        return p.substring(p.lastIndexOf('/') + 1);
    }

    private Dataset csv(Map<String, Object> spec) {
        Path file = file(spec, "file");
        CsvLoader loader = Dataset.csv(file);
        String delimiter = Json.string(spec, "delimiter", ",");
        if (delimiter.length() == 1) {
            loader.delimiter(delimiter.charAt(0));
        }
        List<String> header = header(file, delimiter.isEmpty() ? ',' : delimiter.charAt(0));
        String label = Json.string(spec, "label", null);
        List<String> targets = spec.get("targets") == null ? List.of() : strings(spec.get("targets"));
        if (label == null && targets.isEmpty()) {
            if (header.isEmpty()) {
                throw new IllegalArgumentException(file.getFileName() + " has no header row to pick columns from");
            }
            label = header.getLast();   // the usual layout: the answer in the last column
        }
        List<String> features = spec.get("features") == null ? List.of() : strings(spec.get("features"));
        if (features.isEmpty()) {
            List<String> rest = new ArrayList<>(header);
            rest.remove(label);
            rest.removeAll(targets);
            features = rest;
        }
        loader.features(features.toArray(String[]::new));
        if (label != null) {
            loader.label(label);
            if (!Boolean.FALSE.equals(spec.get("classes"))) {
                loader.oneHot();
            }
        } else {
            loader.targets(targets.toArray(String[]::new));
        }
        if (Boolean.TRUE.equals(spec.get("normalize"))) {
            loader.normalize();
        }
        return loader.load();
    }

    private static List<String> strings(Object o) {
        return Json.array(o).stream().map(String::valueOf).toList();
    }

    /** The column names in the first line of a CSV file, or an empty list if it cannot be read. */
    static List<String> header(Path file, char delimiter) {
        try (BufferedReader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String line = r.readLine();
            if (line == null) {
                return List.of();
            }
            List<String> out = new ArrayList<>();
            for (String part : line.split(java.util.regex.Pattern.quote(String.valueOf(delimiter)), -1)) {
                out.add(part.strip().replace("\"", ""));
            }
            return out;
        } catch (IOException | RuntimeException e) {
            return List.of();
        }
    }

    private Path file(Map<String, Object> spec, String key) {
        String name = Json.string(spec, key, null);
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("the data source needs " + key);
        }
        return resolve(name);
    }

    /**
     * Finds a data file or folder: a relative name is looked up in the data folder.
     *
     * @param name the name or path
     * @return the file or folder
     */
    public Path resolve(String name) {
        Path p = Path.of(name);
        Path resolved = p.isAbsolute() ? p : folder.resolve(p).normalize();
        if (!Files.exists(resolved)) {
            throw new IllegalArgumentException("no such file or folder: " + resolved);
        }
        return resolved;
    }

    /**
     * Lists the built-in sets and what is in the data folder.
     *
     * @return {@code {folder, builtins: [...], files: [...]}}
     * @throws IOException if the folder cannot be listed
     */
    public Map<String, Object> list() throws IOException {
        List<Object> builtins = new ArrayList<>();
        for (Builtin b : BUILTINS) {
            builtins.add(Map.of("name", b.name(), "what", b.what(), "input", b.input(), "targets", b.targets(),
                    "classes", b.classes()));
        }
        List<Object> files = new ArrayList<>();
        if (Files.isDirectory(folder)) {
            List<Path> entries = new ArrayList<>();
            try (DirectoryStream<Path> ds = Files.newDirectoryStream(folder)) {
                ds.forEach(entries::add);
            }
            entries.sort(null);
            for (Path p : entries) {
                String name = p.getFileName().toString();
                String lower = name.toLowerCase(Locale.ROOT);
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("name", name);
                if (Files.isDirectory(p)) {
                    List<String> classes = Images.classes(p);
                    if (classes.isEmpty()) {
                        continue;
                    }
                    m.put("kind", "images");
                    m.put("classes", classes);
                } else if (lower.endsWith(".csv") || lower.endsWith(".tsv")) {
                    m.put("kind", "csv");
                    m.put("columns", header(p, lower.endsWith(".tsv") ? '\t' : ','));
                    m.put("bytes", Files.size(p));
                } else if (lower.contains("images-idx") || lower.contains("images.idx")) {
                    m.put("kind", "idx");
                    String labels = name.replace("images-idx3", "labels-idx1").replace("images.idx3", "labels.idx1");
                    if (!labels.equals(name) && Files.exists(folder.resolve(labels))) {
                        m.put("labels", labels);
                    }
                    m.put("bytes", Files.size(p));
                } else {
                    continue;
                }
                files.add(m);
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("folder", folder.toString());
        out.put("builtins", builtins);
        out.put("files", files);
        return out;
    }

    // ---------------------------------------------------------------- built-in sets

    /**
     * Makes a built-in set. The same name always gives the same samples.
     *
     * @param name its name
     * @return the dataset
     */
    public static Dataset builtin(String name) {
        SplittableRandom r = new SplittableRandom(7);
        return switch (name) {
            case "xor" -> {
                float[][] x = new float[400][];
                float[][] y = new float[400][];
                for (int i = 0; i < x.length; i++) {
                    float a = (float) r.nextDouble(-1, 1);
                    float b = (float) r.nextDouble(-1, 1);
                    x[i] = new float[] {a, b};
                    y[i] = new float[] {a * b > 0 ? 1 : 0};
                }
                yield Dataset.of(x, y);
            }
            case "spirals" -> {
                int per = 300;
                float[][] x = new float[per * 3][];
                float[][] y = new float[per * 3][];
                for (int c = 0; c < 3; c++) {
                    for (int i = 0; i < per; i++) {
                        double t = (double) i / per;
                        double angle = c * 2 * Math.PI / 3 + t * 4 + r.nextGaussian() * 0.2;
                        x[c * per + i] = new float[] {(float) (t * Math.cos(angle)), (float) (t * Math.sin(angle))};
                        y[c * per + i] = oneHot(c, 3);
                    }
                }
                yield Dataset.of(x, y);
            }
            case "circles" -> {
                float[][] x = new float[600][];
                float[][] y = new float[600][];
                for (int i = 0; i < x.length; i++) {
                    int c = i % 2;
                    double radius = (c == 0 ? 0.4 : 1.0) + r.nextGaussian() * 0.08;
                    double angle = r.nextDouble(0, 2 * Math.PI);
                    x[i] = new float[] {(float) (radius * Math.cos(angle)), (float) (radius * Math.sin(angle))};
                    y[i] = oneHot(c, 2);
                }
                yield Dataset.of(x, y);
            }
            case "blobs" -> {
                double[][] centres = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
                float[][] x = new float[800][];
                float[][] y = new float[800][];
                for (int i = 0; i < x.length; i++) {
                    int c = i % 4;
                    x[i] = new float[] {(float) (centres[c][0] + r.nextGaussian() * 0.35),
                            (float) (centres[c][1] + r.nextGaussian() * 0.35)};
                    y[i] = oneHot(c, 4);
                }
                yield Dataset.of(x, y);
            }
            case "sine" -> {
                float[][] x = new float[1000][];
                float[][] y = new float[1000][];
                for (int i = 0; i < x.length; i++) {
                    double v = r.nextDouble(-Math.PI, Math.PI);
                    x[i] = new float[] {(float) v};
                    y[i] = new float[] {(float) (Math.sin(v) + r.nextGaussian() * 0.05)};
                }
                yield Dataset.of(x, y);
            }
            case "stripes" -> images(r, 2000, 3, 32, 4);
            case "strokes" -> images(r, 1500, 1, 8, 3);
            default -> throw new IllegalArgumentException("no built-in data set " + name + "; there are "
                    + String.join(", ", BUILTINS.stream().map(Builtin::name).toList()));
        };
    }

    private static float[] oneHot(int c, int n) {
        float[] v = new float[n];
        v[c] = 1;
        return v;
    }

    /** Images of size×size with a pattern per class: horizontal, vertical, diagonal, checkerboard. */
    private static Dataset images(SplittableRandom r, int n, int channels, int size, int classes) {
        float[][] x = new float[n][];
        float[][] y = new float[n][];
        for (int i = 0; i < n; i++) {
            int c = i % classes;
            float[] img = new float[channels * size * size];
            int period = 2 + r.nextInt(Math.max(1, size / 4));
            int shift = r.nextInt(period);
            float[] colour = new float[channels];
            for (int ch = 0; ch < channels; ch++) {
                colour[ch] = (float) r.nextDouble(0.5, 1);
            }
            int line = r.nextInt(size);
            for (int row = 0; row < size; row++) {
                for (int col = 0; col < size; col++) {
                    boolean on = switch (c) {
                        case 0 -> classes == 3 ? row == line : (row + shift) % period == 0;
                        case 1 -> classes == 3 ? col == line : (col + shift) % period == 0;
                        case 2 -> classes == 3 ? row == col : (row + col + shift) % period == 0;
                        default -> ((row / period) + (col / period)) % 2 == 0;
                    };
                    for (int ch = 0; ch < channels; ch++) {
                        float noise = (float) (r.nextGaussian() * 0.08);
                        img[(ch * size + row) * size + col] = (on ? colour[ch] : 0.1f) + noise;
                    }
                }
            }
            x[i] = img;
            y[i] = oneHot(c, classes);
        }
        return Dataset.of(x, y);
    }
}
