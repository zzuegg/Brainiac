package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * The folders the model finders look into: each model folder the user has, and the folders below it
 * that hold models, a few levels down. A folder such as {@code D:\models\llm\qwen} or a Hugging Face
 * cache ({@code models--Qwen--Qwen2.5-0.5B/snapshots/<hash>}) is then found from its top folder.
 *
 * <p>A folder with a {@code config.json} or {@code model_index.json} is a model: the finders see it
 * from the folder above, and nothing inside it is looked into. Hidden folders and the folders of code
 * projects ({@code .git}, {@code build}, {@code node_modules} and the like) are left out, so a folder
 * of projects can be given too.
 */
final class ModelPlaces implements Supplier<List<Path>> {

    /** How many folders down from a model folder models are still found. */
    static final int DEPTH = 5;
    /** How many folders are looked at below one model folder at most. */
    static final int MOST = 5000;
    private static final long FRESH_NANOS = 5_000_000_000L;
    private static final Set<String> SKIP = Set.of("node_modules", "build", "target", "out", "dist", "bin", "obj",
            "src", "gradle", "__pycache__", "venv", "env", "site-packages", "blobs", "refs", "locks",
            "system volume information", "windows", "program files", "program files (x86)", "programdata");

    private final Supplier<List<Path>> roots;
    private List<Path> lastRoots = List.of();
    private List<Path> last = List.of();
    private long at;

    /**
     * Creates the places below these folders.
     *
     * @param roots the model folders, and the files and folders added on the Models screen
     */
    ModelPlaces(Supplier<List<Path>> roots) {
        this.roots = roots;
    }

    /**
     * Returns the places, looked for again when the folders change or after a few seconds.
     *
     * @return the model folders, each followed by the folders below it that may hold models
     */
    @Override
    public synchronized List<Path> get() {
        List<Path> r = roots.get();
        long now = System.nanoTime();
        if (!r.equals(lastRoots) || now - at > FRESH_NANOS) {
            last = expand(r);
            lastRoots = List.copyOf(r);
            at = System.nanoTime();
        }
        return last;
    }

    /**
     * Returns these places, each followed by the folders below it that may hold models.
     *
     * @param roots the places
     * @return them and the folders below them, without repeats
     */
    static List<Path> expand(List<Path> roots) {
        Set<Path> out = new LinkedHashSet<>();
        for (Path root : roots) {
            out.add(root);
            if (!Files.isDirectory(root) || isModel(root)) {
                continue;
            }
            record Step(Path dir, int depth) {
            }
            ArrayDeque<Step> todo = new ArrayDeque<>();
            todo.add(new Step(root, 0));
            int seen = 0;
            while (!todo.isEmpty() && seen < MOST) {
                Step s = todo.poll();
                if (s.depth() >= DEPTH || holdsWeights(s.dir())) {
                    continue;
                }
                for (Path d : folders(s.dir())) {
                    seen++;
                    if (isModel(d)) {
                        continue;
                    }
                    out.add(d);
                    todo.add(new Step(d, s.depth() + 1));
                }
            }
        }
        return List.copyOf(out);
    }

    /**
     * Returns a model's name from its folder or file: the folder's own name, or for a Hugging Face
     * cache snapshot ({@code models--Qwen--Qwen2.5-0.5B/snapshots/<hash>}) the model's name.
     *
     * @param p the folder or file
     * @return its name
     */
    static String name(Path p) {
        Path parent = p.getParent(), grand = parent == null ? null : parent.getParent();
        if (grand != null && grand.getFileName() != null && "snapshots".equals(String.valueOf(parent.getFileName()))) {
            String repo = grand.getFileName().toString();
            int at = repo.lastIndexOf("--");
            if (repo.startsWith("models--") && at >= "models".length()) {
                return repo.substring(at + 2);
            }
        }
        return p.getFileName() == null ? p.toString() : p.getFileName().toString();
    }

    private static List<Path> folders(Path dir) {
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(ModelPlaces::worthALook).sorted().toList();
        } catch (IOException | RuntimeException e) {
            return List.of();
        }
    }

    private static boolean worthALook(Path p) {
        String n = p.getFileName() == null ? "" : p.getFileName().toString();
        return !n.startsWith(".") && !n.startsWith("$") && !SKIP.contains(n.toLowerCase(Locale.ROOT))
                && Files.isDirectory(p);
    }

    private static boolean isModel(Path p) {
        return Files.isRegularFile(p.resolve("config.json")) || Files.isRegularFile(p.resolve("model_index.json"));
    }

    // a folder of weight files (GGUF files, a checkpoint and its vocoder): looked into, not below
    private static boolean holdsWeights(Path dir) {
        try (Stream<Path> s = Files.list(dir)) {
            return s.anyMatch(f -> {
                String n = String.valueOf(f.getFileName()).toLowerCase(Locale.ROOT);
                return n.endsWith(".gguf") || n.endsWith(".safetensors") || n.endsWith(".ckpt") || n.endsWith(".pt")
                        || n.endsWith(".pth");
            });
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }
}
