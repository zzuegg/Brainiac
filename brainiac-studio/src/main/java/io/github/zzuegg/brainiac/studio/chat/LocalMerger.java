package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.llm.ModelMerge;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Merges chat models of this PC that share an architecture, such as two fine-tunes of one base
 * model, into a new model folder next to the first, with the methods mergekit offers. One merge
 * runs at a time, on the CPU and a tensor at a time, so it needs little memory and a chat on
 * the GPU keeps running; the page asks how it is going.
 */
public final class LocalMerger {

    /**
     * A way to merge.
     *
     * @param id its name here
     * @param label what the page calls it
     * @param base whether it works on differences from a base model
     * @param note what it does
     */
    public record Way(String id, String label, boolean base, String note) {
    }

    /** The ways offered, the simplest first. */
    public static final List<Way> WAYS = List.of(
            new Way("linear", "Average", false, "the weighted mean of the models"),
            new Way("slerp", "SLERP", false, "a path along the sphere between exactly two models; the balance says how far along"),
            new Way("task", "Task arithmetic", true, "adds each model's difference from the base, weighted, onto the base"),
            new Way("ties", "TIES", true, "keeps each model's largest changes from the base and drops the ones that disagree in sign"),
            new Way("dare", "DARE", true, "drops changes from the base at random and scales up the rest, then adds them"));

    private record Job(String name, String way, List<String> models, Path folder, long started, String state,
            String error, long ended) {

        Job to(String next, String why) {
            return new Job(name, way, models, folder, started, next, why, System.currentTimeMillis());
        }
    }

    private final LocalProvider local;
    private volatile Job job;

    /**
     * A merger of a provider's models.
     *
     * @param local this PC's models
     */
    public LocalMerger(LocalProvider local) {
        this.local = local;
    }

    /**
     * Describes the ways to merge and the merge under way or last made, for the page.
     *
     * @return {@code {ways: [{id, label, base, note}], job?: {name, way, models, folder, state,
     *     error?, seconds}}}; a job's state is {@code merging}, {@code done} or {@code failed}
     */
    public Map<String, Object> state() {
        Map<String, Object> out = new LinkedHashMap<>();
        List<Object> ways = new ArrayList<>();
        for (Way w : WAYS) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", w.id());
            m.put("label", w.label());
            m.put("base", w.base());
            m.put("note", w.note());
            ways.add(m);
        }
        out.put("ways", ways);
        Job j = job;
        if (j != null) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("name", j.name());
            m.put("way", j.way());
            m.put("models", j.models());
            m.put("folder", j.folder().toString());
            m.put("state", j.state());
            if (j.error() != null) {
                m.put("error", j.error());
            }
            m.put("seconds", ((j.ended() > 0 ? j.ended() : System.currentTimeMillis()) - j.started()) / 1000.0);
            out.put("job", m);
        }
        return out;
    }

    /**
     * Returns whether a merge is under way.
     *
     * @return whether one is
     */
    public boolean running() {
        Job j = job;
        return j != null && j.state().equals("merging");
    }

    /**
     * Starts merging chat models into a new model folder, {@code <name>} next to the first of
     * them, which shows up in the model list when it is done.
     *
     * @param way one of {@link #WAYS}, by its id
     * @param base the base model for the ways that need one, as the provider lists it
     * @param models the models to merge, as the provider lists them: two or more model folders
     * @param weights one weight per model, or empty for equal weights
     * @param balance for SLERP, how far from the first model to the second, 0 to 1
     * @param density for TIES and DARE, the share of each model's changes that is kept, above 0 up to 1
     * @param name what the merged model is called
     * @return the state, as {@link #state} gives it
     * @throws IOException if a model is not there or is a GGUF file, the name is taken, or another merge is under way
     */
    public synchronized Map<String, Object> start(String way, String base, List<String> models, List<Double> weights,
            double balance, double density, String name) throws IOException {
        Way w = WAYS.stream().filter(x -> x.id().equals(way)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("choose one of " + WAYS.stream().map(Way::id).toList()));
        if (running()) {
            throw new IOException(job.name() + " is being merged: wait for it to finish");
        }
        if (name == null || !name.matches("[A-Za-z0-9._-]{1,60}") || name.startsWith(".")) {
            throw new IllegalArgumentException("name it with up to 60 letters, digits, dots, dashes or underscores");
        }
        if (models.size() < 2 || models.stream().distinct().count() != models.size()) {
            throw new IllegalArgumentException("choose two or more different models");
        }
        if (w.id().equals("slerp") && models.size() != 2) {
            throw new IllegalArgumentException("SLERP merges exactly two models");
        }
        if (!weights.isEmpty() && weights.size() != models.size()) {
            throw new IllegalArgumentException("give one weight for each model, or none");
        }
        if (weights.stream().anyMatch(x -> x == null || !Double.isFinite(x))) {
            throw new IllegalArgumentException("a weight is a number");
        }
        if (balance < 0 || balance > 1 || !(density > 0 && density <= 1)) {
            throw new IllegalArgumentException("the balance is 0 to 1 and the share kept is above 0, up to 1");
        }
        List<Path> folders = new ArrayList<>();
        for (String m : models) {
            folders.add(folder(m));
        }
        Path baseFolder = null;
        if (w.base()) {
            if (base == null || base.isBlank()) {
                throw new IllegalArgumentException(w.label() + " needs the base model the others were tuned from");
            }
            if (models.contains(base)) {
                throw new IllegalArgumentException("the base model is not one of the models to merge");
            }
            baseFolder = folder(base);
        }
        Path target = folders.getFirst().toAbsolutePath().getParent().resolve(name);
        if (Files.exists(target)) {
            throw new IOException(name + " is there already");
        }
        double[] ws = new double[models.size()];
        for (int i = 0; i < ws.length; i++) {
            ws[i] = weights.isEmpty() ? (w.id().equals("linear") ? 1.0 / ws.length : 1) : weights.get(i);
        }
        ModelMerge.Method method = switch (w.id()) {
            case "linear" -> new ModelMerge.Linear(ws);
            case "slerp" -> new ModelMerge.Slerp(balance);
            case "task" -> new ModelMerge.TaskArithmetic(ws);
            case "ties" -> new ModelMerge.Ties(ws, density);
            default -> new ModelMerge.Dare(ws, density, 0);
        };
        Job started = new Job(name, w.id(), List.copyOf(models), target, System.currentTimeMillis(), "merging", null, 0);
        job = started;
        Path from = baseFolder;
        Thread.ofPlatform().daemon().name("merge-" + name).start(() -> run(started, method, from, folders));
        return state();
    }

    private Path folder(String model) throws IOException {
        Path p = local.location(model);
        if (!Files.isDirectory(p)) {
            throw new IOException(model + " is a single GGUF file: merging needs model folders with safetensors weights");
        }
        return p;
    }

    private void run(Job mine, ModelMerge.Method method, Path base, List<Path> folders) {
        Path part = mine.folder().resolveSibling(mine.folder().getFileName() + ".part");
        try {
            ModelMerge.merge(method, base, folders, part);
            Files.move(part, mine.folder(), StandardCopyOption.ATOMIC_MOVE);
            job = mine.to("done", null);
        } catch (IOException | RuntimeException | OutOfMemoryError e) {
            remove(part);
            Throwable cause = e instanceof UncheckedIOException u ? u.getCause() : e;
            job = mine.to("failed", e instanceof OutOfMemoryError ? "not enough memory for the largest tensor"
                    : String.valueOf(cause.getMessage()));
        }
    }

    private static void remove(Path part) {
        if (!Files.isDirectory(part)) {
            return;
        }
        try (Stream<Path> files = Files.walk(part)) {
            for (Path p : files.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(p);
            }
        } catch (IOException e) {
            // the half-written folder stays; its name says what it is
        }
    }
}
