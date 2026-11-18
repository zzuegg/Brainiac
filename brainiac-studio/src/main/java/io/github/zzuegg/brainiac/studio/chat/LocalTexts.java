package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.Bart;
import io.github.zzuegg.brainiac.llm.T5;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * The models on this PC that turn one text into another: BART and mBART (summaries), Marian
 * (translation) and T5 (whatever its task prefix asks for). They are found in the same places as
 * the language models; one is kept loaded, on the best device, with 32-bit weights.
 */
public final class LocalTexts {

    /** The medium these models are listed under. */
    public static final String MEDIUM = "rewrite";

    private static final Set<String> BART = Set.of("bart", "mbart", "marian");
    private static final Set<String> T5S = Set.of("t5", "mt5", "umt5");

    /**
     * A task a model was trained for, from its {@code task_specific_params}.
     *
     * @param name the task's name, such as {@code summarization} or {@code translation_en_to_de}
     * @param prefix what goes before the text, such as {@code "summarize: "}, or an empty string
     * @param maxTokens the longest answer its settings ask for, or 0 if they do not say
     */
    public record Task(String name, String prefix, int maxTokens) {
    }

    /**
     * What a model wrote.
     *
     * @param text the answer
     * @param task the task it did, or an empty string
     */
    public record Rewritten(String text, String task) {
    }

    // the loaded model: one of the two kinds
    private record Loaded(Path path, Device device, Bart bart, T5 t5, PretrainedTokenizer tokenizer) {

        String generate(String text, int maxTokens) {
            return bart != null ? bart.generate(tokenizer, text, maxTokens) : t5.generate(tokenizer, text, maxTokens);
        }
    }

    private final Supplier<List<Path>> places;
    private final LocalDevice where;
    private final Object running = new Object();
    private Loaded loaded;

    /**
     * Finds models in some places and runs them on the device chosen.
     *
     * @param places the model folders
     * @param where the device this PC's models run on
     */
    LocalTexts(Supplier<List<Path>> places, LocalDevice where) {
        this.places = places;
        this.where = where;
    }

    /**
     * Lists the models.
     *
     * @return the models, with the medium {@value #MEDIUM}
     */
    List<ModelInfo> models() {
        List<ModelInfo> out = new ArrayList<>();
        for (Map.Entry<String, Path> e : find().entrySet()) {
            String type = type(e.getValue());
            String what = switch (type) {
                case "marian" -> "Marian translation";
                case "bart", "mbart" -> type.equals("bart") ? "BART text to text" : "mBART text to text";
                default -> "T5 text to text";
            };
            List<Task> tasks = tasks(e.getValue());
            if (!tasks.isEmpty()) {
                what += " · " + String.join(", ", tasks.stream().map(Task::name).toList());
            }
            out.add(new ModelInfo("local", e.getKey(), e.getKey(), List.of(MEDIUM), 0, what));
        }
        return out;
    }

    /**
     * Says whether a name is one of these models.
     *
     * @param id the name
     * @return true if it is
     */
    boolean has(String id) {
        return find().containsKey(id);
    }

    /**
     * Returns the tasks a model names in its configuration.
     *
     * @param id the model's name
     * @return the tasks, maybe none
     * @throws IOException if there is no such model
     */
    List<Task> tasks(String id) throws IOException {
        return tasks(path(id));
    }

    /**
     * Turns a text into another: a summary, a translation, or what a task prefix asks for.
     *
     * @param id the model's name
     * @param text the input
     * @param task one of the model's tasks, whose prefix goes before the text, or blank for none
     * @param maxTokens the most tokens to write, or 0 for the task's own limit (else 200)
     * @return the answer
     * @throws IOException if there is no such model, or it cannot be loaded
     */
    Rewritten rewrite(String id, String text, String task, int maxTokens) throws IOException {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("give a text to rewrite");
        }
        Path p = path(id);
        Task chosen = null;
        if (task != null && !task.isBlank()) {
            for (Task t : tasks(p)) {
                if (t.name().equals(task)) {
                    chosen = t;
                }
            }
            if (chosen == null) {
                throw new IllegalArgumentException(id + " has no task " + task + "; it has "
                        + tasks(p).stream().map(Task::name).toList());
            }
        }
        int limit = maxTokens > 0 ? maxTokens : chosen != null && chosen.maxTokens() > 0 ? chosen.maxTokens() : 200;
        String input = (chosen == null ? "" : chosen.prefix()) + text.strip();
        synchronized (running) {
            Device device = where.device();
            if (loaded == null || !loaded.path().equals(p) || !loaded.device().equals(device)) {
                loaded = null;   // let the old weights go first
                PretrainedTokenizer tokenizer = PretrainedTokenizer.load(p);
                loaded = BART.contains(type(p))
                        ? new Loaded(p, device, Bart.load(p, device, DType.FLOAT32), null, tokenizer)
                        : new Loaded(p, device, null, T5.load(p, device, DType.FLOAT32), tokenizer);
            }
            return new Rewritten(loaded.generate(input, limit).strip(), chosen == null ? "" : chosen.name());
        }
    }

    private Path path(String id) throws IOException {
        Path p = find().get(id);
        if (p == null) {
            throw new IOException("no text-to-text model named " + id + " on this PC");
        }
        return p;
    }

    // name -> folder; the first place that has a name wins
    private Map<String, Path> find() {
        Map<String, Path> found = new LinkedHashMap<>();
        for (Path place : places.get()) {
            if (isModel(place)) {
                found.putIfAbsent(place.getFileName().toString(), place);
                continue;
            }
            if (!Files.isDirectory(place)) {
                continue;
            }
            try (Stream<Path> s = Files.list(place)) {
                s.sorted().filter(LocalTexts::isModel).forEach(p -> found.putIfAbsent(p.getFileName().toString(), p));
            } catch (IOException e) {
                // unreadable: nothing here
            }
        }
        return found;
    }

    /**
     * Says whether a folder holds a model of these kinds, with a tokenizer it can read.
     *
     * @param p the folder
     * @return true if it does
     */
    static boolean isModel(Path p) {
        if (!Files.isRegularFile(p.resolve("config.json"))
                || !(Files.isRegularFile(p.resolve("tokenizer.json")) || Files.isRegularFile(p.resolve("tokenizer.model")))) {
            return false;
        }
        String type = type(p);
        return BART.contains(type) || T5S.contains(type);
    }

    private static String type(Path p) {
        try {
            return Json.string(Json.object(Json.read(p.resolve("config.json"))), "model_type", "");
        } catch (IOException | RuntimeException e) {
            return "";
        }
    }

    private static List<Task> tasks(Path p) {
        List<Task> out = new ArrayList<>();
        try {
            Object params = Json.object(Json.read(p.resolve("config.json"))).get("task_specific_params");
            if (params instanceof Map<?, ?> m) {
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    Map<?, ?> t = e.getValue() instanceof Map<?, ?> v ? v : Map.of();
                    Object prefix = t.get("prefix");
                    Object max = t.get("max_length");
                    out.add(new Task(String.valueOf(e.getKey()), prefix instanceof String s ? s : "",
                            max instanceof Number n ? n.intValue() : 0));
                }
            }
        } catch (IOException | RuntimeException e) {
            // no readable tasks
        }
        return out;
    }
}
