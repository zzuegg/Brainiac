package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Agent;
import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import io.github.zzuegg.brainiac.studio.train.TrainingRun;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * What the studio can do, as tools for agents: the studio's own agents call them, and so can any
 * program holding the studio's token, through {@code /api/tools}. The schemas are generated from
 * these methods by {@link Tools#from}, so a new method here is a new tool.
 *
 * <p>Tools marked {@link Acts} change something, spend money or use the GPU; an agent running in
 * {@code ask} mode waits for the user's approval before calling them.
 */
public final class StudioTools {

    /** Marks a tool that changes something, spends money or keeps the machine busy. */
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    public @interface Acts {

        /** @return what calling it does, for the approval card */
        String value();
    }

    /**
     * A tool with whether it acts.
     *
     * @param tool the tool
     * @param acts what it does if it acts, else null
     */
    public record Entry(Agent.Tool tool, String acts) {

        /**
         * Describes it for the page and for API callers.
         *
         * @return {@code {name, description, parameters, acts?}}
         */
        public Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("name", tool.name());
            m.put("description", tool.description());
            m.put("parameters", tool.parameters());
            if (acts != null) {
                m.put("acts", acts);
            }
            return m;
        }
    }

    private static final int ANSWER_TOKENS = 1024;

    private final ChatService chat;
    private final TrainingService training;

    /**
     * Creates the tools over the studio's services.
     *
     * @param chat the models and the Hugging Face Hub
     * @param training the training runs
     */
    public StudioTools(ChatService chat, TrainingService training) {
        this.chat = chat;
        this.training = training;
    }

    /**
     * Returns every tool, ordered by name.
     *
     * @return the tools
     */
    public List<Entry> entries() {
        Map<String, String> acts = new LinkedHashMap<>();
        for (Method m : StudioTools.class.getMethods()) {
            Tools.Tool t = m.getAnnotation(Tools.Tool.class);
            Acts a = m.getAnnotation(Acts.class);
            if (t != null && a != null) {
                acts.put(t.name().isEmpty() ? m.getName() : t.name(), a.value());
            }
        }
        List<Entry> out = new ArrayList<>();
        for (Agent.Tool t : Tools.from(this)) {
            out.add(new Entry(t, acts.get(t.name())));
        }
        return out;
    }

    // ---------------------------------------------------------------- models

    /**
     * Lists the chat models.
     *
     * @return {@code [{model, provider, name, local, media}]}
     */
    @Tools.Tool(name = "list_models", value = "Lists the models the studio can chat with: those on this PC and those of "
            + "providers the user has a key for. Use the 'model' value with ask_model.")
    public List<Map<String, Object>> listModels() {
        Map<String, Object> d = chat.describe(false);
        Map<String, Boolean> local = new LinkedHashMap<>();
        for (Object p : Json.array(d.get("providers"))) {
            Map<String, Object> m = Json.object(p);
            local.put(String.valueOf(m.get("id")), Boolean.TRUE.equals(m.get("local")));
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object o : Json.array(d.get("models"))) {
            Map<String, Object> m = Json.object(o);
            if (Boolean.FALSE.equals(m.get("shown"))) {
                continue;
            }
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("model", m.get("provider") + "/" + m.get("id"));
            r.put("name", m.get("name"));
            r.put("local", local.getOrDefault(String.valueOf(m.get("provider")), false));
            r.put("media", m.get("media"));
            out.add(r);
        }
        return out;
    }

    /**
     * Asks another model.
     *
     * @param model the model key
     * @param prompt the question
     * @param system instructions
     * @return the answer
     */
    @Tools.Tool(name = "ask_model", value = "Sends one question to a chat model and returns its answer. Remote models "
            + "use the user's key and may cost money.")
    @Acts("sends a question to a model, which may use your key and cost money")
    public String askModel(@Tools.Param("a model from list_models, or 'auto'") String model,
            @Tools.Param("the question") String prompt,
            @Tools.Param("instructions for the model") Optional<String> system) {
        StringBuilder answer = new StringBuilder();
        try {
            chat.chat(model, system.orElse(""), List.of(new ChatMessage("user", prompt)), ANSWER_TOKENS, null,
                    m -> { }, answer::append);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted");
        }
        return answer.toString();
    }

    /**
     * Summarizes or translates with a text-to-text model.
     *
     * @param text the input
     * @param model the model key
     * @param task the model's task
     * @return {@code {model, text, task}}
     */
    @Tools.Tool(name = "rewrite_text", value = "Summarizes, translates or otherwise rewrites a text with a BART, Marian "
            + "or T5 model on this PC. list_models shows them with the media 'rewrite'.")
    @Acts("runs a text-to-text model on this PC")
    public Map<String, Object> rewriteText(@Tools.Param("the text") String text,
            @Tools.Param("a model from list_models with the media 'rewrite', or 'auto'") Optional<String> model,
            @Tools.Param("one of the model's tasks, such as 'summarization' or 'translation_en_to_de'") Optional<String> task) {
        ChatService.Rewritten r = call(() -> chat.rewrite(model.orElse(ChatService.AUTO), text, task.orElse(""), 0));
        return Map.of("model", r.model(), "text", r.text(), "task", r.task());
    }

    // ---------------------------------------------------------------- making media

    /**
     * Draws pictures with the project's image route.
     *
     * @param prompt what they show
     * @param aspect the shape
     * @param count how many
     * @return the model used and where the pictures were saved
     */
    @Tools.Tool(name = "draw_image", value = "Draws pictures and saves them. 'auto' uses the project's image route: "
            + "its model and runner, on this PC or a remote machine.")
    @Acts("draws pictures, which uses the GPU or a paid provider")
    public Map<String, Object> drawImage(@Tools.Param("what the picture shows") String prompt,
            @Tools.Param("a model key, or 'auto' for the project's route") Optional<String> model,
            @Tools.Param("1:1, 4:3, 3:4, 16:9 or 9:16") Optional<String> aspect,
            @Tools.Param("how many, 1 to 4") Optional<Integer> count) {
        try {
            return made(chat.images(model.orElse(ChatService.AUTO), prompt, aspect.orElse("1:1"),
                    Math.clamp(count.orElse(1), 1, 4), null));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted");
        }
    }

    /**
     * Speaks text with the project's speech route.
     *
     * @param text the words
     * @param voice the voice
     * @return the model used and where the audio was saved
     */
    @Tools.Tool(name = "make_speech", value = "Speaks text and saves the audio. 'auto' uses the project's speech route: "
            + "its model and runner, on this PC or a remote machine.")
    @Acts("makes audio, which uses the GPU or a paid provider")
    public Map<String, Object> makeSpeech(@Tools.Param("the words to say") String text,
            @Tools.Param("a model key, or 'auto' for the project's route") Optional<String> model,
            @Tools.Param("the voice's name, if the model has voices") Optional<String> voice) {
        try {
            return made(chat.speech(model.orElse(ChatService.AUTO), text, voice.orElse(""), 1, "", "wav", null));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted");
        }
    }

    private static Map<String, Object> made(ChatService.Generated g) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model", g.model().key());
        m.put("files", g.files().stream().map(Path::toString).toList());
        return m;
    }

    // ---------------------------------------------------------------- the project's files

    /**
     * Lists files in the project's folder.
     *
     * @param path a folder inside it
     * @return the entries
     */
    @Tools.Tool(name = "list_project_files", value = "Lists the files and folders in the project's folder, or in a "
            + "folder inside it. Agents can only read inside their project's folder.")
    public Map<String, Object> listProjectFiles(@Tools.Param("a folder inside the project's folder; empty for the top")
            Optional<String> path) {
        Path dir = project().inFolder(path.orElse(""));
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException(path.orElse("") + " is not a folder");
        }
        List<Map<String, Object>> out = new ArrayList<>();
        try (java.util.stream.Stream<Path> s = Files.list(dir)) {
            for (Path p : s.sorted().limit(500).toList()) {
                Map<String, Object> e = new LinkedHashMap<>();
                e.put("name", p.getFileName().toString());
                e.put("folder", Files.isDirectory(p));
                if (Files.isRegularFile(p)) {
                    e.put("bytes", Files.size(p));
                }
                out.add(e);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return Map.of("folder", dir.toString(), "entries", out);
    }

    /**
     * Reads a text file in the project's folder.
     *
     * @param path the file inside it
     * @return the text, cut at {@value #READ_CHARS} characters
     */
    @Tools.Tool(name = "read_project_file", value = "Reads a text file in the project's folder (the first "
            + READ_CHARS + " characters).")
    public String readProjectFile(@Tools.Param("the file, inside the project's folder") String path) {
        Path f = project().inFolder(path);
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no file " + path + " in the project's folder");
        }
        try (java.io.Reader r = Files.newBufferedReader(f, java.nio.charset.StandardCharsets.UTF_8)) {
            char[] buf = new char[READ_CHARS];
            int n = 0;
            for (int k; n < buf.length && (k = r.read(buf, n, buf.length - n)) > 0; ) {
                n += k;
            }
            return new String(buf, 0, n);
        } catch (java.nio.charset.MalformedInputException e) {
            throw new IllegalArgumentException(path + " is not a text file");
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    private static final int READ_CHARS = 20_000;

    // the project the call belongs to
    private static Project project() {
        if (ChatService.SCOPE.isBound() && ChatService.SCOPE.get() instanceof Project p) {
            return p;
        }
        throw new IllegalArgumentException("this call belongs to no project");
    }

    // ---------------------------------------------------------------- Hugging Face

    /**
     * Searches the Hub.
     *
     * @param query words
     * @return the runnable models
     */
    @Tools.Tool(name = "search_hub", value = "Searches Hugging Face for text models this studio can run, most "
            + "downloaded first.")
    public Map<String, Object> searchHub(@Tools.Param("words in the model's name, such as 'qwen 0.5b'") String query) {
        Map<String, Object> found = call(() -> chat.hub().search(query));
        List<Map<String, Object>> models = new ArrayList<>();
        for (Object o : Json.array(found.get("models"))) {
            Map<String, Object> m = Json.object(o);
            if (models.size() == 10) {
                break;
            }
            Map<String, Object> r = new LinkedHashMap<>();
            for (String k : List.of("id", "family", "parameters", "downloads", "gguf", "safetensors", "gated")) {
                if (m.get(k) != null) {
                    r.put(k, m.get(k));
                }
            }
            models.add(r);
        }
        return Map.of("models", models, "hidden", found.getOrDefault("hidden", 0));
    }

    /**
     * Lists what a repository offers.
     *
     * @param repo the repository
     * @return the options
     */
    @Tools.Tool(name = "hub_files", value = "Lists the files of a Hugging Face model that can be downloaded, and "
            + "whether each would run here and why not.")
    public List<Map<String, Object>> hubFiles(@Tools.Param("the repository, such as 'Qwen/Qwen2.5-0.5B-Instruct-GGUF'")
            String repo) {
        Map<String, Object> found = call(() -> chat.hub().files(repo));
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object o : Json.array(found.get("options"))) {
            Map<String, Object> m = Json.object(o);
            Map<String, Object> r = new LinkedHashMap<>();
            for (String k : List.of("id", "kind", "name", "size", "quant", "supported", "reason", "have")) {
                if (m.get(k) != null) {
                    r.put(k, m.get(k));
                }
            }
            out.add(r);
        }
        return out;
    }

    /**
     * Starts a download.
     *
     * @param repo the repository
     * @param option the file or folder
     * @return the download
     */
    @Tools.Tool(name = "download_model", value = "Starts downloading one option from hub_files into the studio's "
            + "downloads folder; the model is usable when the download is done.")
    @Acts("downloads a model from Hugging Face to this PC")
    public Map<String, Object> downloadModel(@Tools.Param("the repository") String repo,
            @Tools.Param("the option's id from hub_files") String option) {
        return call(() -> chat.hub().download(repo, option));
    }

    /**
     * Lists downloads.
     *
     * @return the downloads
     */
    @Tools.Tool(name = "list_downloads", value = "Lists model downloads with their progress and state.")
    public List<Map<String, Object>> listDownloads() {
        return chat.hub().downloads();
    }

    // ---------------------------------------------------------------- training

    /**
     * Lists the models that can be trained.
     *
     * @return the models
     */
    @Tools.Tool(name = "list_trainable_models", value = "Lists the models built in the studio that can be trained, "
            + "with their input and output shapes.")
    public List<Map<String, Object>> listTrainableModels() {
        return call(training::models);
    }

    /**
     * Lists the data sets.
     *
     * @return the built-in sets and the files
     */
    @Tools.Tool(name = "list_datasets", value = "Lists the built-in data sets and the data files in the studio's "
            + "datasets folder.")
    public Map<String, Object> listDatasets() {
        return call(() -> training.data().list());
    }

    /**
     * Checks a run.
     *
     * @param model the model
     * @param data the data
     * @param label the label column
     * @return whether they fit
     */
    @Tools.Tool(name = "check_training", value = "Checks whether a model and a data set fit together, before "
            + "training.")
    public Map<String, Object> checkTraining(@Tools.Param("a model from list_trainable_models") String model,
            @Tools.Param("a built-in set's name or a file from list_datasets") String data,
            @Tools.Param("for a CSV file, the column to predict") Optional<String> label) {
        return training.check(spec(model, data, label, Optional.empty(), Optional.empty()));
    }

    /**
     * Starts a run.
     *
     * @param model the model
     * @param data the data
     * @param label the label column
     * @param epochs passes over the data
     * @param batchSize samples per step
     * @return the run
     */
    @Tools.Tool(name = "start_training", value = "Trains a model on a data set with the default optimizer and loss. "
            + "Runs one at a time; later ones wait.")
    @Acts("starts training on this PC, which keeps the GPU or CPU busy")
    public Map<String, Object> startTraining(@Tools.Param("a model from list_trainable_models") String model,
            @Tools.Param("a built-in set's name or a file from list_datasets") String data,
            @Tools.Param("for a CSV file, the column to predict") Optional<String> label,
            @Tools.Param("passes over the data, 10 if left out") Optional<Integer> epochs,
            @Tools.Param("samples per step, 32 if left out") Optional<Integer> batchSize) {
        return summary(training.start(spec(model, data, label, epochs, batchSize)));
    }

    /**
     * Lists runs.
     *
     * @return the runs
     */
    @Tools.Tool(name = "list_training_runs", value = "Lists training runs, newest first, with their state and "
            + "latest loss.")
    public List<Map<String, Object>> listTrainingRuns() {
        return training.runs().stream().limit(20).map(StudioTools::summary).toList();
    }

    /**
     * Stops a run.
     *
     * @param id the run
     * @return whether there was one
     */
    @Tools.Tool(name = "stop_training", value = "Stops a training run; what it learned so far is saved.")
    @Acts("stops a training run")
    public String stopTraining(@Tools.Param("the run's id from list_training_runs") String id) {
        return training.stop(id) ? "stopping run " + id : "there is no run " + id;
    }

    private static Map<String, Object> summary(TrainingRun run) {
        Map<String, Object> full = run.toJson(false);
        Map<String, Object> r = new LinkedHashMap<>();
        for (String k : List.of("id", "name", "state", "host", "data", "epoch", "epochs", "loss", "bestEpoch",
                "error", "output")) {
            if (full.get(k) != null) {
                r.put(k, full.get(k));
            }
        }
        return r;
    }

    private Map<String, Object> spec(String model, String data, Optional<String> label, Optional<Integer> epochs,
            Optional<Integer> batchSize) {
        Map<String, Object> d = new LinkedHashMap<>();
        String lower = data.toLowerCase(Locale.ROOT);
        boolean builtin = io.github.zzuegg.brainiac.studio.train.DataSources.BUILTINS.stream()
                .anyMatch(b -> b.name().equals(data));
        if (builtin) {
            d.put("kind", "builtin");
            d.put("name", data);
        } else if (lower.endsWith(".csv") || lower.endsWith(".tsv")) {
            d.put("kind", "csv");
            d.put("file", data);
            if (lower.endsWith(".tsv")) {
                d.put("delimiter", "\t");
            }
            label.ifPresent(l -> d.put("label", l));
        } else if (lower.contains("images-idx") || lower.contains("images.idx")) {
            d.put("kind", "idx");
            d.put("images", data);
            d.put("labels", data.replace("images-idx3", "labels-idx1").replace("images.idx3", "labels.idx1"));
        } else {
            d.put("kind", "images");
            d.put("folder", data);
        }
        Map<String, Object> spec = new LinkedHashMap<>();
        spec.put("model", model);
        spec.put("data", d);
        spec.put("epochs", epochs.orElse(10));
        spec.put("batchSize", batchSize.orElse(32));
        return spec;
    }

    @FunctionalInterface
    private interface Io<T> {
        T get() throws IOException;
    }

    private static <T> T call(Io<T> f) {
        try {
            return f.get();
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }
}
