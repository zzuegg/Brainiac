package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.LocalFineTune;
import io.github.zzuegg.brainiac.studio.train.TrainingRun;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Tools for the model work the studio does on this PC: the GPU, fine-tuning, quantizing,
 * measuring, training curves and building layer graphs.
 */
public final class MlTools {

    private final ChatService chat;
    private final TrainingService training;
    private volatile LayerCatalog catalog;
    private volatile ModelGraph graphs;

    /**
     * Creates the tools.
     *
     * @param chat the models
     * @param training the training runs
     */
    public MlTools(ChatService chat, TrainingService training) {
        this.chat = chat;
        this.training = training;
    }

    /**
     * Says what the GPU holds.
     *
     * @return the GPU's state
     */
    @Tools.Tool(name = "gpu_status", value = "Says what is loaded on this PC's GPU, how much memory is free, the "
            + "policy per kind of task, and what is waiting. Use it before loading a big model.")
    public Map<String, Object> gpuStatus() {
        io.github.zzuegg.brainiac.tensor.Device d;
        try {
            d = chat.localDevice().device();
        } catch (RuntimeException | LinkageError e) {
            d = null;
        }
        return chat.localDevice().residency().toJson(d);
    }

    /**
     * Fine-tunes a model.
     *
     * @param model the base model
     * @param data the data file
     * @param name the adapter's name
     * @param rank the LoRA rank
     * @param epochs passes over the data
     * @param learningRate the step size
     * @return the job
     */
    @Tools.Tool(name = "fine_tune", value = "Fine-tunes a chat model on this PC with LoRA on a data file of "
            + "conversations or prompt and answer pairs (JSONL). One fine-tune runs at a time; ml_jobs shows how it goes.")
    @Tier(value = Risk.COMPUTE, what = "fine-tunes a model on this PC, which keeps the GPU busy for a long time")
    public Map<String, Object> fineTune(@Tools.Param("a local chat model, such as 'local/qwen2.5-0.5b-instruct'") String model,
            @Tools.Param("the data file: a path the fine-tune page accepts, such as one in the datasets folder") String data,
            @Tools.Param("a name for the result") Optional<String> name,
            @Tools.Param("the LoRA rank, 8 if left out") Optional<Integer> rank,
            @Tools.Param("passes over the data, 3 if left out") Optional<Integer> epochs,
            @Tools.Param("the learning rate, 0.0002 if left out") Optional<Double> learningRate) {
        int r = rank.orElse(8);
        try {
            return chat.fineTune().start(local(model), name.orElse(""), data, new LocalFineTune.Settings(r, 2f * r,
                    epochs.orElse(3), learningRate.orElse(2e-4).floatValue(), 512, 1));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    /**
     * Quantizes a model.
     *
     * @param model the model
     * @param size the size
     * @return the job
     */
    @Tools.Tool(name = "quantize_model", value = "Makes a smaller copy of a local chat model, such as Q4_K_M, so it "
            + "needs less memory. ml_jobs shows how it goes.")
    @Tier(value = Risk.COMPUTE, what = "makes a smaller copy of a model on this PC")
    public Map<String, Object> quantizeModel(@Tools.Param("a local chat model") String model,
            @Tools.Param("Q8_0, Q6_K, Q5_K_M, Q4_K_M or another size; Q4_K_M if left out") Optional<String> size) {
        try {
            return chat.quantizer().start(local(model), size.orElse("Q4_K_M"));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    /**
     * Measures a model.
     *
     * @param model the model
     * @param task the benchmark
     * @param data the data
     * @param limit how many questions
     * @param speed whether to measure speed
     * @return the job
     */
    @Tools.Tool(name = "evaluate_model", value = "Measures a local chat model: its score on a task and data set, or "
            + "its speed. ml_jobs shows the result.")
    @Tier(value = Risk.COMPUTE, what = "measures a model on this PC")
    public Map<String, Object> evaluateModel(@Tools.Param("a local chat model") String model,
            @Tools.Param("the task, as the evaluation page names it; empty when measuring speed") Optional<String> task,
            @Tools.Param("the data file for the task") Optional<String> data,
            @Tools.Param("how many questions, 100 if left out") Optional<Integer> limit,
            @Tools.Param("true to measure speed instead of a score") Optional<Boolean> speed) {
        try {
            if (speed.orElse(false)) {
                return chat.evaluation().startSpeed(local(model));
            }
            return chat.evaluation().start(local(model), task.orElse(""), data.orElse(""), limit.orElse(100), 0);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    /**
     * Says how the model jobs go.
     *
     * @return each job's state
     */
    @Tools.Tool(name = "ml_jobs", value = "Says how fine-tuning, quantizing and measuring on this PC go, with their "
            + "results.")
    @Tier(value = Risk.READ, resultChars = 12_000)
    public Map<String, Object> mlJobs() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("fineTune", chat.fineTune().state());
        out.put("quantize", chat.quantizer().state());
        out.put("evaluate", chat.evaluation().state());
        return out;
    }

    /**
     * Reads a run's curve.
     *
     * @param id the run
     * @return its losses per epoch
     */
    @Tools.Tool(name = "training_curve", value = "Reads a training run's loss and accuracy per epoch, to see whether it "
            + "learns, overfits or has stalled.")
    @Tier(value = Risk.READ, resultChars = 12_000)
    public Map<String, Object> trainingCurve(@Tools.Param("the run's id from list_training_runs") String id) {
        for (TrainingRun r : training.runs()) {
            if (r.id().equals(id)) {
                Map<String, Object> full = r.toJson(true);
                Map<String, Object> out = new LinkedHashMap<>();
                for (String k : List.of("id", "name", "state", "epoch", "epochs", "loss", "bestEpoch", "history")) {
                    if (full.get(k) != null) {
                        out.put(k, full.get(k));
                    }
                }
                return out;
            }
        }
        throw new IllegalArgumentException("there is no run " + id);
    }

    /**
     * Lists the building blocks.
     *
     * @param search words in a block's name
     * @return the blocks
     */
    @Tools.Tool(name = "list_layers", value = "Lists the blocks a model can be built from in the builder, with their "
            + "parameters: the 'key' of each goes in a layer node's 'entry'.")
    @Tier(value = Risk.READ, resultChars = 16_000)
    public List<Map<String, Object>> listLayers(@Tools.Param("words in a block's name or group, such as 'conv'") Optional<String> search) {
        List<Map<String, Object>> out = new ArrayList<>();
        String q = search.orElse("").toLowerCase(Locale.ROOT);
        for (LayerCatalog.Entry e : catalog().entries()) {
            if (!q.isBlank() && !(e.name() + " " + e.group() + " " + e.key()).toLowerCase(Locale.ROOT).contains(q)) {
                continue;
            }
            out.add(LayerCatalog.describe(e));
            if (out.size() == 40) {
                break;
            }
        }
        return out;
    }

    /**
     * Checks a graph.
     *
     * @param graph the graph
     * @return each block's shape, or why it does not fit
     */
    @Tools.Tool(name = "check_shapes", value = "Checks a model graph: each block's output shape, or why its input does "
            + "not fit. A graph is {\"nodes\": [...], \"output\": \"id\"}; a node is {\"id\", \"op\": \"input\", "
            + "\"shape\"}, {\"id\", \"op\": \"layer\", \"entry\" (a key from list_layers), \"args\", \"options\", "
            + "\"inputs\"}, or {\"id\", \"op\": \"add\" | \"multiply\" | \"concat\", \"inputs\"}.")
    public Map<String, Object> checkShapes(@Tools.Param("the graph") Map<String, Object> graph) {
        return model().check(graph).toJson();
    }

    /**
     * Builds a graph.
     *
     * @param name the model's name
     * @param graph the graph
     * @return the model
     */
    @Tools.Tool(name = "build_model", value = "Builds a model from a graph that checks out, and keeps it in the "
            + "studio's models and graphs, where training can use it.")
    @Tier(value = Risk.CREATE, what = "builds a model and keeps it in the studio")
    public Map<String, Object> buildModel(@Tools.Param("1 to 64 letters, digits, spaces, dots, dashes or underscores") String name,
            @Tools.Param("the graph, as check_shapes takes it") Map<String, Object> graph) {
        if (name == null || !name.matches("[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}") || name.contains("..")) {
            throw new IllegalArgumentException("a model name is 1 to 64 letters, digits, spaces, dots, dashes or underscores");
        }
        ModelGraph.Check check = model().check(graph);
        if (!check.ok()) {
            throw new IllegalArgumentException("the graph does not check out: " + check.errors());
        }
        Path models = chat.folder().resolve("models");
        Path graphsDir = chat.folder().resolve("graphs");
        if (Files.exists(graphsDir.resolve(name + ".json"))) {
            throw new IllegalArgumentException("a model named " + name + " exists already; pick another name");
        }
        try {
            var built = model().build(graph, 42);
            Files.createDirectories(models);
            Files.createDirectories(graphsDir);
            Path tmp = models.resolve(name + ".brainiac.tmp");
            String saved = null;
            try {
                built.save(tmp);
                Files.move(tmp, models.resolve(name + ".brainiac"), StandardCopyOption.REPLACE_EXISTING);
                saved = models.resolve(name + ".brainiac").toString();
            } catch (IllegalStateException | UnsupportedOperationException cannot) {
                Files.deleteIfExists(tmp);
            }
            Map<String, Object> doc = new LinkedHashMap<>(graph);
            doc.put("name", name);
            doc.put("updated", System.currentTimeMillis());
            Files.writeString(graphsDir.resolve(name + ".json"), Json.write(doc), StandardCharsets.UTF_8);
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("name", name);
            out.put("parameters", built.parameterCount());
            out.put("summary", built.summary());
            if (saved != null) {
                out.put("file", saved);
            }
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    private LayerCatalog catalog() {
        if (catalog == null) {
            synchronized (this) {
                if (catalog == null) {
                    LayerCatalog c = LayerCatalog.discover();
                    graphs = new ModelGraph(c);
                    catalog = c;
                }
            }
        }
        return catalog;
    }

    private ModelGraph model() {
        catalog();
        return graphs;
    }

    private static String local(String model) {
        return model != null && model.startsWith("local/") ? model.substring(6) : model;
    }
}
