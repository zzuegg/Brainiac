package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.llm.InstructionData;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Teaches one of this PC's chat models from examples with LoRA: small adapters are trained
 * while the model's own weights stay as they are. What comes out is the adapter, in the form
 * Hugging Face PEFT reads, and a copy of the model with the adapter merged in, a GGUF file
 * that shows up in the model list. One model is taught at a time; the page asks how it goes.
 *
 * <p>The examples are JSON lines: {@code {"messages": [{"role": "user", "content": "..."},
 * {"role": "assistant", "content": "..."}]}}, or {@code {"prompt": "...", "completion":
 * "..."}}, or {@code {"instruction": "...", "input": "...", "output": "..."}}. Only the
 * assistant's replies are learned.
 */
public final class LocalFineTune {

    /**
     * How to teach.
     *
     * @param rank the adapters' inner size, 1 to 256
     * @param alpha their scale; twice the rank is usual
     * @param epochs how often every example is seen, 1 to 100
     * @param learningRate the step size, such as 2e-4
     * @param length the most tokens of an example that are read, 16 to 8192
     * @param rows the rows per step, 1 to 64
     */
    public record Settings(int rank, float alpha, int epochs, float learningRate, int length, int rows) {

        /** Checks the settings. */
        public Settings {
            if (rank < 1 || rank > 256) {
                throw new IllegalArgumentException("choose a rank from 1 to 256");
            }
            if (!(alpha > 0)) {
                throw new IllegalArgumentException("alpha must be above 0");
            }
            if (epochs < 1 || epochs > 100) {
                throw new IllegalArgumentException("choose 1 to 100 epochs");
            }
            if (!(learningRate > 0 && learningRate <= 1)) {
                throw new IllegalArgumentException("choose a learning rate above 0, such as 0.0002");
            }
            if (length < 16 || length > 8192) {
                throw new IllegalArgumentException("choose an example length from 16 to 8192 tokens");
            }
            if (rows < 1 || rows > 64) {
                throw new IllegalArgumentException("choose 1 to 64 rows a step");
            }
        }
    }

    private static final int MAX_POINTS = 400;

    /** One run, changed only by its own thread except for the stop flag. */
    private static final class Job {
        final String model;
        final String name;
        final Settings settings;
        final Path adapter;
        final Path merged;
        final long started = System.currentTimeMillis();
        final List<double[]> points = new ArrayList<>();   // step, loss
        volatile String state = "loading";
        volatile String error;
        volatile String note;
        volatile int examples;
        volatile int step;
        volatile int steps;
        volatile long ended;
        volatile boolean stop;
        volatile boolean wroteMerged;
        volatile boolean wroteAdapter;

        Job(String model, String name, Settings settings, Path adapter, Path merged) {
            this.model = model;
            this.name = name;
            this.settings = settings;
            this.adapter = adapter;
            this.merged = merged;
        }

        boolean running() {
            return state.equals("loading") || state.equals("training") || state.equals("saving");
        }
    }

    private final LocalProvider local;
    private volatile Job job;

    /**
     * A teacher for a provider's models.
     *
     * @param local this PC's models
     */
    public LocalFineTune(LocalProvider local) {
        this.local = local;
    }

    /**
     * Returns whether a model is being taught.
     *
     * @return whether one is
     */
    public boolean running() {
        Job j = job;
        return j != null && j.running();
    }

    /**
     * Describes the run under way or the last one, for the page.
     *
     * @return {@code {job?: {model, name, state, step, steps, examples, loss?, losses: [[step,
     *     loss]...], seconds, adapter?, merged?, error?, note?, ...settings}}}; a job's state is
     *     {@code loading}, {@code training}, {@code saving}, {@code done}, {@code stopped} or
     *     {@code failed}
     */
    public Map<String, Object> state() {
        Map<String, Object> out = new LinkedHashMap<>();
        Job j = job;
        if (j == null) {
            return out;
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model", j.model);
        m.put("name", j.name);
        m.put("state", j.state);
        m.put("step", j.step);
        m.put("steps", j.steps);
        m.put("examples", j.examples);
        m.put("rank", j.settings.rank());
        m.put("epochs", j.settings.epochs());
        List<Object> losses = new ArrayList<>();
        synchronized (j.points) {
            for (double[] p : j.points) {
                losses.add(List.of((int) p[0], p[1]));
            }
            if (!j.points.isEmpty()) {
                m.put("loss", j.points.getLast()[1]);
            }
        }
        m.put("losses", losses);
        m.put("seconds", ((j.ended > 0 ? j.ended : System.currentTimeMillis()) - j.started) / 1000.0);
        if (j.wroteAdapter) {
            m.put("adapter", j.adapter.toString());
        }
        if (j.wroteMerged) {
            m.put("merged", j.merged.toString());
            m.put("mergedName", j.merged.getFileName().toString());
        }
        if (j.error != null) {
            m.put("error", j.error);
        }
        if (j.note != null) {
            m.put("note", j.note);
        }
        out.put("job", m);
        return out;
    }

    /**
     * Reads examples from JSON lines.
     *
     * @param tokenizer the model's tokenizer
     * @param data the lines
     * @return the examples that have something to learn
     * @throws IllegalArgumentException if a line is not an example, or none has a reply
     */
    static List<InstructionData.Example> examples(PretrainedTokenizer tokenizer, String data) {
        List<InstructionData.Example> out = new ArrayList<>();
        boolean template = tokenizer.chatTemplate() != null;
        int n = 0;
        for (String line : data.split("\\R")) {
            n++;
            if (line.isBlank()) {
                continue;
            }
            Map<String, Object> row;
            try {
                row = Json.object(Json.parse(line));
            } catch (RuntimeException e) {
                throw new IllegalArgumentException("line " + n + " is not a JSON object: " + e.getMessage());
            }
            InstructionData.Example example;
            if (row.get("messages") instanceof List<?> list) {
                if (!template) {
                    throw new IllegalArgumentException("line " + n + " is a conversation, and this model has no chat template"
                            + " to write it with: use prompt and completion lines");
                }
                List<Map<String, Object>> messages = new ArrayList<>();
                for (Object o : list) {
                    Map<String, Object> m = Json.object(o);
                    messages.add(Map.of("role", Json.string(m, "role", "user"), "content", Json.string(m, "content", "")));
                }
                example = InstructionData.encode(tokenizer, messages);
            } else {
                String prompt = row.containsKey("instruction")
                        ? (Json.string(row, "instruction", "") + "\n\n" + Json.string(row, "input", "")).strip()
                        : Json.string(row, "prompt", "");
                String answer = Json.string(row, "completion", Json.string(row, "response", Json.string(row, "output", "")));
                if (prompt.isBlank() || answer.isBlank()) {
                    throw new IllegalArgumentException("line " + n + " needs messages, or a prompt with a completion,"
                            + " or an instruction with an output");
                }
                example = template
                        ? InstructionData.encode(tokenizer, List.of(Map.of("role", "user", "content", prompt),
                                Map.of("role", "assistant", "content", answer)))
                        : InstructionData.completion(tokenizer, prompt, answer);
            }
            if (example.trained() > 0) {
                out.add(example);
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("there is no example with a reply to learn");
        }
        return out;
    }

    /**
     * Starts teaching a chat model. The adapter is written next to the model as the folder
     * {@code <model>-<name>-lora}, the merged copy as {@code <model>-<name>.gguf}.
     *
     * @param model the model as the provider lists it
     * @param name what the taught model is called: letters, digits, dots, dashes and underscores
     * @param data the examples, as JSON lines
     * @param settings how to teach
     * @return the state, as {@link #state} gives it
     * @throws IOException if there is no such model, it looks at pictures, the name is taken, or
     *     another model is being taught
     */
    public synchronized Map<String, Object> start(String model, String name, String data, Settings settings)
            throws IOException {
        if (running()) {
            throw new IOException(job.model + " is being taught: wait for it to finish, or stop it");
        }
        if (name == null || !name.matches("[A-Za-z0-9._-]{1,40}")) {
            throw new IllegalArgumentException("name it with up to 40 letters, digits, dots, dashes or underscores");
        }
        if (data == null || data.isBlank()) {
            throw new IllegalArgumentException("choose a file of examples first");
        }
        Path source = local.location(model);
        if (LocalProvider.looksAtPictures(source)) {
            throw new IOException(model + " looks at pictures: the studio teaches text models only");
        }
        String stem = source.getFileName().toString().replaceFirst("(?i)\\.gguf$", "");
        Path folder = source.toAbsolutePath().getParent();
        Path adapter = folder.resolve(stem + "-" + name + "-lora");
        Path merged = folder.resolve(stem + "-" + name + ".gguf");
        if (Files.exists(adapter) || Files.exists(merged)) {
            throw new IOException("there is a " + stem + "-" + name + " already: choose another name");
        }
        Job started = new Job(model, name, settings, adapter, merged);
        job = started;
        Thread.ofPlatform().daemon().name("finetune-" + model).start(() -> run(started, source, data));
        return state();
    }

    /**
     * Asks the run under way to stop after its current step; what it learned so far is saved.
     *
     * @return whether a run was under way
     */
    public boolean stop() {
        Job j = job;
        if (j == null || !j.running()) {
            return false;
        }
        j.stop = true;
        return true;
    }

    private void run(Job mine, Path source, String data) {
        try {
            PretrainedTokenizer tokenizer;
            try {
                tokenizer = PretrainedTokenizer.load(source);
            } catch (RuntimeException e) {
                throw new IOException(mine.model + " has no tokenizer the studio can read: " + e.getMessage(), e);
            }
            List<InstructionData.Example> examples = new ArrayList<>(examples(tokenizer, data));
            mine.examples = examples.size();
            Settings s = mine.settings;
            Llama model = Llama.load(source, local.device().device(), DType.FLOAT32).checkpointActivations(true);
            List<Var> adapters = model.addLora(s.rank(), s.alpha(), Llama.attentionAndMlp(), 1);
            Optimizer optimizer = Optimizer.adamW(s.learningRate(), 0f);
            int perEpoch = InstructionData.pack(examples, s.rows(), s.length()).size();
            mine.steps = perEpoch * s.epochs();
            mine.state = "training";
            train:
            for (int epoch = 0; epoch < s.epochs(); epoch++) {
                Collections.shuffle(examples, new Random(17L + epoch));
                for (InstructionData.Batch batch : InstructionData.pack(examples, s.rows(), s.length())) {
                    if (mine.stop) {
                        break train;
                    }
                    Var loss = model.loss(batch);
                    loss.backward();
                    double value = loss.value().item();
                    optimizer.step(adapters);
                    if (!Double.isFinite(value)) {
                        throw new IOException("the loss left the numbers at step " + (mine.step + 1)
                                + ": try a smaller learning rate");
                    }
                    mine.step++;
                    record(mine, value);
                }
            }
            if (mine.step == 0) {
                mine.state = "stopped";
                return;
            }
            mine.state = "saving";
            model.saveLora(mine.adapter);
            mine.wroteAdapter = true;
            Path part = mine.merged.resolveSibling(mine.merged.getFileName() + ".part");
            try {
                model.mergeLora();
                model.saveGguf(part, tokenizer, true);
                Files.move(part, mine.merged, StandardCopyOption.ATOMIC_MOVE);
                mine.wroteMerged = true;
            } catch (RuntimeException e) {
                Files.deleteIfExists(part);
                // packed 4- and 8-bit weights cannot take an adapter in; the adapter itself is saved
                mine.note = "The adapter is saved, but no merged copy: " + e.getMessage()
                        + ". Teach a full-precision model (a safetensors folder or an F16 GGUF) to get one.";
            }
            mine.state = mine.stop ? "stopped" : "done";
        } catch (IOException | RuntimeException | OutOfMemoryError e) {
            mine.error = e instanceof OutOfMemoryError
                    ? "not enough memory: try shorter examples, fewer rows a step or a smaller model"
                    : String.valueOf(e.getMessage());
            mine.state = "failed";
        } finally {
            mine.ended = System.currentTimeMillis();
        }
    }

    // keeps the curve small: once it is full, every second point goes and later ones come half as often
    private static void record(Job j, double loss) {
        synchronized (j.points) {
            int stride = Math.max(1, Integer.highestOneBit(Math.max(1, j.steps / MAX_POINTS)));
            if (j.step % stride == 0 || j.step == 1 || j.step == j.steps) {
                j.points.add(new double[] {j.step, loss});
            }
        }
    }
}
