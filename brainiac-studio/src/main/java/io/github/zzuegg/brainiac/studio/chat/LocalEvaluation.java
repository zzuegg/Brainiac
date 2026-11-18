package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Benchmark;
import io.github.zzuegg.brainiac.llm.Generator;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Measures this PC's chat models: how many questions of a benchmark file one answers right,
 * and how fast it reads and writes. Questions are scored as EleutherAI's lm-evaluation-harness
 * scores them: questions with choices by which choice the model finds likeliest, questions with
 * one answer by what it writes. Results are kept, so models can be compared; one measurement
 * runs at a time, and the page asks how it goes.
 *
 * <p>A benchmark file is JSON lines in one of the common layouts: MMLU ({@code question},
 * {@code choices}, {@code answer} as an index or a letter), ARC ({@code question}, {@code
 * choices.text}, {@code choices.label}, {@code answerKey}), HellaSwag ({@code ctx}, {@code
 * endings}, {@code label}), GSM8K ({@code question}, {@code answer} ending in {@code ####
 * number}), or plain {@code question} and {@code answer}.
 */
public final class LocalEvaluation {

    private static final int KEPT = 200;
    private static final int CHUNK = 4;
    private static final String PASSAGE = "The lighthouse keeper climbed the spiral stairs every evening before"
            + " dusk, carrying a small brass can of oil and a cloth for the lens. From the gallery he could see the"
            + " fishing boats turning home, their sails catching the last of the light, and beyond them the long"
            + " grey line where the sea met the sky. He trimmed the wick, polished the glass until it shone, and"
            + " set the great lamp turning. Its beam swept across the water, over the rocks that had broken so"
            + " many hulls, and out to the ships that would pass in the night without ever knowing his name."
            + " When the work was done he sat by the window with a cup of tea and wrote in his log: the wind,"
            + " the tide, the boats he had counted, and anything else worth remembering about the day.\n\n"
            + "Describe the keeper's evening in your own words.";

    /** One measurement, changed only by its own thread except for the stop flag. */
    private static final class Run {
        final String id;
        final String model;
        final String task;
        final String kind;   // choice, answer or speed
        final long started;
        volatile String state = "loading";
        volatile String error;
        volatile int total;
        volatile int done;
        volatile int correct;
        volatile int normalized;
        volatile int shots;
        volatile double readPerSecond;
        volatile double writePerSecond;
        volatile long ended;
        volatile boolean stop;

        Run(String id, String model, String task, String kind, long started) {
            this.id = id;
            this.model = model;
            this.task = task;
            this.kind = kind;
            this.started = started;
        }

        boolean running() {
            return state.equals("loading") || state.equals("running");
        }

        Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("model", model);
            m.put("task", task);
            m.put("kind", kind);
            m.put("state", state);
            m.put("started", started);
            m.put("seconds", ((ended > 0 ? ended : System.currentTimeMillis()) - started) / 1000.0);
            if (kind.equals("speed")) {
                m.put("readPerSecond", readPerSecond);
                m.put("writePerSecond", writePerSecond);
            } else {
                m.put("total", total);
                m.put("done", done);
                m.put("correct", correct);
                m.put("shots", shots);
                if (kind.equals("choice")) {
                    m.put("normalized", normalized);
                }
                if (done > 0) {
                    double p = (double) correct / done;
                    m.put("accuracy", p);
                    m.put("stderr", Math.sqrt(p * (1 - p) / done));
                }
            }
            if (error != null) {
                m.put("error", error);
            }
            return m;
        }

        static Run of(Map<String, Object> m) {
            Run r = new Run(Json.string(m, "id", ""), Json.string(m, "model", ""), Json.string(m, "task", ""),
                    Json.string(m, "kind", "choice"), Json.integer(m, "started", 0));
            String state = Json.string(m, "state", "done");
            r.state = state.equals("loading") || state.equals("running") ? "stopped" : state;
            r.error = Json.string(m, "error", null);
            r.total = (int) Json.integer(m, "total", 0);
            r.done = (int) Json.integer(m, "done", 0);
            r.correct = (int) Json.integer(m, "correct", 0);
            r.normalized = (int) Json.integer(m, "normalized", 0);
            r.shots = (int) Json.integer(m, "shots", 0);
            r.readPerSecond = Json.number(m, "readPerSecond", 0);
            r.writePerSecond = Json.number(m, "writePerSecond", 0);
            r.ended = r.started + Math.round(Json.number(m, "seconds", 0) * 1000);
            return r;
        }
    }

    private final LocalProvider local;
    private final Path file;
    private final List<Run> runs = new ArrayList<>();   // newest first
    private long counter;

    /**
     * Measurements of a provider's models.
     *
     * @param local this PC's models
     * @param file where the results are kept, as JSON
     */
    public LocalEvaluation(LocalProvider local, Path file) {
        this.local = local;
        this.file = file;
        if (Files.isRegularFile(file)) {
            try {
                for (Object o : Json.array(Json.object(Json.read(file)).getOrDefault("runs", List.of()))) {
                    runs.add(Run.of(Json.object(o)));
                }
            } catch (IOException | RuntimeException e) {
                System.getLogger(LocalEvaluation.class.getName()).log(System.Logger.Level.WARNING,
                        "cannot read the kept measurements in " + file + ": " + e.getMessage());
            }
        }
    }

    /**
     * Returns whether a measurement is under way.
     *
     * @return whether one is
     */
    public synchronized boolean running() {
        return runs.stream().anyMatch(Run::running);
    }

    /**
     * Describes the measurements, newest first, for the page.
     *
     * @return {@code {running, runs: [{id, model, task, kind, state, seconds, total?, done?,
     *     correct?, accuracy?, stderr?, normalized?, shots?, readPerSecond?, writePerSecond?,
     *     error?}]}}; a run's state is {@code loading}, {@code running}, {@code done}, {@code
     *     stopped} or {@code failed}
     */
    public synchronized Map<String, Object> state() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("running", runs.stream().anyMatch(Run::running));
        out.put("runs", runs.stream().map(Run::toJson).toList());
        return out;
    }

    /**
     * Reads a benchmark file's questions.
     *
     * @param data the JSON lines
     * @return the questions, all with choices or all without
     * @throws IllegalArgumentException if a line is in none of the layouts, or the file mixes kinds
     */
    static List<Benchmark.Example> questions(String data) {
        List<Benchmark.Example> out = new ArrayList<>();
        int n = 0;
        for (String line : data.split("\\R")) {
            n++;
            if (line.isBlank()) {
                continue;
            }
            try {
                Map<String, Object> row = Json.object(Json.parse(line));
                out.add(question(row));
            } catch (RuntimeException e) {
                throw new IllegalArgumentException("line " + n + " is not a question the studio can read: " + e.getMessage());
            }
            if (out.getLast().choices().isEmpty() != out.getFirst().choices().isEmpty()) {
                throw new IllegalArgumentException("line " + n + " is another kind of question than the lines before it:"
                        + " keep questions with choices and questions with one answer in files of their own");
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("the file has no questions");
        }
        return out;
    }

    private static Benchmark.Example question(Map<String, Object> row) {
        if (row.containsKey("endings")) {
            return Benchmark.hellaswag(row);
        }
        if (row.get("choices") instanceof Map<?, ?>) {
            return Benchmark.arc(row);
        }
        if (row.get("choices") instanceof List<?> choices) {
            if (row.get("answer") instanceof String letter) {
                int index = letter.strip().length() == 1 ? Character.toUpperCase(letter.strip().charAt(0)) - 'A' : -1;
                if (index < 0 || index >= choices.size()) {
                    index = choices.stream().map(String::valueOf).toList().indexOf(letter);
                }
                if (index < 0) {
                    throw new IllegalArgumentException("the answer " + letter + " is not one of the choices");
                }
                Map<String, Object> copy = new LinkedHashMap<>(row);
                copy.put("answer", index);
                return Benchmark.mmlu(copy);
            }
            return Benchmark.mmlu(row);
        }
        String question = Json.string(row, "question", Json.string(row, "prompt", ""));
        Object given = row.get("answer");
        String answer = given == null ? "" : String.valueOf(given).strip();
        if (question.isBlank() || answer.isBlank()) {
            throw new IllegalArgumentException("it needs a question with choices, or a question with an answer");
        }
        if (answer.contains("####")) {
            return Benchmark.gsm8k(row);
        }
        return Benchmark.Example.open("Question: " + question.strip() + "\nAnswer:", answer, answer);
    }

    /**
     * Starts scoring a model on a benchmark file.
     *
     * @param model the model as the provider lists it
     * @param task what the file is called
     * @param data the questions, as JSON lines
     * @param limit the most questions to ask, or 0 for all
     * @param shots how many of the file's first questions are shown solved before each question
     *     (and not asked), 0 to 8
     * @return the state, as {@link #state} gives it
     * @throws IOException if there is no such model or another measurement is under way
     */
    public synchronized Map<String, Object> start(String model, String task, String data, int limit, int shots)
            throws IOException {
        busy();
        if (data == null || data.isBlank()) {
            throw new IllegalArgumentException("choose a file of questions first");
        }
        if (shots < 0 || shots > 8 || limit < 0) {
            throw new IllegalArgumentException("show 0 to 8 solved questions, and ask 0 (all) or more");
        }
        local.location(model);
        List<Benchmark.Example> all = questions(data);
        if (all.size() <= shots) {
            throw new IllegalArgumentException("the file has " + all.size() + " questions: too few to show " + shots
                    + " solved and still ask some");
        }
        List<Benchmark.Example> shown = List.copyOf(all.subList(0, shots));
        List<Benchmark.Example> asked = all.subList(shots, all.size());
        asked = List.copyOf(limit > 0 && limit < asked.size() ? asked.subList(0, limit) : asked);
        Run run = add(model, task == null || task.isBlank() ? "questions" : task.strip(),
                asked.getFirst().choices().isEmpty() ? "answer" : "choice");
        run.total = asked.size();
        run.shots = shots;
        List<Benchmark.Example> questions = asked;
        Thread.ofPlatform().daemon().name("evaluate-" + model).start(() -> score(run, questions, shown));
        return state();
    }

    /**
     * Starts measuring how fast a model reads a prompt and writes an answer.
     *
     * @param model the model as the provider lists it
     * @return the state, as {@link #state} gives it
     * @throws IOException if there is no such model or another measurement is under way
     */
    public synchronized Map<String, Object> startSpeed(String model) throws IOException {
        busy();
        local.location(model);
        Run run = add(model, "speed", "speed");
        Thread.ofPlatform().daemon().name("speed-" + model).start(() -> speed(run));
        return state();
    }

    private void busy() throws IOException {
        Run now = runs.stream().filter(Run::running).findFirst().orElse(null);
        if (now != null) {
            throw new IOException(now.model + " is being measured: wait for it to finish, or stop it");
        }
    }

    private Run add(String model, String task, String kind) {
        Run run = new Run(Long.toString(System.currentTimeMillis(), 36) + "-" + counter++, model, task, kind,
                System.currentTimeMillis());
        runs.addFirst(run);
        while (runs.size() > KEPT) {
            runs.removeLast();
        }
        return run;
    }

    /**
     * Asks the measurement under way to stop after the questions it is on; its score so far is kept.
     *
     * @return whether one was under way
     */
    public synchronized boolean stop() {
        Run now = runs.stream().filter(Run::running).findFirst().orElse(null);
        if (now == null) {
            return false;
        }
        now.stop = true;
        return true;
    }

    /**
     * Forgets a finished measurement.
     *
     * @param id its id
     * @return whether there was one
     */
    public synchronized boolean remove(String id) {
        boolean removed = runs.removeIf(r -> r.id.equals(id) && !r.running());
        if (removed) {
            save();
        }
        return removed;
    }

    private void score(Run run, List<Benchmark.Example> questions, List<Benchmark.Example> shown) {
        try {
            Generator generator = local.open(run.model).generator();
            run.state = "running";
            boolean numbers = questions.stream().allMatch(q -> q.target() != null && q.target().matches("-?[$0-9.,]+"));
            Function<String, String> extract = numbers ? Benchmark::lastNumber
                    : text -> text.strip().lines().findFirst().map(String::strip).orElse("");
            for (int from = 0; from < questions.size() && !run.stop; from += CHUNK) {
                List<Benchmark.Example> part = questions.subList(from, Math.min(questions.size(), from + CHUNK));
                Benchmark.Result r = run.kind.equals("choice") ? Benchmark.multipleChoice(generator, part, shown)
                        : Benchmark.generative(generator, part, shown, extract, numbers ? 256 : 48, CHUNK);
                run.correct += r.correct();
                run.normalized += r.normalizedCorrect();
                run.done += r.total();
            }
            run.state = run.stop ? "stopped" : "done";
        } catch (IOException | RuntimeException | OutOfMemoryError e) {
            fail(run, e);
        } finally {
            finish(run);
        }
    }

    private void speed(Run run) {
        try {
            Generator generator = local.open(run.model).generator();
            run.state = "running";
            generator.generate(List.of("Hello"), Generator.Options.greedy(2), null);   // the first call pays for warming up
            int read = generator.encode(PASSAGE).length;
            long[] first = {0};
            long start = System.nanoTime();
            // a model may end its answer early, so no end is taken: every run writes the same number of tokens
            List<Generator.Output> out = generator.generate(List.of(PASSAGE),
                    Generator.Options.greedy(96).withStopTokens(java.util.Set.of()), (row, piece) -> {
                        if (first[0] == 0) {
                            first[0] = System.nanoTime();
                        }
                    });
            long end = System.nanoTime();
            int written = out.getFirst().tokens().length;
            if (first[0] == 0 || written < 2) {
                throw new IOException("the model wrote nothing to time");
            }
            run.readPerSecond = read / ((first[0] - start) / 1e9);
            run.writePerSecond = (written - 1) / Math.max(1e-9, (end - first[0]) / 1e9);
            run.state = "done";
        } catch (IOException | RuntimeException | OutOfMemoryError e) {
            fail(run, e);
        } finally {
            finish(run);
        }
    }

    private static void fail(Run run, Throwable e) {
        run.error = e instanceof OutOfMemoryError ? "not enough memory for this model" : String.valueOf(e.getMessage());
        run.state = "failed";
    }

    private synchronized void finish(Run run) {
        run.ended = System.currentTimeMillis();
        save();
    }

    private void save() {
        try {
            Path parent = file.toAbsolutePath().getParent();
            Files.createDirectories(parent);
            Path part = file.resolveSibling(file.getFileName() + ".part");
            Files.writeString(part, Json.write(Map.of("runs", runs.stream().filter(r -> !r.running())
                    .map(Run::toJson).toList())));
            Files.move(part, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            System.getLogger(LocalEvaluation.class.getName()).log(System.Logger.Level.WARNING,
                    "cannot keep the measurements in " + file + ": " + e.getMessage());
        }
    }
}
