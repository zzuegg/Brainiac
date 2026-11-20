package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;
import io.github.zzuegg.brainiac.train.TrainingListener;
import io.github.zzuegg.brainiac.train.TrainingResult;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

/**
 * Trains models built in the studio, one run at a time on this machine, in the order they were
 * started. Each run reads a model from {@code models}, trains it on a data source with an
 * optimizer, loss, schedule and metrics found in the installed modules, and saves the result as a
 * new model next to it. Runs are kept in {@code runs} so the list survives a restart.
 */
public final class TrainingService implements AutoCloseable {

    /** What the run list calls this machine. */
    public static final String THIS_PC = "This PC";

    private static final System.Logger LOG = System.getLogger(TrainingService.class.getName());
    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}");
    private static final int MAX_EPOCHS = 1_000_000;

    private final Path models;
    private final Path runsFolder;
    private final DataSources data;
    private final Hosts hosts;
    private final Map<String, TrainingRun> runs = new ConcurrentHashMap<>();
    private final ExecutorService worker = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().name("studio-training").daemon(true).factory());
    private final Map<Path, Map<String, Object>> shapes = new ConcurrentHashMap<>();
    private volatile LayerCatalog options;
    private int counter;

    /**
     * Uses a data folder.
     *
     * @param folder the studio's data folder; models, runs and datasets are kept inside it
     */
    public TrainingService(Path folder) {
        this.models = folder.resolve("models");
        this.runsFolder = folder.resolve("runs");
        this.data = new DataSources(folder.resolve("datasets"));
        this.hosts = new Hosts(folder.resolve("hosts.properties"));
        loadRuns();
    }

    /**
     * Returns the data sources.
     *
     * @return the data sources
     */
    public DataSources data() {
        return data;
    }

    /**
     * Returns the other machines runs can train on.
     *
     * @return the hosts
     */
    public Hosts hosts() {
        return hosts;
    }

    /**
     * Returns the folder models are read from and saved to.
     *
     * @return {@code models} in the data folder
     */
    public Path modelsFolder() {
        return models;
    }

    /**
     * Finds a run.
     *
     * @param id its id
     * @return the run
     */
    public java.util.Optional<TrainingRun> run(String id) {
        return java.util.Optional.ofNullable(runs.get(id));
    }

    /**
     * Returns the model a run saved.
     *
     * @param id the run
     * @return the file, if the run saved one
     */
    public java.util.Optional<Path> output(String id) {
        return run(id).map(r -> r.toJson(false).get("output")).map(o -> models.resolve(o + ".brainiac"))
                .filter(Files::isRegularFile);
    }

    /** What a type is for training: {@code optimizer}, {@code loss}, {@code schedule}, {@code metric}, or null. */
    static String kind(Class<?> t) {
        if (Optimizer.class.isAssignableFrom(t)) {
            return "optimizer";
        }
        if (Loss.class.isAssignableFrom(t)) {
            return "loss";
        }
        if (LearningRateSchedule.class.isAssignableFrom(t)) {
            return "schedule";
        }
        if (Metric.class.isAssignableFrom(t)) {
            return "metric";
        }
        return null;
    }

    /**
     * Returns the optimizers, losses, schedules and metrics the installed modules offer.
     *
     * @return the catalog
     */
    public LayerCatalog options() {
        LayerCatalog c = options;
        if (c == null) {
            synchronized (this) {
                if (options == null) {
                    options = LayerCatalog.discover(TrainingService::kind);
                }
                c = options;
            }
        }
        return c;
    }

    /**
     * Describes the training options for the page, grouped by kind, with the devices.
     *
     * @return {@code {optimizers, losses, schedules, metrics, devices}}
     */
    public Map<String, Object> optionsJson() {
        Map<String, List<Object>> byKind = new LinkedHashMap<>();
        for (String k : List.of("optimizer", "loss", "schedule", "metric")) {
            byKind.put(k, new ArrayList<>());
        }
        for (LayerCatalog.Entry e : options().entries()) {
            byKind.get(e.produces()).add(LayerCatalog.describe(e));
        }
        List<Object> devices = new ArrayList<>();
        devices.add(Map.of("key", "best", "name", "Best available (" + safeName(Device::best) + ")"));
        for (Device d : Device.available()) {
            // each GPU by its name, so a card's Vulkan, HIP and CUDA devices can each be chosen
            devices.add(Map.of("key", d.kind() == Device.Kind.CPU ? "cpu" : d.name(), "name", d.name()));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("optimizers", byKind.get("optimizer"));
        out.put("losses", byKind.get("loss"));
        out.put("schedules", byKind.get("schedule"));
        out.put("metrics", byKind.get("metric"));
        out.put("devices", devices);
        return out;
    }

    private static String safeName(java.util.function.Supplier<Device> d) {
        try {
            return d.get().name();
        } catch (RuntimeException e) {
            return "none";
        }
    }

    /**
     * Lists the models that can be trained, with their input and output shapes.
     *
     * @return the models, newest first
     * @throws IOException if the folder cannot be read
     */
    public List<Map<String, Object>> models() throws IOException {
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.isDirectory(models)) {
            return out;
        }
        List<Path> files = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(models, "*.brainiac")) {
            ds.forEach(files::add);
        }
        files.sort(Comparator.comparing((Path p) -> p.toFile().lastModified()).reversed());
        for (Path f : files) {
            out.add(describeModel(f));
        }
        return out;
    }

    private Map<String, Object> describeModel(Path f) {
        long modified = f.toFile().lastModified();
        Map<String, Object> cached = shapes.get(f);
        if (cached != null && Long.valueOf(modified).equals(cached.get("modified"))) {
            return cached;
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", f.getFileName().toString().replaceFirst("\\.brainiac$", ""));
        m.put("modified", modified);
        try {
            Model model = Model.load(f, Device.cpu());
            m.put("input", dims(model.inputShape()));
            m.put("output", dims(model.outputShape()));
            m.put("parameters", model.parameterCount());
        } catch (IOException | RuntimeException e) {
            m.put("problem", "cannot read it: " + e.getMessage());
        }
        shapes.put(f, m);
        return m;
    }

    private static List<Integer> dims(Shape s) {
        List<Integer> out = new ArrayList<>();
        for (int d : s.toArray()) {
            out.add(d);
        }
        return out;
    }

    /**
     * Checks that a run's model and data fit together before it is started.
     *
     * @param spec the run settings
     * @return {@code {ok, problem?, samples, input, output, features, targets, stepsPerEpoch}}
     */
    public Map<String, Object> check(Map<String, Object> spec) {
        Map<String, Object> out = new LinkedHashMap<>();
        try {
            Path file = modelFile(spec);
            Map<String, Object> model = describeModel(file);
            if (model.containsKey("problem")) {
                throw new IllegalArgumentException(String.valueOf(model.get("problem")));
            }
            Dataset d = data.load(Json.object(spec.getOrDefault("data", Map.of())));
            out.put("samples", d.size());
            out.put("features", d.featureCount());
            out.put("targets", d.targetCount());
            out.put("classes", d.classes());
            out.put("input", model.get("input"));
            out.put("output", model.get("output"));
            String problem = fits(d, Json.array(model.get("input")), Json.array(model.get("output")));
            int batch = (int) Math.max(1, Json.integer(spec, "batchSize", 32));
            double validation = validationFraction(spec);
            int trainSize = validation > 0 ? (int) Math.round(d.size() * (1 - validation)) : d.size();
            out.put("trainSamples", trainSize);
            out.put("stepsPerEpoch", (trainSize + batch - 1) / batch);
            out.put("ok", problem == null);
            if (problem != null) {
                out.put("problem", problem);
            }
        } catch (IOException | RuntimeException e) {
            out.put("ok", false);
            out.put("problem", String.valueOf(e.getMessage()));
        }
        return out;
    }

    /**
     * Runs a saved model on one sample, so a model that was built or trained here can be tried:
     * on a sample of a data set (which also says what the right answer is) or on values typed in.
     *
     * @param request {@code {model, values?: [numbers], data?: a run's data settings, index?, full?}};
     *     with {@code data} and no {@code index}, a sample is picked at random; with {@code full} the
     *     whole input comes back, up to the size of a picture, where it is else cut to 64 values
     * @return {@code {input, output, best, label?, expected?, expectedLabel?, index?, samples?}}:
     *     {@code best} is the position of the largest output, {@code label} its class name when the
     *     data has classes
     * @throws IOException if the model or the data cannot be read
     */
    public Map<String, Object> predict(Map<String, Object> request) throws IOException {
        Model model = Model.load(modelFile(request), Device.cpu());
        Map<String, Object> out = new LinkedHashMap<>();
        float[] input;
        float[] target = null;
        List<String> classes = List.of();
        if (request.get("data") != null) {
            Dataset d = data.load(Json.object(request.get("data")));
            if (d.size() == 0) {
                throw new IllegalArgumentException("the data has no samples");
            }
            if (d.featureCount() != model.inputSize()) {
                throw new IllegalArgumentException("each sample has " + d.featureCount() + " values, but the model takes "
                        + model.inputSize());
            }
            long asked = Json.integer(request, "index", -1);
            int index = asked < 0 ? java.util.concurrent.ThreadLocalRandom.current().nextInt(d.size())
                    : (int) (asked % d.size());
            input = d.featuresOf(index);
            classes = d.classes() == null ? List.of() : d.classes();
            target = d.targetsOf(index);
            out.put("index", index);
            out.put("samples", d.size());
            out.put("expected", floats(target, 256));
        } else {
            List<Object> values = Json.array(request.getOrDefault("values", List.of()));
            if (values.size() != model.inputSize()) {
                throw new IllegalArgumentException("the model takes " + model.inputSize() + " values ("
                        + dims(model.inputShape()) + "), and " + values.size() + " were given");
            }
            input = new float[values.size()];
            for (int i = 0; i < input.length; i++) {
                if (!(values.get(i) instanceof Number n) || !Double.isFinite(n.doubleValue())) {
                    throw new IllegalArgumentException("value " + (i + 1) + " is not a number");
                }
                input[i] = n.floatValue();
            }
        }
        float[] output = model.predict(input);
        int best = largest(output);
        if (target != null && output.length > 1) {
            // the right class: a class number, or the 1 of a one-hot row
            int want = target.length == 1 ? Math.round(target[0]) : target.length == output.length && oneHot(target)
                    ? largest(target) : -1;
            if (want >= 0 && want < output.length) {
                out.put("expectedBest", want);
                if (classes.size() == output.length) {
                    out.put("expectedLabel", classes.get(want));
                }
            }
        }
        out.put("input", floats(input, Boolean.TRUE.equals(request.get("full")) ? FULL_INPUT : 64));
        out.put("inputSize", input.length);
        out.put("output", floats(output, 256));
        out.put("outputSize", output.length);
        out.put("best", best);
        if (classes.size() == output.length) {
            out.put("label", classes.get(best));
            out.put("classes", classes.size() <= 256 ? classes : classes.subList(0, 256));
        }
        return out;
    }

    // all zeros but a single one
    private static boolean oneHot(float[] values) {
        int ones = 0;
        for (float v : values) {
            if (v == 1f) {
                ones++;
            } else if (v != 0f) {
                return false;
            }
        }
        return ones == 1;
    }

    private static int largest(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    // the first values of an array as JSON numbers; what is not a number (an overflow) is sent as null
    /** The most input values a prediction gives back when asked for all: a 128 by 128 colour picture. */
    static final int FULL_INPUT = 3 * 128 * 128;

    private static List<Object> floats(float[] values, int most) {
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < Math.min(values.length, most); i++) {
            out.add(Float.isFinite(values[i]) ? (Object) (double) values[i] : null);
        }
        return out;
    }

    /** Null if the data fits the model, otherwise why not. */
    static String fits(Dataset d, List<Object> input, List<Object> output) {
        long in = input.stream().mapToLong(v -> ((Number) v).longValue()).reduce(1, (a, b) -> a * b);
        long outCount = output.stream().mapToLong(v -> ((Number) v).longValue()).reduce(1, (a, b) -> a * b);
        if (d.featureCount() != in) {
            return "each sample has " + d.featureCount() + " values, but the model takes " + input + " = " + in
                    + " values; change the Input block or pick other data";
        }
        boolean classIds = d.targetCount() == 1 && outCount > 1;
        if (d.targetCount() != outCount && !classIds) {
            return "each sample has " + d.targetCount() + " target values, but the model gives " + output + " = "
                    + outCount + "; make the last block give " + d.targetCount();
        }
        return null;
    }

    /**
     * Starts a run, or queues it behind the one that is training.
     *
     * @param spec {@code {model, name?, data, optimizer, loss, schedule?, metrics?, epochs, batchSize,
     *     validation?, patience?, seed?, clipNorm?, device?}}
     * @return the run
     */
    public synchronized TrainingRun start(Map<String, Object> spec) {
        modelFile(spec);   // fails early if the model is missing
        int epochs = (int) Json.integer(spec, "epochs", 10);
        if (epochs < 1 || epochs > MAX_EPOCHS) {
            throw new IllegalArgumentException("epochs must be between 1 and " + MAX_EPOCHS);
        }
        if (Json.integer(spec, "batchSize", 32) < 1) {
            throw new IllegalArgumentException("the batch size must be at least 1");
        }
        validationFraction(spec);
        String hostId = Json.string(spec, "host", "");
        Hosts.Host host = hostId.isEmpty() || hostId.equals("local") ? null : hosts.find(hostId)
                .orElseThrow(() -> new IllegalArgumentException("there is no host " + hostId));
        String id = String.valueOf(++counter);
        String model = Json.string(spec, "model", "");
        String name = Json.string(spec, "name", "");
        if (name.isBlank()) {
            name = model + " · " + optimizerLabel(spec);
        }
        TrainingRun run = new TrainingRun(id, name.strip(), spec, host == null ? THIS_PC : host.name());
        runs.put(id, run);
        save(run);
        if (host == null) {
            worker.submit(() -> execute(run));
        } else {
            Thread.ofVirtual().name("studio-remote-run-" + id).start(() -> executeRemote(run, host));
        }
        return run;
    }

    private static String optimizerLabel(Map<String, Object> spec) {
        Object o = spec.get("optimizer");
        if (o instanceof Map<?, ?> m && m.get("entry") instanceof String key) {
            String member = key.substring(key.indexOf('#') + 1);
            return member.contains("(") ? member.substring(0, member.indexOf('(')) : member;
        }
        return "adam";
    }

    /**
     * Asks a run to stop. A queued run is dropped; a running one stops after its current step and
     * saves what it learned so far.
     *
     * @param id the run
     * @return false if there is no such run
     */
    public boolean stop(String id) {
        TrainingRun r = runs.get(id);
        if (r == null) {
            return false;
        }
        r.askToStop();
        return true;
    }

    /**
     * Forgets a run that has ended. The model it saved is kept.
     *
     * @param id the run
     * @return false if there is no such run
     * @throws IOException if its record cannot be deleted
     */
    public boolean delete(String id) throws IOException {
        TrainingRun r = runs.get(id);
        if (r == null) {
            return false;
        }
        if (r.state().active()) {
            throw new IllegalArgumentException("stop the run before removing it");
        }
        runs.remove(id);
        Files.deleteIfExists(runsFolder.resolve(id + ".json"));
        return true;
    }

    /**
     * Returns every run, newest first.
     *
     * @return the runs
     */
    public List<TrainingRun> runs() {
        List<TrainingRun> list = new ArrayList<>(runs.values());
        list.sort(Comparator.comparingInt((TrainingRun r) -> number(r.id())).reversed());
        return list;
    }

    private static int number(String id) {
        try {
            return Integer.parseInt(id);
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private Path modelFile(Map<String, Object> spec) {
        String name = Json.string(spec, "model", "");
        if (!NAME.matcher(name).matches() || name.contains("..")) {
            throw new IllegalArgumentException("pick a model to train");
        }
        Path f = models.resolve(name + ".brainiac");
        if (!Files.isRegularFile(f)) {
            throw new IllegalArgumentException("there is no model " + name + "; build it first");
        }
        return f;
    }

    private static double validationFraction(Map<String, Object> spec) {
        double v = Json.number(spec, "validation", 0);
        if (v < 0 || v > 0.5) {
            throw new IllegalArgumentException("hold back between 0 and 50% for validation");
        }
        return v;
    }

    /** Thrown from inside training to end a run that was asked to stop. */
    private static final class Stop extends RuntimeException {
        private static final long serialVersionUID = 1L;

        Stop() {
            super("stopped", null, false, false);
        }
    }

    private void execute(TrainingRun run) {
        if (run.stopAsked()) {
            run.finish(TrainingRun.State.STOPPED, null, null, 0, false);
            save(run);
            return;
        }
        Map<String, Object> spec = run.spec();
        Model model = null;
        try {
            run.loading();
            save(run);
            Device device = device(Json.string(spec, "device", "best"));
            model = Model.load(modelFile(spec), device);
            Dataset all = data.load(Json.object(spec.getOrDefault("data", Map.of())));
            String problem = fits(all, dims(model.inputShape()).stream().map(i -> (Object) i).toList(),
                    dims(model.outputShape()).stream().map(i -> (Object) i).toList());
            if (problem != null) {
                throw new IllegalArgumentException(problem);
            }
            long seed = Json.integer(spec, "seed", 42);
            double validation = validationFraction(spec);
            Dataset train = all;
            Dataset check = null;
            if (validation > 0) {
                Dataset.Split split = all.shuffle(seed).split(1 - validation);
                train = split.train();
                check = split.test();
            }
            Trainer trainer = trainer(model, spec, seed, check);
            int batch = (int) Json.integer(spec, "batchSize", 32);
            run.running(device.name(), train.size(), (train.size() + batch - 1) / batch);
            save(run);
            long[] epochStart = {System.currentTimeMillis()};
            long[] lastSave = {System.currentTimeMillis()};
            trainer.listener(new TrainingListener() {
                @Override
                public void onEpoch(EpochResult result) {
                    long now = System.currentTimeMillis();
                    run.epoch(result, now - epochStart[0]);
                    epochStart[0] = now;
                    if (now - lastSave[0] > 2000) {
                        save(run);
                        lastSave[0] = now;
                    }
                    if (run.stopAsked()) {
                        throw new Stop();
                    }
                }

                @Override
                public void onBatch(int epoch, long step, float loss) {
                    run.batch(epoch, step, loss);
                    if (run.stopAsked()) {
                        throw new Stop();
                    }
                }
            });
            TrainingResult result = trainer.fit(train);
            String out = saveModel(model, spec, run);
            run.finish(TrainingRun.State.FINISHED, out, null, result.bestEpoch(), result.stoppedEarly());
        } catch (Stop stop) {
            String out = null;
            String error = null;
            try {
                out = model == null ? null : saveModel(model, spec, run);
            } catch (IOException | RuntimeException e) {
                error = "stopped, but not saved: " + e.getMessage();
            }
            run.finish(TrainingRun.State.STOPPED, out, error, 0, false);
        } catch (IOException | RuntimeException | OutOfMemoryError e) {
            LOG.log(System.Logger.Level.INFO, "training run " + run.id() + " failed", e);
            run.finish(TrainingRun.State.FAILED, null, message(e), 0, false);
        }
        save(run);
    }

    // ---------------------------------------------------------------- runs on other machines

    /** How long a host may not answer before its run is given up. */
    private static final long LOST_MILLIS = 120_000;

    private void executeRemote(TrainingRun run, Hosts.Host host) {
        WorkerClient c = hosts.client(host);
        Map<String, Object> spec = run.spec();
        Map<String, Object> last = Map.of();
        try {
            if (run.remoteId() == null) {
                if (run.stopAsked()) {
                    run.finish(TrainingRun.State.STOPPED, null, null, 0, false);
                    save(run);
                    return;
                }
                run.loading();
                save(run);
                String model = c.upload(modelFile(spec));
                Map<String, Object> there = new LinkedHashMap<>(spec);
                there.remove("host");
                there.put("name", run.name());
                there.put("data", remoteData(Json.object(spec.getOrDefault("data", Map.of())), c));
                Map<String, Object> started = c.start(there, model);
                run.remoteId(Json.string(started, "id", null));
                save(run);
            }
            boolean stopSent = false;
            long lastAnswer = System.currentTimeMillis();
            long lastSave = 0;
            while (true) {
                if (run.stopAsked() && !stopSent) {
                    try {
                        c.stop(run.remoteId());
                        stopSent = true;
                    } catch (IOException e) {
                        // asked again on the next round
                    }
                }
                try {
                    last = c.run(run.remoteId());
                    lastAnswer = System.currentTimeMillis();
                } catch (IOException e) {
                    if (System.currentTimeMillis() - lastAnswer > LOST_MILLIS) {
                        throw new IOException(host.name() + " has not answered for two minutes: " + e.getMessage(), e);
                    }
                }
                run.mirror(last);
                String state = Json.string(last, "state", "running");
                boolean ended = !List.of("queued", "loading", "running").contains(state) && !last.isEmpty();
                long now = System.currentTimeMillis();
                if (ended) {
                    break;
                }
                if (now - lastSave > 2000) {
                    save(run);
                    lastSave = now;
                }
                Thread.sleep(1000);
            }
            String state = Json.string(last, "state", "failed");
            String out = null;
            String error = Json.string(last, "error", null);
            if (last.get("output") != null) {
                String name = outputName(spec, run);
                Files.createDirectories(models);
                c.download(run.remoteId(), models.resolve(name + ".brainiac"));
                out = name;
            }
            TrainingRun.State end = switch (state) {
                case "finished" -> TrainingRun.State.FINISHED;
                case "stopped" -> TrainingRun.State.STOPPED;
                case "interrupted" -> TrainingRun.State.INTERRUPTED;
                default -> TrainingRun.State.FAILED;
            };
            run.finish(end, out, error, (int) Json.integer(last, "bestEpoch", 0), Boolean.TRUE.equals(last.get("stoppedEarly")));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;   // the studio is closing; the run carries on there and is picked up again at the next start
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.INFO, "training run " + run.id() + " on " + host.name() + " failed", e);
            run.finish(TrainingRun.State.FAILED, null, message(e), 0, false);
        }
        save(run);
    }

    /** The data settings with each file sent to the host and referred to by its hash. */
    private Map<String, Object> remoteData(Map<String, Object> d, WorkerClient c) throws IOException {
        Map<String, Object> out = new LinkedHashMap<>(d);
        switch (Json.string(d, "kind", "builtin")) {
            case "csv" -> out.put("file", c.upload(data.resolve(Json.string(d, "file", ""))));
            case "idx" -> {
                out.put("images", c.upload(data.resolve(Json.string(d, "images", ""))));
                out.put("labels", c.upload(data.resolve(Json.string(d, "labels", ""))));
            }
            case "images" -> out.put("folder", c.uploadFolder(data.resolve(Json.string(d, "folder", ""))));
            default -> {
                // built-in sets are made on the host
            }
        }
        return out;
    }

    private static String message(Throwable e) {
        if (e instanceof OutOfMemoryError) {
            return "ran out of memory; try a smaller batch, a smaller model or less data";
        }
        String m = e.getMessage();
        return m == null || m.isBlank() ? e.getClass().getSimpleName() : m;
    }

    private Trainer trainer(Model model, Map<String, Object> spec, long seed, Dataset validation) {
        SplittableRandom random = new SplittableRandom(seed);
        Trainer t = Trainer.of(model)
                .epochs((int) Json.integer(spec, "epochs", 10))
                .batchSize((int) Json.integer(spec, "batchSize", 32))
                .seed(seed)
                .shuffle(!Boolean.FALSE.equals(spec.get("shuffle")));
        if (spec.get("optimizer") != null) {
            t.optimizer((Optimizer) make(spec.get("optimizer"), "optimizer", random));
        }
        if (spec.get("loss") != null) {
            t.loss((Loss) make(spec.get("loss"), "loss", random));
        }
        if (spec.get("schedule") != null) {
            t.schedule((LearningRateSchedule) make(spec.get("schedule"), "schedule", random));
        }
        if (spec.get("metrics") != null) {
            List<Metric> metrics = new ArrayList<>();
            for (Object m : Json.array(spec.get("metrics"))) {
                metrics.add((Metric) make(m instanceof String key ? Map.of("entry", key) : m, "metric", random));
            }
            t.metric(metrics.toArray(Metric[]::new));
        }
        if (validation != null) {
            t.validation(validation);
            long patience = Json.integer(spec, "patience", 0);
            if (patience > 0) {
                t.earlyStopping((int) patience);
            }
        }
        double clip = Json.number(spec, "clipNorm", 0);
        if (clip > 0) {
            t.clipGradientNorm((float) clip);
        }
        return t;
    }

    private Object make(Object choice, String kind, SplittableRandom random) {
        Map<String, Object> c = Json.object(choice);
        String key = Json.string(c, "entry", "");
        LayerCatalog.Entry entry = options().entry(key)
                .orElseThrow(() -> new IllegalArgumentException("no " + kind + " " + key));
        if (!kind.equals(entry.produces())) {
            throw new IllegalArgumentException(key + " is no " + kind);
        }
        return ModelGraph.make(entry, c.get("args"), c.get("options"), random);
    }

    private static Device device(String key) {
        return switch (key) {
            case "cpu" -> Device.cpu();
            case "gpu" -> Device.gpu();
            case "best", "" -> Device.best();
            default -> Device.available().stream().filter(d -> d.name().equals(key)).findFirst()
                    .orElseGet(Device::best);
        };
    }

    private static String outputName(Map<String, Object> spec, TrainingRun run) {
        String base = Json.string(spec, "model", "model");
        return (base.length() > 50 ? base.substring(0, 50) : base) + "-run" + run.id();
    }

    private String saveModel(Model model, Map<String, Object> spec, TrainingRun run) throws IOException {
        String name = outputName(spec, run);
        Files.createDirectories(models);
        Path tmp = models.resolve(name + ".brainiac.tmp");
        model.save(tmp);
        Files.move(tmp, models.resolve(name + ".brainiac"), StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE);
        return name;
    }

    private void save(TrainingRun run) {
        try {
            Files.createDirectories(runsFolder);
            Path tmp = runsFolder.resolve(run.id() + ".json.tmp");
            Files.writeString(tmp, Json.write(run.toJson(true)), StandardCharsets.UTF_8);
            Files.move(tmp, runsFolder.resolve(run.id() + ".json"), StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot save training run " + run.id(), e);
        }
    }

    private void loadRuns() {
        if (!Files.isDirectory(runsFolder)) {
            return;
        }
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(runsFolder, "*.json")) {
            for (Path f : ds) {
                try {
                    TrainingRun r = TrainingRun.fromJson(Json.object(Json.parse(Files.readString(f, StandardCharsets.UTF_8))));
                    runs.put(r.id(), r);
                    counter = Math.max(counter, number(r.id()));
                } catch (IOException | RuntimeException e) {
                    LOG.log(System.Logger.Level.WARNING, "cannot read training run " + f, e);
                }
            }
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot list " + runsFolder, e);
        }
        for (TrainingRun r : runs.values()) {
            Hosts.Host host = r.host().equals(THIS_PC) || r.remoteId() == null ? null
                    : hosts.list().stream().filter(h -> h.name().equals(r.host())).findFirst().orElse(null);
            if (r.state().active() && host != null) {
                Thread.ofVirtual().name("studio-remote-run-" + r.id()).start(() -> executeRemote(r, host));   // still training there
            } else {
                r.interrupted();
                save(r);
            }
        }
    }

    @Override
    public void close() {
        runs.values().forEach(TrainingRun::askToStop);
        worker.shutdownNow();
        // a run marks itself ended a moment before its file is written; let that write finish
        try {
            if (!worker.awaitTermination(10, java.util.concurrent.TimeUnit.SECONDS)) {
                LOG.log(System.Logger.Level.WARNING, "training runs were still ending when the studio closed");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
