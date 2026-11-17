package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Trains and tests any model on a fixed task, so different models can be compared on
 * equal terms: the same data, the same split, the same training settings unless you
 * change them, and the same report.
 *
 * <pre>{@code
 * Benchmark.Task task = ...;   // for example Tasks.mnist() from the examples
 * Benchmark.Report report = Benchmark.run(task, "my cnn", myModel);
 * System.out.println(report);
 *
 * // several models at once, each trained for 2 epochs
 * List<Benchmark.Report> reports = Benchmark.compare(task,
 *         Map.of("small", () -> small(), "big", () -> big()), t -> t.epochs(2));
 * System.out.println(Benchmark.table(reports));
 * }</pre>
 *
 * <p>A model fits a task when its input and output have as many values as the task's
 * shapes; a flat model ({@code input(784)}) works on a task with images
 * {@code [1, 28, 28]}, and the other way round.
 */
public final class Benchmark {

    private Benchmark() {
    }

    /** What a task asks for, which decides its loss, metric and report. */
    public enum Kind {
        /**
         * Pick one class per output row: the last axis of the output holds the class
         * scores, so a sequence output {@code [steps, classes]} is one choice per step.
         * One output value means a yes/no choice at 0.5.
         */
        CLASSIFICATION,
        /** Predict numbers. */
        REGRESSION
    }

    /**
     * A fixed problem to train and test models on.
     *
     * @param name a short name
     * @param description one line on what the task is
     * @param kind classification or regression
     * @param loss the loss to train with
     * @param train the training data
     * @param test the held-out data the report is computed on
     * @param inputShape the shape of one sample
     * @param outputShape the shape of one target
     * @param classNames names for the classes, or an empty list
     * @param defaults training settings that suit the task (optimizer, batch size,
     *        epochs, clipping...); applied before any settings passed to a run
     */
    public record Task(String name, String description, Kind kind, Loss loss, Dataset train, Dataset test,
                       Shape inputShape, Shape outputShape, List<String> classNames, Consumer<Trainer> defaults) {

        /** Checks that the data fits the shapes. */
        public Task {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(loss, "loss");
            classNames = List.copyOf(classNames);
            description = description == null ? "" : description;
            defaults = defaults == null ? t -> { } : defaults;
            for (Dataset d : List.of(train, test)) {
                boolean ids = kind == Kind.CLASSIFICATION && outputShape.dim(-1) > 1
                        && (long) d.targetCount() * outputShape.dim(-1) == outputShape.size();
                if (d.featureCount() != inputShape.size() || (d.targetCount() != outputShape.size() && !ids)) {
                    throw new IllegalArgumentException("task " + name + ": data has " + d.featureCount()
                            + " features and " + d.targetCount() + " targets, but the shapes are "
                            + inputShape + " and " + outputShape);
                }
            }
        }

        /**
         * Creates a classification task trained with cross-entropy.
         *
         * @param name a short name
         * @param train the training data, one-hot targets
         * @param test the test data
         * @param inputShape the shape of one sample
         * @param outputShape the shape of one target; the last axis holds the classes
         * @return the task
         */
        public static Task classification(String name, Dataset train, Dataset test, Shape inputShape,
                                          Shape outputShape) {
            return new Task(name, "", Kind.CLASSIFICATION, Loss.CROSS_ENTROPY, train, test, inputShape,
                    outputShape, List.of(), null);
        }

        /**
         * Creates a regression task trained with mean squared error.
         *
         * @param name a short name
         * @param train the training data
         * @param test the test data
         * @param inputShape the shape of one sample
         * @param outputShape the shape of one target
         * @return the task
         */
        public static Task regression(String name, Dataset train, Dataset test, Shape inputShape,
                                      Shape outputShape) {
            return new Task(name, "", Kind.REGRESSION, Loss.MSE, train, test, inputShape, outputShape,
                    List.of(), null);
        }

        /**
         * @param text one line on what the task is
         * @return a copy with this description
         */
        public Task describedAs(String text) {
            return new Task(name, text, kind, loss, train, test, inputShape, outputShape, classNames, defaults);
        }

        /**
         * @param newLoss the loss to train with
         * @return a copy with this loss
         */
        public Task withLoss(Loss newLoss) {
            return new Task(name, description, kind, newLoss, train, test, inputShape, outputShape, classNames,
                    defaults);
        }

        /**
         * @param names names for the classes
         * @return a copy with these names
         */
        public Task withClassNames(List<String> names) {
            return new Task(name, description, kind, loss, train, test, inputShape, outputShape, names, defaults);
        }

        /**
         * @param settings training settings that suit the task
         * @return a copy with these defaults
         */
        public Task withDefaults(Consumer<Trainer> settings) {
            return new Task(name, description, kind, loss, train, test, inputShape, outputShape, classNames,
                    settings);
        }

        /**
         * Returns a smaller copy, for quick runs.
         *
         * @param trainSamples the most training samples to keep
         * @param testSamples the most test samples to keep
         * @return the copy
         */
        public Task limit(int trainSamples, int testSamples) {
            return new Task(name, description, kind, loss,
                    train.slice(0, Math.min(trainSamples, train.size())),
                    test.slice(0, Math.min(testSamples, test.size())),
                    inputShape, outputShape, classNames, defaults);
        }

        /** @return the number of classes; 2 for a single yes/no output */
        public int classes() {
            int last = outputShape.dim(outputShape.rank() - 1);
            return last == 1 ? 2 : last;
        }

        /** @return the metric the report leads with: accuracy or mean absolute error */
        public Metric metric() {
            return kind == Kind.CLASSIFICATION ? Metric.ACCURACY : Metric.MEAN_ABSOLUTE_ERROR;
        }
    }

    /**
     * The outcome of one model on one task.
     *
     * @param task the task's name
     * @param kind the task's kind
     * @param model the model's name
     * @param parameters the number of trainable values
     * @param trainSeconds wall time spent training
     * @param samplesPerSecond inference speed on the test data
     * @param test loss and metrics on the test data
     * @param training the training history, or {@code null} if the model was only tested
     * @param confusion for classification, {@code confusion[actual][predicted]} counts;
     *        otherwise {@code null}
     * @param classNames names for the classes, or an empty list
     */
    public record Report(String task, Kind kind, String model, long parameters, double trainSeconds,
                         double samplesPerSecond, Evaluation test, TrainingResult training, int[][] confusion,
                         List<String> classNames) {

        /** @return the epochs trained */
        public int epochs() {
            return training == null ? 0 : training.history().size();
        }

        /** @return test accuracy, or NaN for regression */
        public float accuracy() {
            return kind == Kind.CLASSIFICATION ? test.get(Metric.ACCURACY) : Float.NaN;
        }

        /** @return test mean absolute error, or NaN for classification */
        public float meanAbsoluteError() {
            return kind == Kind.REGRESSION ? test.get(Metric.MEAN_ABSOLUTE_ERROR) : Float.NaN;
        }

        /** @return the share of each class's test samples that were classified correctly */
        public float[] perClassAccuracy() {
            if (confusion == null) {
                return new float[0];
            }
            float[] out = new float[confusion.length];
            for (int c = 0; c < confusion.length; c++) {
                long total = 0;
                for (int v : confusion[c]) {
                    total += v;
                }
                out[c] = total == 0 ? Float.NaN : (float) confusion[c][c] / total;
            }
            return out;
        }

        /**
         * Lists the most frequent mistakes.
         *
         * @param count how many to list
         * @return lines like {@code "4 -> 9: 12"}, most frequent first
         */
        public List<String> topMistakes(int count) {
            List<int[]> mistakes = new ArrayList<>();
            if (confusion != null) {
                for (int a = 0; a < confusion.length; a++) {
                    for (int p = 0; p < confusion.length; p++) {
                        if (a != p && confusion[a][p] > 0) {
                            mistakes.add(new int[] {a, p, confusion[a][p]});
                        }
                    }
                }
            }
            mistakes.sort((x, y) -> Integer.compare(y[2], x[2]));
            List<String> out = new ArrayList<>();
            for (int[] m : mistakes.subList(0, Math.min(count, mistakes.size()))) {
                out.add(className(m[0]) + " -> " + className(m[1]) + ": " + m[2]);
            }
            return out;
        }

        /** @return the confusion matrix as a text table, rows actual and columns predicted */
        public String confusionTable() {
            if (confusion == null) {
                return "";
            }
            int width = 5;
            for (int c = 0; c < confusion.length; c++) {
                width = Math.max(width, className(c).length() + 1);
            }
            StringBuilder sb = new StringBuilder(" ".repeat(width) + " predicted\n");
            sb.append(String.format("%" + width + "s", "actual"));
            for (int c = 0; c < confusion.length; c++) {
                sb.append(String.format("%6s", shortName(c)));
            }
            sb.append('\n');
            for (int a = 0; a < confusion.length; a++) {
                sb.append(String.format("%" + width + "s", className(a)));
                for (int p = 0; p < confusion.length; p++) {
                    sb.append(String.format("%6d", confusion[a][p]));
                }
                sb.append('\n');
            }
            return sb.toString();
        }

        private String className(int c) {
            return c < classNames.size() ? classNames.get(c) : Integer.toString(c);
        }

        private String shortName(int c) {
            String n = className(c);
            return n.length() > 5 ? n.substring(0, 5) : n;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            String trained = training == null ? "tested only"
                    : String.format("%d epoch%s in %.1f s", epochs(), epochs() == 1 ? "" : "s", trainSeconds);
            sb.append(String.format("%s on %s: %,d parameters, %s, %,.0f samples/s inference%n",
                    model, task, parameters, trained, samplesPerSecond));
            if (kind == Kind.CLASSIFICATION) {
                sb.append(String.format("  test accuracy %.2f%%, loss %.4f%n", 100 * accuracy(), test.loss()));
                float[] perClass = perClassAccuracy();
                if (perClass.length > 2 && perClass.length <= 128) {   // not for thousands of tokens
                    sb.append("  per class:");
                    for (int c = 0; c < perClass.length; c++) {
                        sb.append(String.format(" %s %.1f%%", className(c), 100 * perClass[c]));
                    }
                    sb.append('\n');
                }
                List<String> mistakes = topMistakes(3);
                if (!mistakes.isEmpty()) {
                    sb.append("  most common mistakes: ").append(String.join(", ", mistakes)).append('\n');
                }
            } else {
                sb.append(String.format("  test mean absolute error %.4f, MSE %.5f%n", meanAbsoluteError(),
                        test.loss()));
            }
            return sb.toString();
        }
    }

    /**
     * Trains a model with the task's default settings and tests it.
     *
     * @param task the task
     * @param name a name for the report
     * @param model an untrained model whose input and output fit the task
     * @return the report
     */
    public static Report run(Task task, String name, Model model) {
        return run(task, name, model, t -> { });
    }

    /**
     * Trains a model with the task's default settings, changed by {@code settings}, and
     * tests it.
     *
     * @param task the task
     * @param name a name for the report
     * @param model an untrained model whose input and output fit the task
     * @param settings changes to the training settings, such as
     *        {@code t -> t.epochs(1).listener(Progress.every(1))}
     * @return the report
     */
    public static Report run(Task task, String name, Model model, Consumer<Trainer> settings) {
        check(task, model);
        Trainer trainer = Trainer.of(model).loss(task.loss()).metric(task.metric()).seed(1);
        task.defaults().accept(trainer);
        settings.accept(trainer);
        long start = System.nanoTime();
        TrainingResult history = trainer.fit(task.train());
        double seconds = (System.nanoTime() - start) / 1e9;
        return test(task, name, model, seconds, history);
    }

    /**
     * Tests a model that is already trained, for example one loaded from a file.
     *
     * @param task the task
     * @param name a name for the report
     * @param model a trained model whose input and output fit the task
     * @return the report
     */
    public static Report evaluate(Task task, String name, Model model) {
        check(task, model);
        return test(task, name, model, 0, null);
    }

    /**
     * Trains and tests several models, one after another, each with the same settings.
     *
     * @param task the task
     * @param models a fresh untrained model for each name
     * @param settings changes to the task's default training settings
     * @return one report per model, in the map's order
     */
    public static List<Report> compare(Task task, Map<String, Supplier<Model>> models, Consumer<Trainer> settings) {
        List<Report> reports = new ArrayList<>();
        for (Map.Entry<String, Supplier<Model>> e : new LinkedHashMap<>(models).entrySet()) {
            reports.add(run(task, e.getKey(), e.getValue().get(), settings));
        }
        return reports;
    }

    /**
     * Formats reports as one table, one row each.
     *
     * @param reports the reports
     * @return the table
     */
    public static String table(List<Report> reports) {
        int nameWidth = 5;
        int taskWidth = 4;
        for (Report r : reports) {
            nameWidth = Math.max(nameWidth, r.model().length());
            taskWidth = Math.max(taskWidth, r.task().length());
        }
        String row = "%-" + taskWidth + "s  %-" + nameWidth + "s  %10s  %6s  %8s  %10s  %12s%n";
        StringBuilder sb = new StringBuilder(String.format(row, "task", "model", "parameters", "epochs",
                "train s", "test loss", "score"));
        for (Report r : reports) {
            String score = r.kind() == Kind.CLASSIFICATION
                    ? String.format("%.2f%% acc", 100 * r.accuracy())
                    : String.format("%.4f MAE", r.meanAbsoluteError());
            sb.append(String.format(row, r.task(), r.model(), String.format("%,d", r.parameters()), r.epochs(),
                    String.format("%.2f", r.trainSeconds()), String.format("%.4f", tidy(r.test().loss())), score));
        }
        return sb.toString();
    }

    private static float tidy(float v) {
        return Math.abs(v) < 5e-5f ? 0f : v;     // no "-0.0000" from rounding
    }

    /**
     * Checks that a model's input and output fit a task.
     *
     * @param task the task
     * @param model the model
     * @throws IllegalArgumentException if they do not, naming the shapes needed
     */
    public static void check(Task task, Model model) {
        if (model.inputShape().size() != task.inputShape().size()
                || model.outputShape().size() != task.outputShape().size()) {
            throw new IllegalArgumentException("task " + task.name() + " needs a model from " + task.inputShape()
                    + " to " + task.outputShape() + ", but the model goes from " + model.inputShape() + " to "
                    + model.outputShape());
        }
    }

    private static final int INFERENCE_PASSES = 5;

    private static Report test(Task task, String name, Model model, double trainSeconds, TrainingResult history) {
        Evaluation evaluation = Trainer.of(model).loss(task.loss()).metric(task.metric()).evaluate(task.test());
        boolean classify = task.kind() == Kind.CLASSIFICATION;
        int last = task.outputShape().dim(task.outputShape().rank() - 1);
        int classes = task.classes();
        int[][] confusion = classify ? new int[classes][classes] : null;
        // Inference speed is the median of several passes: a single short pass is easily
        // skewed by clock ramp-up or a garbage collection.
        double[] rates = new double[INFERENCE_PASSES];
        for (int pass = 0; pass < rates.length; pass++) {
            long samples = 0;
            long start = System.nanoTime();
            // every batch is queued before any result is read, so a GPU never waits for the host
            List<Batch> batches = task.test().batches(256, model.device());
            List<Tensor> outputs = new ArrayList<>(batches.size());
            for (Batch batch : batches) {
                Tensor out = model.predict(batch.features());
                Tensor.evaluate(List.of(out));
                outputs.add(out);
            }
            // read at once: one wait for the device rather than one per batch
            float[] all = Tensor.concat(0, outputs.toArray(Tensor[]::new)).toArray();
            float[] targets = classify && pass == 0
                    ? Tensor.concat(0, batches.stream().map(Batch::targets).toArray(Tensor[]::new)).toArray()
                    : null;
            int offset = 0;
            int targetOffset = 0;
            for (int b = 0; b < batches.size(); b++) {
                Batch batch = batches.get(b);
                int n = Math.toIntExact(outputs.get(b).shape().size());
                float[] predicted = java.util.Arrays.copyOfRange(all, offset, offset + n);
                offset += n;
                samples += batch.size();
                if (!classify || pass > 0) {
                    continue;
                }
                int m = Math.toIntExact(batch.targets().size());
                float[] actual = java.util.Arrays.copyOfRange(targets, targetOffset, targetOffset + m);
                targetOffset += m;
                for (int row = 0; row < predicted.length / last; row++) {
                    int a = actual.length == predicted.length ? choice(actual, row, last) : (int) actual[row];
                    int p = choice(predicted, row, last);
                    confusion[a][p]++;
                }
            }
            rates[pass] = samples / Math.max((System.nanoTime() - start) / 1e9, 1e-9);
        }
        java.util.Arrays.sort(rates);
        double samplesPerSecond = rates[rates.length / 2];
        return new Report(task.name(), task.kind(), name, model.parameterCount(), trainSeconds,
                samplesPerSecond, evaluation, history, confusion, task.classNames());
    }

    private static int choice(float[] values, int row, int width) {
        int offset = row * width;
        if (width == 1) {
            return values[offset] > 0.5f ? 1 : 0;
        }
        int best = 0;
        for (int i = 1; i < width; i++) {
            if (values[offset + i] > values[offset + best]) {
                best = i;
            }
        }
        return best;
    }
}
