package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.data.Standardizer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Ready-made benchmark tasks: every example's problem, with its data split into training
 * and test sets and its training settings as defaults. Hand any model to
 * {@link io.github.zzuegg.brainiac.train.Benchmark#run} with one of these.
 *
 * <table>
 *   <caption>Tasks</caption>
 *   <tr><th>task</th><th>input</th><th>output</th><th>what</th></tr>
 *   <tr><td>mnist</td><td>[1, 28, 28]</td><td>[10]</td><td>handwritten digits (downloads 11 MB)</td></tr>
 *   <tr><td>fashion</td><td>[1, 28, 28]</td><td>[10]</td><td>Fashion-MNIST clothing photos (downloads 30 MB)</td></tr>
 *   <tr><td>shapes</td><td>[1, 16, 16]</td><td>[3]</td><td>synthetic bars and crosses</td></tr>
 *   <tr><td>waves</td><td>[24, 1]</td><td>[1]</td><td>forecast the next value of a noisy wave</td></tr>
 *   <tr><td>reversal</td><td>[8, 6]</td><td>[8, 6]</td><td>reverse a sequence of digits</td></tr>
 *   <tr><td>sentiment</td><td>[12]</td><td>[2]</td><td>made-up reviews as word ids, with negation</td></tr>
 *   <tr><td>spiral</td><td>[2]</td><td>[3]</td><td>three interleaved spiral arms</td></tr>
 *   <tr><td>iris</td><td>[4]</td><td>[3]</td><td>iris species from flower measurements</td></tr>
 *   <tr><td>xor</td><td>[2]</td><td>[1]</td><td>exclusive or</td></tr>
 * </table>
 */
public final class Tasks {

    private Tasks() {
    }

    /** The task names, in the order {@link #all()} returns them. */
    public static final List<String> NAMES =
            List.of("mnist", "fashion", "shapes", "waves", "reversal", "sentiment", "spiral", "iris", "xor");

    /** Where Fashion-MNIST is downloaded to, relative to the working directory. */
    public static final Path FASHION_DIR = Path.of("data", "fashion-mnist");

    private static final String FASHION_MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/";

    private static final List<String> DIGITS = List.of("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");

    private static final List<String> CLOTHES = List.of("T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Boot");

    /**
     * Returns a task by name.
     *
     * @param name one of {@link #NAMES}
     * @return the task
     * @throws IllegalArgumentException for an unknown name
     */
    public static Task get(String name) {
        return switch (name) {
            case "mnist" -> mnist();
            case "fashion" -> fashionMnist();
            case "shapes" -> shapes();
            case "waves" -> waves();
            case "reversal" -> reversal();
            case "sentiment" -> SentimentExample.task();
            case "spiral" -> spiral();
            case "iris" -> iris();
            case "xor" -> xor();
            default -> throw new IllegalArgumentException("unknown task " + name + ", try one of " + NAMES);
        };
    }

    /**
     * Returns every task that needs no download.
     *
     * @return the tasks
     */
    public static List<Task> offline() {
        return List.of(shapes(), waves(), reversal(), SentimentExample.task(), spiral(), iris(), xor());
    }

    /**
     * Returns every task, downloading MNIST and Fashion-MNIST if needed.
     *
     * @return the tasks
     */
    public static List<Task> all() {
        return NAMES.stream().map(Tasks::get).toList();
    }

    /**
     * MNIST handwritten digits from {@link MnistExample#DATA_DIR}: 60,000 training and
     * 10,000 test images, 3 epochs of Adam by default.
     *
     * @return the task
     */
    public static Task mnist() {
        return mnist(MnistExample.DATA_DIR);
    }

    /**
     * MNIST handwritten digits.
     *
     * @param dir the directory with (or for) the four data files
     * @return the task
     */
    public static Task mnist(Path dir) {
        return images("mnist", dir, MnistExample.mirror())
                .describedAs("handwritten digits 0-9, 28x28 grey")
                .withClassNames(DIGITS);
    }

    /**
     * Fashion-MNIST: photos of ten kinds of clothing in the same format as MNIST, and
     * clearly harder. Downloaded into {@link #FASHION_DIR}.
     *
     * @return the task
     */
    public static Task fashionMnist() {
        return fashionMnist(FASHION_DIR);
    }

    /**
     * Fashion-MNIST.
     *
     * @param dir the directory with (or for) the four data files
     * @return the task
     */
    public static Task fashionMnist(Path dir) {
        return images("fashion", dir, FASHION_MIRROR)
                .describedAs("clothing photos in 10 classes, 28x28 grey")
                .withClassNames(CLOTHES);
    }

    private static Task images(String name, Path dir, String mirror) {
        Dataset.Split split;
        try {
            split = MnistExample.load(dir, mirror, Integer.MAX_VALUE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return Task.classification(name, split.train(), split.test(), Shape.of(1, 28, 28), Shape.of(10))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.002f)).batchSize(64).epochs(3));
    }

    /**
     * Synthetic 16x16 images of a horizontal bar, a vertical bar or a cross.
     *
     * @return the task
     */
    public static Task shapes() {
        return Task.classification("shapes", MnistExample.shapes(1500, 16, 1), MnistExample.shapes(500, 16, 2),
                        Shape.of(1, 16, 16), Shape.of(3))
                .describedAs("bars and crosses at random places, 16x16")
                .withClassNames(List.of("row", "column", "cross"))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.003f)).batchSize(32).epochs(8));
    }

    /**
     * Forecast the next value of a sum of two noisy sine waves from the last 24 values.
     *
     * @return the task
     */
    public static Task waves() {
        return Task.regression("waves", SequenceExample.waves(2000, 1), SequenceExample.waves(500, 2),
                        Shape.of(SequenceExample.WINDOW, 1), Shape.of(1))
                .describedAs("forecast the next value of a noisy wave from 24 steps")
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.01f)).clipGradientNorm(1).batchSize(32)
                        .epochs(30));
    }

    /**
     * Reverse a sequence of 8 digits from 0 to 5; scored per digit.
     *
     * @return the task
     */
    public static Task reversal() {
        return Task.classification("reversal", SequenceExample.reversal(4000, 3), SequenceExample.reversal(1000, 4),
                        Shape.of(SequenceExample.LENGTH, SequenceExample.DIGITS),
                        Shape.of(SequenceExample.LENGTH, SequenceExample.DIGITS))
                .describedAs("reverse 8 digits; accuracy per digit")
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.003f)).clipGradientNorm(1).batchSize(32)
                        .epochs(30));
    }

    /**
     * Classify points on three interleaved spiral arms.
     *
     * @return the task
     */
    public static Task spiral() {
        return Task.classification("spiral", SpiralExample.data(200, 3, 1), SpiralExample.data(100, 3, 2),
                        Shape.of(2), Shape.of(3))
                .describedAs("which of three spiral arms a point lies on")
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.01f)).batchSize(32).epochs(200));
    }

    /**
     * Iris species from four measurements, read from {@link IrisExample#DATA_FILE}: 120
     * flowers to train on and 30 to test, standardized on the training flowers.
     *
     * @return the task
     */
    public static Task iris() {
        Dataset all = Dataset.csv(IrisExample.DATA_FILE)
                .features("sepal_length", "sepal_width", "petal_length", "petal_width")
                .label("species").oneHot()
                .load();
        Dataset.Split split = all.shuffle(42).split(0.8);
        Standardizer standardizer = Standardizer.fit(split.train());
        return Task.classification("iris", split.train().standardize(standardizer),
                        split.test().standardize(standardizer), Shape.of(4), Shape.of(3))
                .describedAs("iris species from sepal and petal sizes")
                .withClassNames(all.classes())
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.01f)).batchSize(16).epochs(150));
    }

    /**
     * Exclusive or of two inputs, tested on the same four cases.
     *
     * @return the task
     */
    public static Task xor() {
        return Task.classification("xor", XorExample.DATA, XorExample.DATA, Shape.of(2), Shape.of(1))
                .describedAs("exclusive or; one sigmoid output")
                .withLoss(io.github.zzuegg.brainiac.nn.Loss.MSE)
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.05f)).batchSize(4).epochs(2000));
    }
}
