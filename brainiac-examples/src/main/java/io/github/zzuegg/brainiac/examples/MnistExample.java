package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.train.Evaluation;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Progress;
import io.github.zzuegg.brainiac.train.Trainer;
import io.github.zzuegg.brainiac.train.TrainingResult;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.SplittableRandom;

/**
 * Handwritten digit recognition with a convolutional network on MNIST: two convolution
 * blocks with batch normalization and max pooling, then a small classifier. Reaches
 * about 99% test accuracy in a few epochs.
 *
 * <p>The data (about 11 MB) is downloaded on first use into {@code data/mnist}, or read
 * from the directory given as the second argument:
 * {@code ./gradlew :brainiac-examples:run --args="mnist [dir] [epochs] [train-limit]"}.
 *
 * <p>{@link #shapes} generates a small synthetic image task (bars and crosses at random
 * places) that the tests train on without a download.
 */
public final class MnistExample {

    private MnistExample() {
    }

    /** Where the data is downloaded to, relative to the working directory. */
    public static final Path DATA_DIR = Path.of("data", "mnist");

    private static final String MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/";

    private static final String[] FILES = {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};

    /**
     * Builds the network for square grey images.
     *
     * @param size the image width and height
     * @param classes the number of classes
     * @return an untrained model
     */
    public static Model model(int size, int classes) {
        return model(1, size, classes);
    }

    /**
     * Builds the network for square images.
     *
     * @param channels the input channels
     * @param size the image width and height
     * @param classes the number of classes
     * @return an untrained model
     */
    public static Model model(int channels, int size, int classes) {
        return Model.sequential()
                .input(channels, size, size)
                .add(Conv.of(16, 3, 3).padding(Padding.SAME).bias(false))
                .batchNorm()
                .activation(Activation.RELU)
                .maxPool(2, 2)
                .add(Conv.of(32, 3, 3).padding(Padding.SAME).bias(false))
                .batchNorm()
                .activation(Activation.RELU)
                .maxPool(2, 2)
                .flatten()
                .dropout(0.25f)
                .dense(64, Activation.RELU)
                .dense(classes, Activation.SOFTMAX)
                .seed(42)
                .build();
    }

    /**
     * Trains a model.
     *
     * @param model the model
     * @param train the training data
     * @param test the data to validate on after each epoch, or {@code null}
     * @param epochs the number of epochs
     * @param verbose whether to print progress
     * @return the training result
     */
    public static TrainingResult train(Model model, Dataset train, Dataset test, int epochs, boolean verbose) {
        Trainer trainer = Trainer.of(model)
                .loss(Loss.CROSS_ENTROPY)
                .optimizer(Optimizer.adam(0.002f))
                .batchSize(64)
                .epochs(epochs)
                .metric(Metric.ACCURACY)
                .seed(1);
        if (test != null) {
            trainer.validation(test);
        }
        if (verbose) {
            trainer.listener(Progress.every(1));
        }
        return trainer.fit(train);
    }

    /**
     * Generates a synthetic image task: a horizontal bar, a vertical bar or a cross at a
     * random place, on a noisy background.
     *
     * @param samples the number of images
     * @param size the image width and height
     * @param seed the random seed
     * @return images flattened to {@code size * size} features, with 3 one-hot classes
     */
    public static Dataset shapes(int samples, int size, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] x = new float[samples][size * size];
        float[][] y = new float[samples][3];
        for (int s = 0; s < samples; s++) {
            int label = random.nextInt(3);
            for (int i = 0; i < size * size; i++) {
                x[s][i] = (float) random.nextDouble(0, 0.3);
            }
            int length = 4 + random.nextInt(size / 2 - 3);
            int row = random.nextInt(size);
            int col = random.nextInt(size);
            int start = random.nextInt(size - length + 1);
            if (label == 0 || label == 2) {
                for (int i = 0; i < length; i++) {
                    x[s][row * size + start + i] = 1;
                }
            }
            if (label == 1 || label == 2) {
                for (int i = 0; i < length; i++) {
                    x[s][(start + i) * size + col] = 1;
                }
            }
            y[s][label] = 1;
        }
        return Dataset.of(x, y);
    }

    /**
     * Downloads MNIST into a directory unless it is there already.
     *
     * @param dir the directory
     * @throws IOException if the download fails
     */
    public static void download(Path dir) throws IOException {
        download(dir, MIRROR);
    }

    /**
     * Downloads the four MNIST-format files from a mirror into a directory, skipping files
     * that are there already. Fashion-MNIST uses the same file names.
     *
     * @param dir the directory
     * @param mirror the base URL, ending in {@code /}
     * @throws IOException if the download fails
     */
    public static void download(Path dir, String mirror) throws IOException {
        Files.createDirectories(dir);
        for (String file : FILES) {
            Path target = dir.resolve(file);
            if (Files.exists(target)) {
                continue;
            }
            System.out.println("downloading " + mirror + file);
            Path part = dir.resolve(file + ".part");
            try (InputStream in = URI.create(mirror + file).toURL().openStream()) {
                Files.copy(in, part, StandardCopyOption.REPLACE_EXISTING);
            }
            Files.move(part, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Downloads if needed and loads MNIST-format data.
     *
     * @param dir the directory with (or for) the four files
     * @param mirror where to download missing files from
     * @param trainLimit the most training images to load
     * @return the 60,000 training and 10,000 test images
     * @throws IOException if the data cannot be downloaded or read
     */
    public static Dataset.Split load(Path dir, String mirror, int trainLimit) throws IOException {
        download(dir, mirror);
        return new Dataset.Split(Dataset.idx(dir.resolve(FILES[0]), dir.resolve(FILES[1]), trainLimit),
                Dataset.idx(dir.resolve(FILES[2]), dir.resolve(FILES[3])));
    }

    /** @return the default MNIST mirror */
    public static String mirror() {
        return MIRROR;
    }

    /**
     * Runs the example.
     *
     * @param args optional data directory, epochs and a limit on training samples
     * @throws IOException if the data cannot be downloaded or read
     */
    public static void main(String[] args) throws IOException {
        Path dir = args.length > 0 ? Path.of(args[0]) : DATA_DIR;
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        int limit = args.length > 2 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE;
        download(dir);
        Dataset train = Dataset.idx(dir.resolve(FILES[0]), dir.resolve(FILES[1]), limit);
        Dataset test = Dataset.idx(dir.resolve(FILES[2]), dir.resolve(FILES[3]));
        Model model = model(28, 10);
        System.out.println(model.summary());
        System.out.printf("training on %d images, testing on %d%n", train.size(), test.size());
        long start = System.nanoTime();
        train(model, train, test, epochs, true);
        Evaluation result = Trainer.of(model).loss(Loss.CROSS_ENTROPY).metric(Metric.ACCURACY).evaluate(test);
        System.out.printf("%ntest accuracy %.2f%% after %.0f s%n", 100 * result.get(Metric.ACCURACY),
                (System.nanoTime() - start) / 1e9);
    }
}
