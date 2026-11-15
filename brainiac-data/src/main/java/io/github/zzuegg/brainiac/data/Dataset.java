package io.github.zzuegg.brainiac.data;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Training data: one row of features and one row of targets per sample.
 *
 * <p>A dataset is immutable and keeps its values on the Java heap; {@link #batches} turns
 * slices of it into tensors on a device.
 *
 * <pre>{@code
 * Dataset xor = Dataset.of(
 *         new float[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
 *         new float[][] {{0}, {1}, {1}, {0}});
 *
 * Dataset iris = Dataset.csv(Path.of("iris.csv"))
 *         .features("sepal_length", "sepal_width", "petal_length", "petal_width")
 *         .label("species").oneHot()
 *         .load();
 * Dataset.Split split = iris.shuffle(42).split(0.8);
 * }</pre>
 */
public final class Dataset {

    private final float[] features;
    private final float[] targets;
    private final int size;
    private final int featureCount;
    private final int targetCount;
    private final List<String> classes;

    private Dataset(float[] features, float[] targets, int size, int featureCount, int targetCount, List<String> classes) {
        this.features = features;
        this.targets = targets;
        this.size = size;
        this.featureCount = featureCount;
        this.targetCount = targetCount;
        this.classes = classes;
    }

    /**
     * Creates a dataset from rows.
     *
     * @param features one row of features per sample
     * @param targets one row of targets per sample
     * @return the dataset
     */
    public static Dataset of(float[][] features, float[][] targets) {
        if (features.length != targets.length) {
            throw new IllegalArgumentException(
                    features.length + " feature rows but " + targets.length + " target rows");
        }
        if (features.length == 0) {
            throw new IllegalArgumentException("a dataset needs at least one sample");
        }
        return new Dataset(flatten(features, "features"), flatten(targets, "targets"),
                features.length, features[0].length, targets[0].length, List.of());
    }

    /**
     * Creates a dataset from two matrices with one row per sample.
     *
     * @param features shape {@code [samples, featureCount]}
     * @param targets shape {@code [samples, targetCount]}
     * @return the dataset
     */
    public static Dataset of(Tensor features, Tensor targets) {
        if (features.rank() != 2 || targets.rank() != 2 || features.shape().dim(0) != targets.shape().dim(0)) {
            throw new IllegalArgumentException("need [samples, features] and [samples, targets], got "
                    + features.shape() + " and " + targets.shape());
        }
        return new Dataset(features.toArray(), targets.toArray(), features.shape().dim(0),
                features.shape().dim(1), targets.shape().dim(1), List.of());
    }

    /**
     * Starts loading a dataset from a CSV file with a header row.
     *
     * @param file the file
     * @return a loader to choose columns with
     */
    public static CsvLoader csv(Path file) {
        return new CsvLoader(file);
    }

    /**
     * Loads an image dataset in the IDX format of MNIST and Fashion-MNIST (plain or
     * gzipped). Pixels are scaled to {@code [0, 1]} and flattened, so use a model with
     * {@code input(1, 28, 28)}; labels become one-hot rows.
     *
     * @param images the image file, such as {@code train-images-idx3-ubyte.gz}
     * @param labels the label file, such as {@code train-labels-idx1-ubyte.gz}
     * @return the dataset
     * @throws IOException if a file cannot be read or is not in IDX format
     */
    public static Dataset idx(Path images, Path labels) throws IOException {
        return IdxLoader.load(images, labels, Integer.MAX_VALUE);
    }

    /**
     * Loads at most {@code limit} samples of an IDX dataset; see {@link #idx(Path, Path)}.
     *
     * @param images the image file
     * @param labels the label file
     * @param limit the most samples to read
     * @return the dataset
     * @throws IOException if a file cannot be read or is not in IDX format
     */
    public static Dataset idx(Path images, Path labels, int limit) throws IOException {
        return IdxLoader.load(images, labels, limit);
    }

    static Dataset create(float[] features, float[] targets, int size, int featureCount, int targetCount, List<String> classes) {
        return new Dataset(features, targets, size, featureCount, targetCount, List.copyOf(classes));
    }

    private static float[] flatten(float[][] rows, String what) {
        int width = rows[0].length;
        float[] flat = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i].length != width) {
                throw new IllegalArgumentException(what + " row " + i + " has " + rows[i].length
                        + " values, expected " + width);
            }
            System.arraycopy(rows[i], 0, flat, i * width, width);
        }
        return flat;
    }

    // ---------------------------------------------------------------- properties

    /** @return the number of samples */
    public int size() {
        return size;
    }

    /** @return the number of features per sample */
    public int featureCount() {
        return featureCount;
    }

    /** @return the number of target values per sample */
    public int targetCount() {
        return targetCount;
    }

    /**
     * Returns the class names for one-hot targets, in target column order.
     *
     * @return the class names; empty unless loaded with {@link CsvLoader#oneHot()}
     */
    public List<String> classes() {
        return classes;
    }

    /**
     * Returns all features as a matrix on the default device.
     *
     * @return shape {@code [size, featureCount]}
     */
    public Tensor features() {
        return Tensor.of(Shape.of(size, featureCount), features);
    }

    /**
     * Returns all targets as a matrix on the default device.
     *
     * @return shape {@code [size, targetCount]}
     */
    public Tensor targets() {
        return Tensor.of(Shape.of(size, targetCount), targets);
    }

    /**
     * Returns one sample's features.
     *
     * @param index the sample index
     * @return a copy of the features
     */
    public float[] featuresOf(int index) {
        float[] row = new float[featureCount];
        System.arraycopy(features, Objects.checkIndex(index, size) * featureCount, row, 0, featureCount);
        return row;
    }

    /**
     * Returns one sample's targets.
     *
     * @param index the sample index
     * @return a copy of the targets
     */
    public float[] targetsOf(int index) {
        float[] row = new float[targetCount];
        System.arraycopy(targets, Objects.checkIndex(index, size) * targetCount, row, 0, targetCount);
        return row;
    }

    // ---------------------------------------------------------------- reordering and splitting

    /**
     * Returns the samples in a random order.
     *
     * @param seed the random seed; the same seed gives the same order
     * @return the shuffled dataset
     */
    public Dataset shuffle(long seed) {
        int[] order = new int[size];
        for (int i = 0; i < size; i++) {
            order[i] = i;
        }
        SplittableRandom random = new SplittableRandom(seed);
        for (int i = size - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        return select(order);
    }

    /**
     * Returns samples {@code from} (inclusive) to {@code to} (exclusive).
     *
     * @param from the first index
     * @param to the end index
     * @return the slice
     */
    public Dataset slice(int from, int to) {
        Objects.checkFromToIndex(from, to, size);
        if (from == to) {
            throw new IllegalArgumentException("slice " + from + ".." + to + " is empty");
        }
        int n = to - from;
        float[] f = new float[n * featureCount];
        float[] t = new float[n * targetCount];
        System.arraycopy(features, from * featureCount, f, 0, f.length);
        System.arraycopy(targets, from * targetCount, t, 0, t.length);
        return new Dataset(f, t, n, featureCount, targetCount, classes);
    }

    /** A dataset split into a training part and a test part. */
    public record Split(Dataset train, Dataset test) {
    }

    /**
     * Splits into a training part with the first {@code trainFraction} of the samples and
     * a test part with the rest. Shuffle first if the file is sorted.
     *
     * @param trainFraction the share of samples for training, between 0 and 1
     * @return the two parts
     */
    public Split split(double trainFraction) {
        if (!(trainFraction > 0 && trainFraction < 1)) {
            throw new IllegalArgumentException("trainFraction must be between 0 and 1, got " + trainFraction);
        }
        int cut = (int) Math.round(size * trainFraction);
        if (cut == 0 || cut == size) {
            throw new IllegalArgumentException("splitting " + size + " samples at " + trainFraction + " leaves one part empty");
        }
        return new Split(slice(0, cut), slice(cut, size));
    }

    private Dataset select(int[] order) {
        float[] f = new float[features.length];
        float[] t = new float[targets.length];
        for (int i = 0; i < order.length; i++) {
            System.arraycopy(features, order[i] * featureCount, f, i * featureCount, featureCount);
            System.arraycopy(targets, order[i] * targetCount, t, i * targetCount, targetCount);
        }
        return new Dataset(f, t, size, featureCount, targetCount, classes);
    }

    // ---------------------------------------------------------------- batching and preprocessing

    /**
     * Cuts the dataset into batches on the default device, in order. The last batch may be
     * smaller.
     *
     * @param batchSize the samples per batch
     * @return the batches
     */
    public List<Batch> batches(int batchSize) {
        return batches(batchSize, Device.getDefault());
    }

    /**
     * Cuts the dataset into batches on a device, in order. The last batch may be smaller.
     *
     * @param batchSize the samples per batch
     * @param device where the batch tensors live
     * @return the batches
     */
    public List<Batch> batches(int batchSize, Device device) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be at least 1, got " + batchSize);
        }
        List<Batch> out = new ArrayList<>();
        for (int from = 0; from < size; from += batchSize) {
            int n = Math.min(batchSize, size - from);
            float[] f = new float[n * featureCount];
            float[] t = new float[n * targetCount];
            System.arraycopy(features, from * featureCount, f, 0, f.length);
            System.arraycopy(targets, from * targetCount, t, 0, t.length);
            out.add(new Batch(Tensor.of(Shape.of(n, featureCount), f, device),
                    Tensor.of(Shape.of(n, targetCount), t, device)));
        }
        return out;
    }

    /**
     * Returns the dataset with features transformed by a standardizer.
     *
     * @param standardizer the scaling to apply, usually fitted on the training data
     * @return the scaled dataset
     */
    public Dataset standardize(Standardizer standardizer) {
        return new Dataset(standardizer.apply(features, featureCount), targets, size, featureCount, targetCount, classes);
    }

    float[] rawFeatures() {
        return features;
    }

    @Override
    public String toString() {
        return "Dataset(" + size + " samples, " + featureCount + " features, " + targetCount + " targets"
                + (classes.isEmpty() ? "" : ", classes " + classes) + ")";
    }
}
