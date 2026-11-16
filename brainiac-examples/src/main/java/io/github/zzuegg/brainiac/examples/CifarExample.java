package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Node;
import io.github.zzuegg.brainiac.nn.Padding;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * CIFAR-10: 60,000 colour photos of 32x32 pixels in ten classes (50,000 to train on, 10,000
 * to test), the usual first "real" image benchmark. The binary version (about 160 MB) is
 * downloaded on first use into {@link #DATA_DIR}. A ResNet-20 ({@link #resnet}), trained
 * with shifted and mirrored copies of the photos, is known to reach about 91-92%.
 */
public final class CifarExample {

    private CifarExample() {
    }

    /** Where the data is downloaded to, relative to the working directory. */
    public static final Path DATA_DIR = Path.of("data", "cifar-10");

    /** The archive of the binary version. */
    public static final String URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

    /** The class names, in label order. */
    public static final List<String> CLASSES = List.of("airplane", "automobile", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck");

    /** Per-channel mean and standard deviation of the training photos, for normalizing. */
    private static final float[] MEAN = {0.4914f, 0.4822f, 0.4465f};
    private static final float[] STD = {0.2470f, 0.2435f, 0.2616f};

    private static final int PIXELS = 32 * 32;
    private static final int RECORD = 1 + 3 * PIXELS;

    /**
     * Downloads (if needed) and loads CIFAR-10, normalized per channel.
     *
     * @param dir the directory with (or for) the archive
     * @return the 50,000 training and 10,000 test photos, as {@code [3, 32, 32]} features
     *     and one-hot labels
     * @throws IOException if the data cannot be downloaded or read
     */
    public static Dataset.Split load(Path dir) throws IOException {
        Path archive = dir.resolve("cifar-10-binary.tar.gz");
        if (!Files.exists(archive)) {
            Files.createDirectories(dir);
            System.out.println("downloading " + URL);
            Path part = dir.resolve("cifar-10-binary.tar.gz.part");
            try (InputStream in = URI.create(URL).toURL().openStream()) {
                Files.copy(in, part, StandardCopyOption.REPLACE_EXISTING);
            }
            Files.move(part, archive, StandardCopyOption.REPLACE_EXISTING);
        }
        try (InputStream in = new GZIPInputStream(Files.newInputStream(archive), 1 << 16)) {
            return read(in);
        }
    }

    /**
     * Reads the tar archive of the binary version: {@code data_batch_1.bin} to
     * {@code data_batch_5.bin} for training and {@code test_batch.bin} for testing, each a
     * sequence of records of one label byte and 3,072 pixel bytes (red, green, blue planes).
     *
     * @param tar the uncompressed archive
     * @return the training and test photos
     * @throws IOException if the archive cannot be read or misses a file
     */
    static Dataset.Split read(InputStream tar) throws IOException {
        List<byte[]> train = new ArrayList<>();
        byte[] test = null;
        byte[] header = new byte[512];
        while (tar.readNBytes(header, 0, 512) == 512) {
            String name = field(header, 0, 100);
            if (name.isEmpty()) {
                break;   // the two empty blocks that end an archive
            }
            long size = Long.parseLong(field(header, 124, 12).trim(), 8);
            String file = name.substring(name.lastIndexOf('/') + 1);
            byte[] data = tar.readNBytes(Math.toIntExact(size));
            if (data.length != size) {
                throw new IOException("CIFAR-10 archive ends inside " + name);
            }
            tar.skipNBytes((512 - size % 512) % 512);
            if (file.matches("data_batch_[1-5]\\.bin")) {
                train.add(data);
            } else if (file.equals("test_batch.bin")) {
                test = data;
            }
        }
        if (train.size() != 5 || test == null) {
            throw new IOException("CIFAR-10 archive is missing files: found " + train.size()
                    + " of 5 training batches" + (test == null ? " and no test batch" : ""));
        }
        ByteArrayOutputStream all = new ByteArrayOutputStream();
        for (byte[] b : train) {
            all.writeBytes(b);
        }
        return new Dataset.Split(decode(all.toByteArray()), decode(test));
    }

    private static String field(byte[] header, int from, int length) {
        int end = from;
        while (end < from + length && header[end] != 0) {
            end++;
        }
        return new String(header, from, end - from, StandardCharsets.US_ASCII);
    }

    /** Records of a label byte and 3,072 pixel bytes into normalized features and one-hot labels. */
    static Dataset decode(byte[] records) throws IOException {
        if (records.length == 0 || records.length % RECORD != 0) {
            throw new IOException("CIFAR-10 batch of " + records.length + " bytes is not whole records");
        }
        int n = records.length / RECORD;
        float[] features = new float[n * 3 * PIXELS];
        float[] targets = new float[n * 10];
        for (int i = 0; i < n; i++) {
            int r = i * RECORD;
            int label = records[r] & 0xff;
            if (label > 9) {
                throw new IOException("CIFAR-10 label " + label + " in record " + i);
            }
            targets[i * 10 + label] = 1f;
            for (int c = 0; c < 3; c++) {
                int from = r + 1 + c * PIXELS;
                int to = (i * 3 + c) * PIXELS;
                for (int p = 0; p < PIXELS; p++) {
                    features[to + p] = ((records[from + p] & 0xff) / 255f - MEAN[c]) / STD[c];
                }
            }
        }
        return Dataset.ofRows(features, 3 * PIXELS, targets, 10);
    }

    /**
     * A residual network for small photos as in "Deep Residual Learning" (He et al. 2015):
     * a 3x3 convolution, then three stages of {@code blocks} residual blocks with 16, 32
     * and 64 channels (the second and third halve the image), global average pooling and
     * the classifier. {@code blocks = 3} is ResNet-20 (about 0.27 million parameters).
     *
     * @param channels the input channels
     * @param size the image width and height
     * @param classes the number of classes
     * @param blocks the residual blocks per stage
     * @return an untrained model
     */
    public static Model resnet(int channels, int size, int classes, int blocks) {
        Node input = Node.input(channels, size, size);
        Node x = convBn(input, 16, 3, 1).activation(Activation.RELU);
        int width = 16;
        for (int stage = 0; stage < 3; stage++) {
            for (int b = 0; b < blocks; b++) {
                int stride = stage > 0 && b == 0 ? 2 : 1;
                int out = 16 << stage;
                Node y = convBn(x, out, 3, stride).activation(Activation.RELU);
                y = convBn(y, out, 3, 1);
                // the shortcut: the input itself, or a strided 1x1 projection where the shape changes
                Node shortcut = stride == 1 && out == width ? x : convBn(x, out, 1, stride);
                x = y.add(shortcut).activation(Activation.RELU);
                width = out;
            }
        }
        Node output = x.globalAvgPool().dense(classes, Activation.SOFTMAX);
        return Model.graph(input, output).seed(42).build();
    }

    private static Node convBn(Node x, int filters, int kernel, int stride) {
        return x.add(Conv.of(filters, kernel, kernel).stride(stride, stride).padding(Padding.SAME).bias(false))
                .batchNorm();
    }
}
