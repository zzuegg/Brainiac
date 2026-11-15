package io.github.zzuegg.brainiac.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Reads image datasets in the IDX format used by MNIST, Fashion-MNIST and EMNIST, plain
 * or gzipped. Pixels become floats in {@code [0, 1]} and labels one-hot rows.
 *
 * <pre>{@code
 * Dataset train = Dataset.idx(Path.of("train-images-idx3-ubyte.gz"), Path.of("train-labels-idx1-ubyte.gz"));
 * // features: 784 per sample (use a model with input(1, 28, 28)); targets: 10 one-hot
 * }</pre>
 */
final class IdxLoader {

    private IdxLoader() {
    }

    static Dataset load(Path images, Path labels, int limit) throws IOException {
        int[] dims;
        float[] pixels;
        try (DataInputStream in = open(images)) {
            int magic = in.readInt();
            if ((magic >>> 8) != 0x08) {
                throw new IOException(images + " is not an IDX file of unsigned bytes");
            }
            int rank = magic & 0xFF;
            dims = new int[rank];
            for (int i = 0; i < rank; i++) {
                dims[i] = in.readInt();
            }
            int count = Math.min(dims[0], limit);
            long per = 1;
            for (int i = 1; i < rank; i++) {
                per *= dims[i];
            }
            if ((long) count * per > Integer.MAX_VALUE - 8) {
                throw new IOException("too many pixels; pass a limit");
            }
            byte[] raw = in.readNBytes((int) (count * per));
            if (raw.length != count * per) {
                throw new IOException(images + " ends early");
            }
            pixels = new float[raw.length];
            for (int i = 0; i < raw.length; i++) {
                pixels[i] = (raw[i] & 0xFF) / 255f;
            }
            dims[0] = count;
        }
        byte[] rawLabels;
        try (DataInputStream in = open(labels)) {
            if (in.readInt() != 0x00000801) {
                throw new IOException(labels + " is not an IDX label file");
            }
            int count = in.readInt();
            if (count < dims[0]) {
                throw new IOException(labels + " has " + count + " labels for " + dims[0] + " images");
            }
            rawLabels = in.readNBytes(dims[0]);
        }
        int classes = 0;
        for (byte b : rawLabels) {
            classes = Math.max(classes, (b & 0xFF) + 1);
        }
        float[] targets = new float[rawLabels.length * classes];
        for (int i = 0; i < rawLabels.length; i++) {
            targets[i * classes + (rawLabels[i] & 0xFF)] = 1;
        }
        List<String> names = new ArrayList<>();
        for (int i = 0; i < classes; i++) {
            names.add(Integer.toString(i));
        }
        int per = pixels.length / dims[0];
        return Dataset.create(pixels, targets, dims[0], per, classes, names);
    }

    private static DataInputStream open(Path file) throws IOException {
        InputStream raw = new BufferedInputStream(Files.newInputStream(file));
        raw.mark(2);
        int b0 = raw.read();
        int b1 = raw.read();
        raw.reset();
        boolean gz = b0 == 0x1f && b1 == 0x8b;
        return new DataInputStream(new BufferedInputStream(gz ? new GZIPInputStream(raw) : raw));
    }
}
