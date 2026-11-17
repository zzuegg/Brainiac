package io.github.zzuegg.brainiac.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

/**
 * Reads and writes images (PNG, JPEG, BMP, GIF: what the JDK's ImageIO reads) as tensors
 * of values in {@code [0, 1]}, channels first: {@code [channels, height, width]}, as
 * convolution layers take them.
 */
public final class Images {

    private Images() {
    }

    /**
     * Reads an image.
     *
     * @param file the image file
     * @param channels 1 for grey, 3 for RGB, 4 for RGBA
     * @return {@code [channels, height, width]}
     * @throws IOException if the file cannot be read or is not an image
     */
    public static Tensor read(Path file, int channels) throws IOException {
        return Tensor.of(shape(read(file), channels), pixels(read(file), channels));
    }

    /**
     * Reads an image and scales it to a size.
     *
     * @param file the image file
     * @param channels 1, 3 or 4
     * @param height the height to scale to
     * @param width the width to scale to
     * @return {@code [channels, height, width]}
     * @throws IOException if the file cannot be read or is not an image
     */
    public static Tensor read(Path file, int channels, int height, int width) throws IOException {
        BufferedImage img = scaled(read(file), height, width);
        return Tensor.of(Shape.of(channels, height, width), pixels(img, channels));
    }

    private static BufferedImage read(Path file) throws IOException {
        BufferedImage img = ImageIO.read(file.toFile());
        if (img == null) {
            throw new IOException(file + " is not an image ImageIO can read");
        }
        return img;
    }

    private static Shape shape(BufferedImage img, int channels) {
        return Shape.of(channels, img.getHeight(), img.getWidth());
    }

    private static BufferedImage scaled(BufferedImage img, int height, int width) {
        if (img.getHeight() == height && img.getWidth() == width) {
            return img;
        }
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, width, height, null);
        g.dispose();
        return out;
    }

    /** Channels-first values in [0, 1]; grey is the mean of red, green and blue. */
    private static float[] pixels(BufferedImage img, int channels) {
        if (channels != 1 && channels != 3 && channels != 4) {
            throw new IllegalArgumentException("channels must be 1, 3 or 4, got " + channels);
        }
        int h = img.getHeight();
        int w = img.getWidth();
        int plane = h * w;
        float[] out = new float[channels * plane];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int argb = img.getRGB(x, y);
                float a = (argb >>> 24) / 255f;
                float r = (argb >> 16 & 0xff) / 255f;
                float g = (argb >> 8 & 0xff) / 255f;
                float b = (argb & 0xff) / 255f;
                int i = y * w + x;
                if (channels == 1) {
                    out[i] = (r + g + b) / 3;
                } else {
                    out[i] = r;
                    out[plane + i] = g;
                    out[2 * plane + i] = b;
                    if (channels == 4) {
                        out[3 * plane + i] = a;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Writes an image.
     *
     * @param image {@code [channels, height, width]} with 1, 3 or 4 channels, values in
     *     {@code [0, 1]} (clamped)
     * @param file where to write; the format comes from the extension (png, jpg, bmp, gif)
     * @throws IOException if it cannot be written
     */
    public static void write(Tensor image, Path file) throws IOException {
        Shape s = image.shape();
        if (s.rank() != 3 || s.dim(0) != 1 && s.dim(0) != 3 && s.dim(0) != 4) {
            throw new IllegalArgumentException("needs [1|3|4, height, width], got " + s);
        }
        int c = s.dim(0);
        int h = s.dim(1);
        int w = s.dim(2);
        float[] v = image.toArray();
        String name = file.getFileName().toString();
        String format = name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT);
        boolean alpha = c == 4 && format.equals("png");
        BufferedImage img = new BufferedImage(w, h, alpha ? BufferedImage.TYPE_INT_ARGB : BufferedImage.TYPE_INT_RGB);
        int plane = h * w;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                int r = byteOf(v[i]);
                int g = c == 1 ? r : byteOf(v[plane + i]);
                int b = c == 1 ? r : byteOf(v[2 * plane + i]);
                int a = c == 4 ? byteOf(v[3 * plane + i]) : 255;
                img.setRGB(x, y, a << 24 | r << 16 | g << 8 | b);
            }
        }
        if (!ImageIO.write(img, format.equals("jpg") ? "jpeg" : format, file.toFile())) {
            throw new IOException("no ImageIO writer for " + format);
        }
    }

    private static int byteOf(float v) {
        return Math.round(Math.clamp(v, 0f, 1f) * 255);
    }

    /**
     * Loads an image classification dataset laid out as one folder per class, the class
     * names in sorted order giving the ids 0, 1, ... (as torchvision's ImageFolder).
     *
     * @param root the folder holding the class folders
     * @param channels 1, 3 or 4
     * @param height the height every image is scaled to
     * @param width the width
     * @return features {@code [n, channels * height * width]} (flat rows, which a model with
     *     input shape {@code [channels, height, width]} reshapes) and targets {@code [n, 1]}
     *     class ids; {@link #classes} gives the names
     * @throws IOException if a file cannot be read
     */
    public static Dataset folder(Path root, int channels, int height, int width) throws IOException {
        List<String> names = classes(root);
        List<float[]> rows = new ArrayList<>();
        List<Float> ids = new ArrayList<>();
        for (int c = 0; c < names.size(); c++) {
            try (Stream<Path> files = Files.list(root.resolve(names.get(c)))) {
                for (Path f : files.filter(Files::isRegularFile).sorted().toList()) {
                    if (!isImage(f)) {
                        continue;
                    }
                    rows.add(pixels(scaled(read(f), height, width), channels));
                    ids.add((float) c);
                }
            }
        }
        int per = channels * height * width;
        float[] features = new float[rows.size() * per];
        float[] targets = new float[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, features, i * per, per);
            targets[i] = ids.get(i);
        }
        return Dataset.of(Tensor.of(Shape.of(rows.size(), per), features), Tensor.of(Shape.of(rows.size(), 1), targets));
    }

    /**
     * The class names of a {@link #folder} dataset: its subfolders, sorted.
     *
     * @param root the folder holding the class folders
     * @return the names; index = class id
     * @throws IOException if the folder cannot be listed
     */
    public static List<String> classes(Path root) throws IOException {
        try (Stream<Path> dirs = Files.list(root)) {
            return dirs.filter(Files::isDirectory).map(p -> p.getFileName().toString()).sorted().toList();
        }
    }

    private static boolean isImage(Path f) {
        String n = f.getFileName().toString().toLowerCase(Locale.ROOT);
        return n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".bmp")
                || n.endsWith(".gif");
    }
}
