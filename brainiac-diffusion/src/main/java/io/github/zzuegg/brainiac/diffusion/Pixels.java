package io.github.zzuegg.brainiac.diffusion;

import java.util.function.DoubleUnaryOperator;

/**
 * Resizing and cropping of 8-bit images the way Pillow does it: a separable convolution whose
 * kernel widens with the downscale factor (so shrinking averages instead of skipping pixels),
 * the horizontal pass first, each pass rounded to bytes. Pipelines ported from Python resize
 * their inputs with Pillow, so this keeps their conditioning images close.
 */
public final class Pixels {

    /** A resampling filter: its kernel and the kernel's support at scale 1. */
    public enum Filter {
        /** Bicubic with {@code a = -0.5}, as Pillow's {@code BICUBIC} (CLIP's preprocessing). */
        BICUBIC(2, Pixels::cubic),
        /** Lanczos with three lobes, as Pillow's {@code LANCZOS} (diffusers' image processors). */
        LANCZOS(3, x -> Math.abs(x) < 3 ? sinc(x) * sinc(x / 3) : 0);

        final double support;
        final DoubleUnaryOperator kernel;

        Filter(double support, DoubleUnaryOperator kernel) {
            this.support = support;
            this.kernel = kernel;
        }
    }

    private Pixels() {
    }

    /**
     * Resizes an image; the same size returns it unchanged.
     *
     * @param image the image
     * @param width the new width
     * @param height the new height
     * @param filter the filter
     * @return the resized image, with the input's seed
     */
    public static GeneratedImage resize(GeneratedImage image, int width, int height, Filter filter) {
        if (width < 1 || height < 1) {
            throw new IllegalArgumentException("sizes must be positive, got " + width + "x" + height);
        }
        int w = image.width();
        int h = image.height();
        byte[] px = image.rgb();
        if (width != w) {
            px = horizontal(px, w, h, width, filter);
            w = width;
        }
        if (height != h) {
            px = vertical(px, w, h, height, filter);
        }
        return px == image.rgb() ? image : new GeneratedImage(width, height, px, image.seed());
    }

    /**
     * Cuts a rectangle out of an image.
     *
     * @param image the image
     * @param left the first column
     * @param top the first row
     * @param width the width
     * @param height the height
     * @return the rectangle
     */
    public static GeneratedImage crop(GeneratedImage image, int left, int top, int width, int height) {
        if (left < 0 || top < 0 || width < 1 || height < 1 || left + width > image.width()
                || top + height > image.height()) {
            throw new IllegalArgumentException("crop " + width + "x" + height + " at " + left + "," + top
                    + " is outside " + image.width() + "x" + image.height());
        }
        if (width == image.width() && height == image.height()) {
            return image;
        }
        byte[] out = new byte[width * height * 3];
        for (int y = 0; y < height; y++) {
            System.arraycopy(image.rgb(), ((top + y) * image.width() + left) * 3, out, y * width * 3, width * 3);
        }
        return new GeneratedImage(width, height, out, image.seed());
    }

    /**
     * Scales an image to cover {@code width x height} and cuts out the middle, keeping its
     * proportions: a photo of another shape loses a band at two sides instead of being
     * stretched.
     *
     * @param image the image
     * @param width the width
     * @param height the height
     * @param filter the filter
     * @return the image at that size
     */
    public static GeneratedImage cover(GeneratedImage image, int width, int height, Filter filter) {
        double scale = Math.max((double) width / image.width(), (double) height / image.height());
        int w = Math.max(width, (int) Math.round(image.width() * scale));
        int h = Math.max(height, (int) Math.round(image.height() * scale));
        GeneratedImage scaled = resize(image, w, h, filter);
        return crop(scaled, (w - width) / 2, (h - height) / 2, width, height);
    }

    private static byte[] horizontal(byte[] in, int w, int h, int out, Filter f) {
        Weights k = weights(w, out, f);
        byte[] r = new byte[out * h * 3];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < out; x++) {
                for (int c = 0; c < 3; c++) {
                    double s = 0;
                    for (int j = 0; j < k.count[x]; j++) {
                        s += k.weight[x][j] * (in[(y * w + k.first[x] + j) * 3 + c] & 0xff);
                    }
                    r[(y * out + x) * 3 + c] = clip(s);
                }
            }
        }
        return r;
    }

    private static byte[] vertical(byte[] in, int w, int h, int out, Filter f) {
        Weights k = weights(h, out, f);
        byte[] r = new byte[w * out * 3];
        for (int y = 0; y < out; y++) {
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    double s = 0;
                    for (int j = 0; j < k.count[y]; j++) {
                        s += k.weight[y][j] * (in[((k.first[y] + j) * w + x) * 3 + c] & 0xff);
                    }
                    r[(y * w + x) * 3 + c] = clip(s);
                }
            }
        }
        return r;
    }

    private record Weights(int[] first, int[] count, double[][] weight) {
    }

    /** Pillow's {@code precompute_coeffs}: each output pixel's input window and normalized weights. */
    private static Weights weights(int in, int out, Filter f) {
        double scale = (double) in / out;
        double filterScale = Math.max(scale, 1);
        double support = f.support * filterScale;
        int[] first = new int[out];
        int[] count = new int[out];
        double[][] weight = new double[out][];
        for (int i = 0; i < out; i++) {
            double center = (i + 0.5) * scale;
            int min = Math.max((int) (center - support + 0.5), 0);
            int max = Math.min((int) (center + support + 0.5), in);
            double[] k = new double[max - min];
            double total = 0;
            for (int x = 0; x < k.length; x++) {
                k[x] = f.kernel.applyAsDouble((x + min - center + 0.5) / filterScale);
                total += k[x];
            }
            for (int x = 0; x < k.length; x++) {
                k[x] = total != 0 ? k[x] / total : 0;
            }
            first[i] = min;
            count[i] = k.length;
            weight[i] = k;
        }
        return new Weights(first, count, weight);
    }

    private static byte clip(double v) {
        long r = Math.round(v);
        return (byte) (r < 0 ? 0 : r > 255 ? 255 : r);
    }

    private static double cubic(double x) {
        double a = -0.5;
        x = Math.abs(x);
        if (x < 1) {
            return ((a + 2) * x - (a + 3)) * x * x + 1;
        }
        if (x < 2) {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0;
    }

    private static double sinc(double x) {
        if (x == 0) {
            return 1;
        }
        double p = Math.PI * x;
        return Math.sin(p) / p;
    }
}
