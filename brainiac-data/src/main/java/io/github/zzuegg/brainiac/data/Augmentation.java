package io.github.zzuegg.brainiac.data;

import java.util.SplittableRandom;

/**
 * Changes a training sample at random, such as shifting or mirroring an image, so a model
 * sees a slightly different version of every sample in every epoch. Applied while batches
 * are cut (see {@link Dataset#batches(int, io.github.zzuegg.brainiac.tensor.Device,
 * Augmentation, long)}), so it costs no extra copy of the dataset.
 */
@FunctionalInterface
public interface Augmentation {

    /**
     * Transforms one sample's features in place.
     *
     * @param features the array holding the sample
     * @param offset where the sample starts in it
     * @param random the source of randomness, seeded per epoch
     */
    void apply(float[] features, int offset, SplittableRandom random);

    /**
     * The standard augmentation for small photos (as for CIFAR-10): a random shift of up to
     * {@code pad} pixels in each direction, filling the uncovered border with zeros (as if
     * cropped from an image padded with zeros), and a horizontal mirror half the time.
     *
     * @param channels the channels, stored first ({@code [channels, height, width]})
     * @param height the image height
     * @param width the image width
     * @param pad the largest shift
     * @return the augmentation
     */
    static Augmentation cropAndFlip(int channels, int height, int width, int pad) {
        if (channels < 1 || height < 1 || width < 1 || pad < 0) {
            throw new IllegalArgumentException("invalid image " + channels + "x" + height + "x" + width
                    + " or pad " + pad);
        }
        int plane = height * width;
        return (features, offset, random) -> {
            int dy = random.nextInt(2 * pad + 1) - pad;
            int dx = random.nextInt(2 * pad + 1) - pad;
            boolean flip = random.nextBoolean();
            if (dy == 0 && dx == 0 && !flip) {
                return;
            }
            float[] copy = new float[plane];
            for (int c = 0; c < channels; c++) {
                int base = offset + c * plane;
                System.arraycopy(features, base, copy, 0, plane);
                for (int y = 0; y < height; y++) {
                    int sy = y + dy;
                    for (int x = 0; x < width; x++) {
                        int sx = (flip ? width - 1 - x : x) + dx;
                        features[base + y * width + x] = sy >= 0 && sy < height && sx >= 0 && sx < width
                                ? copy[sy * width + sx] : 0f;
                    }
                }
            }
        };
    }

    /**
     * @param next the augmentation applied after this one
     * @return both, in order
     */
    default Augmentation then(Augmentation next) {
        return (features, offset, random) -> {
            apply(features, offset, random);
            next.apply(features, offset, random);
        };
    }

    private static void checkImage(int channels, int height, int width) {
        if (channels < 1 || height < 1 || width < 1) {
            throw new IllegalArgumentException("invalid image " + channels + "x" + height + "x" + width);
        }
    }

    /**
     * Mirrors the image left to right half the time.
     *
     * @param channels the channels, stored first
     * @param height the image height
     * @param width the image width
     * @return the augmentation
     */
    static Augmentation horizontalFlip(int channels, int height, int width) {
        checkImage(channels, height, width);
        return (f, offset, random) -> {
            if (!random.nextBoolean()) {
                return;
            }
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    int row = offset + (c * height + y) * width;
                    for (int x = 0; x < width / 2; x++) {
                        float t = f[row + x];
                        f[row + x] = f[row + width - 1 - x];
                        f[row + width - 1 - x] = t;
                    }
                }
            }
        };
    }

    /**
     * The crop of Inception training: a random region covering a share of the area in
     * {@code [minScale, maxScale]} with an aspect ratio between 3/4 and 4/3 (log-uniform),
     * resized back to the full image bilinearly.
     *
     * @param channels the channels, stored first
     * @param height the image height
     * @param width the image width
     * @param minScale the smallest share of the area, such as 0.08
     * @param maxScale the largest share, such as 1
     * @return the augmentation
     */
    static Augmentation randomResizedCrop(int channels, int height, int width, double minScale, double maxScale) {
        checkImage(channels, height, width);
        if (!(minScale > 0 && minScale <= maxScale && maxScale <= 1)) {
            throw new IllegalArgumentException("need 0 < minScale <= maxScale <= 1, got " + minScale + ", " + maxScale);
        }
        int plane = height * width;
        return (f, offset, random) -> {
            double area = height * (double) width;
            double cw = width;
            double ch = height;
            for (int attempt = 0; attempt < 10; attempt++) {
                double target = area * (minScale + random.nextDouble() * (maxScale - minScale));
                double ratio = Math.exp(Math.log(3 / 4.0) + random.nextDouble() * Math.log(16 / 9.0));
                double w = Math.sqrt(target * ratio);
                double h = Math.sqrt(target / ratio);
                if (w <= width && h <= height) {
                    cw = w;
                    ch = h;
                    break;
                }
            }
            double x0 = random.nextDouble() * (width - cw);
            double y0 = random.nextDouble() * (height - ch);
            float[] copy = new float[plane];
            for (int c = 0; c < channels; c++) {
                int base = offset + c * plane;
                System.arraycopy(f, base, copy, 0, plane);
                for (int y = 0; y < height; y++) {
                    double sy = Math.clamp(y0 + (y + 0.5) * ch / height - 0.5, 0, height - 1);
                    int ya = (int) sy;
                    int yb = Math.min(ya + 1, height - 1);
                    double fy = sy - ya;
                    for (int x = 0; x < width; x++) {
                        double sx = Math.clamp(x0 + (x + 0.5) * cw / width - 0.5, 0, width - 1);
                        int xa = (int) sx;
                        int xb = Math.min(xa + 1, width - 1);
                        double fx = sx - xa;
                        double top = copy[ya * width + xa] * (1 - fx) + copy[ya * width + xb] * fx;
                        double bottom = copy[yb * width + xa] * (1 - fx) + copy[yb * width + xb] * fx;
                        f[base + y * width + x] = (float) (top * (1 - fy) + bottom * fy);
                    }
                }
            }
        };
    }

    /**
     * Random brightness, contrast and (for three channels) saturation changes, each a factor
     * drawn from {@code [1 - amount, 1 + amount]}, in random order; values are taken to be in
     * {@code [0, 1]} and clipped to it.
     *
     * @param channels the channels, stored first
     * @param height the image height
     * @param width the image width
     * @param brightness the brightness range
     * @param contrast the contrast range
     * @param saturation the saturation range, used with three channels
     * @return the augmentation
     */
    static Augmentation colorJitter(int channels, int height, int width, float brightness, float contrast,
            float saturation) {
        checkImage(channels, height, width);
        int plane = height * width;
        return (f, offset, random) -> {
            int[] order = {0, 1, 2};
            for (int i = 2; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int op : order) {
                float amount = op == 0 ? brightness : op == 1 ? contrast : saturation;
                if (amount <= 0 || op == 2 && channels != 3) {
                    continue;
                }
                float factor = (float) (1 - amount + random.nextDouble() * 2 * amount);
                if (op == 0) {
                    for (int i = 0; i < channels * plane; i++) {
                        f[offset + i] = Math.clamp(f[offset + i] * factor, 0f, 1f);
                    }
                } else {
                    float[] gray = new float[plane];
                    for (int i = 0; i < plane; i++) {
                        gray[i] = channels == 3 ? 0.299f * f[offset + i] + 0.587f * f[offset + plane + i]
                                + 0.114f * f[offset + 2 * plane + i] : f[offset + i];
                    }
                    float mean = 0;
                    for (float g : gray) {
                        mean += g / plane;
                    }
                    for (int c = 0; c < channels; c++) {
                        for (int i = 0; i < plane; i++) {
                            float toward = op == 1 ? mean : gray[i];
                            int k = offset + c * plane + i;
                            f[k] = Math.clamp(toward + (f[k] - toward) * factor, 0f, 1f);
                        }
                    }
                }
            }
        };
    }

    /**
     * Random erasing (Zhong et al., 2020): with probability {@code p}, a rectangle covering a
     * share of the area in {@code [minArea, maxArea]} is set to zero.
     *
     * @param channels the channels, stored first
     * @param height the image height
     * @param width the image width
     * @param p how often
     * @param minArea the smallest share, such as 0.02
     * @param maxArea the largest share, such as 0.33
     * @return the augmentation
     */
    static Augmentation randomErasing(int channels, int height, int width, double p, double minArea,
            double maxArea) {
        checkImage(channels, height, width);
        int plane = height * width;
        return (f, offset, random) -> {
            if (random.nextDouble() >= p) {
                return;
            }
            double target = plane * (minArea + random.nextDouble() * (maxArea - minArea));
            double ratio = Math.exp(Math.log(0.3) + random.nextDouble() * Math.log(1 / 0.09));
            int h = Math.clamp(Math.round(Math.sqrt(target * ratio)), 1, height);
            int w = Math.clamp(Math.round(Math.sqrt(target / ratio)), 1, width);
            int y0 = random.nextInt(height - h + 1);
            int x0 = random.nextInt(width - w + 1);
            for (int c = 0; c < channels; c++) {
                for (int y = y0; y < y0 + h; y++) {
                    java.util.Arrays.fill(f, offset + c * plane + y * width + x0, offset + c * plane + y * width + x0 + w,
                            0f);
                }
            }
        };
    }

    /**
     * RandAugment (Cubuk et al., 2020): {@code n} operations picked at random, each at the
     * same {@code magnitude} from 0 to 10: identity, shifts along x and y, rotation,
     * brightness, contrast, solarize, posterize and horizontal flip. Values are taken to be
     * in {@code [0, 1]}.
     *
     * @param channels the channels, stored first
     * @param height the image height
     * @param width the image width
     * @param n operations per sample, such as 2
     * @param magnitude their strength, 0 to 10, such as 9
     * @return the augmentation
     */
    static Augmentation randAugment(int channels, int height, int width, int n, double magnitude) {
        checkImage(channels, height, width);
        double m = Math.clamp(magnitude, 0, 10) / 10;
        int plane = height * width;
        Augmentation flip = horizontalFlip(channels, height, width);
        return (f, offset, random) -> {
            for (int k = 0; k < n; k++) {
                double sign = random.nextBoolean() ? 1 : -1;
                switch (random.nextInt(9)) {
                    case 0 -> {
                    }
                    case 1 -> warp(f, offset, channels, height, width, 0, sign * m * 0.3 * width, 0);
                    case 2 -> warp(f, offset, channels, height, width, 0, 0, sign * m * 0.3 * height);
                    case 3 -> warp(f, offset, channels, height, width, sign * m * Math.toRadians(30), 0, 0);
                    case 4, 5 -> {
                        float factor = (float) (1 + sign * m * 0.9);
                        float mean = 0;
                        for (int i = 0; i < channels * plane; i++) {
                            mean += f[offset + i] / (channels * plane);
                        }
                        float toward = random.nextBoolean() ? 0 : mean;
                        for (int i = 0; i < channels * plane; i++) {
                            f[offset + i] = Math.clamp(toward + (f[offset + i] - toward) * factor, 0f, 1f);
                        }
                    }
                    case 6 -> {
                        float threshold = (float) (1 - m);
                        for (int i = 0; i < channels * plane; i++) {
                            if (f[offset + i] >= threshold) {
                                f[offset + i] = 1 - f[offset + i];
                            }
                        }
                    }
                    case 7 -> {
                        int levels = 1 << Math.max(1, 8 - (int) Math.round(m * 4));
                        for (int i = 0; i < channels * plane; i++) {
                            f[offset + i] = (float) Math.floor(f[offset + i] * levels) / levels;
                        }
                    }
                    default -> flip.apply(f, offset, random);
                }
            }
        };
    }

    /** Rotates by {@code angle} about the center and shifts, nearest neighbour, zeros outside. */
    private static void warp(float[] f, int offset, int channels, int height, int width, double angle,
            double dx, double dy) {
        int plane = height * width;
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        double cy = (height - 1) / 2.0;
        double cx = (width - 1) / 2.0;
        float[] copy = new float[plane];
        for (int c = 0; c < channels; c++) {
            int base = offset + c * plane;
            System.arraycopy(f, base, copy, 0, plane);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double ux = x - cx - dx;
                    double uy = y - cy - dy;
                    long sx = Math.round(cos * ux + sin * uy + cx);
                    long sy = Math.round(-sin * ux + cos * uy + cy);
                    f[base + y * width + x] = sx >= 0 && sx < width && sy >= 0 && sy < height
                            ? copy[(int) sy * width + (int) sx] : 0f;
                }
            }
        }
    }
}
