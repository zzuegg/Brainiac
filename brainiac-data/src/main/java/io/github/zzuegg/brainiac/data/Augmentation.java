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
}
