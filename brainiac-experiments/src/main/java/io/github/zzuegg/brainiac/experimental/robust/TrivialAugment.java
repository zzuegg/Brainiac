package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.data.Augmentation;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.DoubleFunction;

/**
 * TrivialAugment (Müller and Hutter, ICCV 2021, <a href="https://arxiv.org/abs/2103.10158">arXiv:2103.10158</a>):
 * per image, one operation drawn uniformly from a list, at a strength drawn uniformly from
 * {@code [0, 1]}; no search and no tuned magnitude. It is a core {@link Augmentation}, so it
 * drops in wherever RandAugment does.
 *
 * <p>The operations are strength-parametrised core augmentations; {@link #geometric} builds
 * the default list from {@link Augmentation}'s shift, crop-and-zoom and erasing, leaving out
 * photometric operations (contrast, posterize) so that test-time corruptions of that kind
 * stay held out of training.
 */
public final class TrivialAugment implements Augmentation {

    private final List<DoubleFunction<Augmentation>> operations;

    /**
     * Creates the augmentation.
     *
     * @param operations operations by strength in {@code [0, 1]}
     */
    public TrivialAugment(List<DoubleFunction<Augmentation>> operations) {
        this.operations = List.copyOf(operations);
        if (this.operations.isEmpty()) {
            throw new IllegalArgumentException("no operations");
        }
    }

    /**
     * The default operations for {@code [channels, height, width]} images in {@code [0, 1]}:
     * identity, shift by up to three pixels, random-resized crop down to half the area, and
     * random erasing of up to a fifth of the image.
     *
     * @param channels the channels
     * @param height the height
     * @param width the width
     * @return strength-parametrised operations
     */
    public static List<DoubleFunction<Augmentation>> geometric(int channels, int height, int width) {
        return List.of(
                s -> (f, o, r) -> {
                },
                s -> Augmentation.cropAndFlip(channels, height, width, (int) Math.round(3 * s)),
                s -> Augmentation.randomResizedCrop(channels, height, width, 1 - 0.5 * s, 1),
                s -> Augmentation.randomErasing(channels, height, width, 1, 0.02, Math.max(0.03, 0.2 * s)));
    }

    /**
     * Draws one random operation at one random strength.
     *
     * @param operations the operations
     * @param random the random source
     * @return the operation
     */
    static Augmentation draw(List<DoubleFunction<Augmentation>> operations, SplittableRandom random) {
        return Objects.requireNonNull(operations.get(random.nextInt(operations.size())).apply(random.nextDouble()));
    }

    @Override
    public void apply(float[] features, int offset, SplittableRandom random) {
        draw(operations, random).apply(features, offset, random);
    }
}
