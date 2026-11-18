package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialAttacks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Backdoor triggers for single-channel images stored as flat rows of {@code side x side}
 * pixels with values in about {@code [0, 1]}:
 *
 * <ul>
 *   <li>{@link Kind#PATCH}: BadNets stamps a small checkerboard in the bottom-right corner
 *       (Gu, Dolan-Gavitt and Garg 2017, <a href="https://arxiv.org/abs/1708.06733">arXiv:1708.06733</a>).</li>
 *   <li>{@link Kind#BLENDED}: alpha-blends a fixed random full-image pattern (Chen et al.
 *       2017, <a href="https://arxiv.org/abs/1712.05526">arXiv:1712.05526</a>).</li>
 *   <li>{@link Kind#CLEAN_LABEL}: keeps the correct labels of target-class images, first
 *       perturbs them adversarially with PGD on a surrogate so their natural features stop
 *       being predictive, then stamps the patch, so a model must use the trigger to fit them
 *       (Turner, Tsipras and Madry 2019, <a href="https://arxiv.org/abs/1912.02771">arXiv:1912.02771</a>).</li>
 * </ul>
 */
public final class Backdoors {

    /** A trigger type. */
    public enum Kind {
        /** a corner patch, with relabelling */
        PATCH,
        /** a blended pattern, with relabelling */
        BLENDED,
        /** adversarially perturbed target images with the patch, labels kept */
        CLEAN_LABEL
    }

    /** The blend weight of {@link Kind#BLENDED}. */
    public static final float BLEND_ALPHA = 0.25f;

    private Backdoors() {
    }

    /**
     * A fixed random pattern for the blended trigger.
     *
     * @param pixels the image size
     * @param seed the seed
     * @return values uniform in {@code [0, 1]}
     */
    public static float[] pattern(int pixels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] p = new float[pixels];
        for (int i = 0; i < pixels; i++) {
            p[i] = r.nextFloat();
        }
        return p;
    }

    /**
     * Stamps the 3x3 checkerboard patch in the bottom-right corner.
     *
     * @param image a flat image, not changed
     * @param side the image side
     * @return the triggered copy
     */
    public static float[] patch(float[] image, int side) {
        float[] out = image.clone();
        for (int y = side - 3; y < side; y++) {
            for (int x = side - 3; x < side; x++) {
                out[y * side + x] = (x + y) % 2 == 0 ? 1.5f : 0f;
            }
        }
        return out;
    }

    /**
     * Blends a pattern into an image.
     *
     * @param image a flat image, not changed
     * @param pattern the pattern
     * @param alpha the pattern's weight
     * @return {@code (1 - α) x + α p}
     */
    public static float[] blend(float[] image, float[] pattern, float alpha) {
        float[] out = new float[image.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (1 - alpha) * image[i] + alpha * pattern[i];
        }
        return out;
    }

    /**
     * Applies the test-time trigger of a kind.
     *
     * @param kind the kind
     * @param image a flat image, not changed
     * @param side the image side
     * @param pattern the blended pattern (only for {@link Kind#BLENDED})
     * @return the triggered copy
     */
    public static float[] stamp(Kind kind, float[] image, int side, float[] pattern) {
        Objects.requireNonNull(kind);
        return kind == Kind.BLENDED ? blend(image, pattern, BLEND_ALPHA) : patch(image, side);
    }

    /**
     * The clean-label perturbation: untargeted {@code ℓ∞} PGD on a surrogate, pushing the
     * images away from their (target) class.
     *
     * @param surrogate a model trained on clean data
     * @param images the target-class images
     * @param target their (correct) class
     * @param epsilon the budget, such as 0.3
     * @param steps PGD steps
     * @param seed the seed
     * @return the perturbed images
     */
    public static float[][] cleanLabelPerturb(UnaryOperator<Var> surrogate, float[][] images, int target,
            float epsilon, int steps, long seed) {
        if (images.length == 0) {
            return images;
        }
        Tensor x = Tensor.of(images);
        float[] ids = new float[images.length];
        java.util.Arrays.fill(ids, target);
        Tensor adv = AdversarialAttacks.pgd(surrogate, x, Tensor.of(Shape.of(ids.length), ids),
                AdversarialAttacks.Norm.LINF, epsilon, 2.5f * epsilon / steps, steps, 1, new SplittableRandom(seed));
        float[] v = adv.toArray();
        int d = images[0].length;
        float[][] out = new float[images.length][d];
        for (int i = 0; i < images.length; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * The attack success rate: the share of non-target test images that the model sends to
     * the target class once triggered.
     *
     * @param logits the model
     * @param x test images
     * @param y their labels
     * @param kind the trigger
     * @param side the image side
     * @param pattern the blended pattern
     * @param target the target class
     * @return the success rate
     */
    public static double attackSuccess(UnaryOperator<Var> logits, float[][] x, int[] y, Kind kind, int side,
            float[] pattern, int target) {
        int count = 0;
        for (int yi : y) {
            count += yi == target ? 0 : 1;
        }
        float[][] t = new float[count][];
        int c = 0;
        for (int i = 0; i < x.length; i++) {
            if (y[i] != target) {
                t[c++] = stamp(kind, x[i], side, pattern);
            }
        }
        int[] pred = logits.apply(Var.of(Tensor.of(t))).value().argmax(1).toIntArray();
        int hit = 0;
        for (int p : pred) {
            hit += p == target ? 1 : 0;
        }
        return hit / (double) Math.max(1, count);
    }
}
