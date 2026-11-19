package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Small two-channel images of one object with ground-truth concepts, for concept-based and
 * case-based explanations: the object's shape (square outline, plus, or diagonal), its colour
 * (which channel it is drawn in), its size (3 or 5 pixels), its position (top or bottom half)
 * and its texture (solid or striped, every other row dimmed), drawn in the same style as
 * {@link ShapeImages}. The task label is a known function of the concepts; an optional 2x2
 * background patch in the top-right corner can be correlated with a binary label at a chosen
 * rate, the spurious shortcut of explanation-guided training. Per-pixel masks of every concept
 * support network dissection and pointing scores.
 */
public final class ConceptShapes {

    /** The number of binary concepts in {@link Batch#concepts()}: plus, colour 1, large, top, striped. */
    public static final int CONCEPTS = 5;

    /** The names of the binary concepts. */
    public static final String[] CONCEPT_NAMES = {"plus", "colour1", "large", "top", "striped"};

    /** The names of the per-pixel masks of {@link Batch#masks()}. */
    public static final String[] MASK_NAMES = {"square", "plus", "diagonal", "channel0", "channel1", "large",
        "small", "striped", "solid", "top-half", "bottom-half", "patch"};

    /** The index of the background patch in the masks. */
    public static final int PATCH_MASK = 11;

    /** How the label is defined. */
    public enum Task {
        /** Six classes: shape (three kinds) times texture. */
        SIX_CLASS,
        /** Binary: {@code (plus and colour1) or (large and not top)}, squares and pluses only. */
        BOOLEAN,
        /** Binary: {@code (plus and colour1) or (large and striped)}, squares and pluses only. */
        BOOLEAN_TEXTURE
    }

    /**
     * A batch of images.
     *
     * @param images {@code [n, 2 * side * side]}, channel-major
     * @param labels the task labels
     * @param concepts {@code [n][CONCEPTS]} binary concepts
     * @param masks {@code [n][MASK_NAMES.length][side * side]} per-pixel concept masks
     * @param patch whether each image has the background patch
     */
    public record Batch(Tensor images, int[] labels, int[][] concepts, boolean[][][] masks, boolean[] patch) {

        /** @return the number of images */
        public int size() {
            return labels.length;
        }

        /** @return the concepts as a {@code [n][CONCEPTS]} float matrix */
        public float[][] conceptMatrix() {
            float[][] c = new float[concepts.length][];
            for (int i = 0; i < c.length; i++) {
                c[i] = new float[concepts[i].length];
                for (int k = 0; k < c[i].length; k++) {
                    c[i][k] = concepts[i][k];
                }
            }
            return c;
        }
    }

    private final int side;
    private final float noise;

    /**
     * Creates the generator.
     *
     * @param side the image side, at least 12
     * @param noise the pixel noise, such as 0.05
     */
    public ConceptShapes(int side, float noise) {
        if (side < 12) {
            throw new IllegalArgumentException("side must be at least 12, got " + side);
        }
        this.side = side;
        this.noise = noise;
    }

    /** @return the image side */
    public int side() {
        return side;
    }

    /** @return the number of input features, {@code 2 * side * side} */
    public int features() {
        return 2 * side * side;
    }

    /**
     * The number of classes of a task.
     *
     * @param task the task
     * @return 6 or 2
     */
    public static int classes(Task task) {
        return task == Task.SIX_CLASS ? 6 : 2;
    }

    /**
     * The label of a concept assignment.
     *
     * @param task the task
     * @param shape 0 square, 1 plus, 2 diagonal
     * @param c the binary concepts
     * @return the label
     */
    public static int label(Task task, int shape, int[] c) {
        return switch (task) {
            case SIX_CLASS -> shape * 2 + c[4];
            case BOOLEAN -> (c[0] == 1 && c[1] == 1) || (c[2] == 1 && c[3] == 0) ? 1 : 0;
            case BOOLEAN_TEXTURE -> (c[0] == 1 && c[1] == 1) || (c[2] == 1 && c[4] == 1) ? 1 : 0;
        };
    }

    /**
     * Generates images.
     *
     * @param task the label
     * @param n the number of images
     * @param patchRate for a binary task, the probability that the patch is present when the
     *     label is 1 and absent when it is 0 (0.5 makes it uninformative); negative disables the
     *     patch
     * @param seed the seed
     * @return the batch
     */
    public Batch generate(Task task, int n, double patchRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int s = side;
        int area = s * s;
        float[] img = new float[n * 2 * area];
        int[] labels = new int[n];
        int[][] concepts = new int[n][CONCEPTS];
        boolean[][][] masks = new boolean[n][MASK_NAMES.length][area];
        boolean[] patch = new boolean[n];
        int shapes = task == Task.SIX_CLASS ? 3 : 2;
        for (int i = 0; i < n; i++) {
            int shape = r.nextInt(shapes);
            int[] c = concepts[i];
            c[0] = shape == 1 ? 1 : 0;
            c[1] = r.nextInt(2);
            c[2] = r.nextInt(2);
            c[3] = r.nextInt(2);
            c[4] = r.nextInt(2);
            labels[i] = label(task, shape, c);
            int size = c[2] == 1 ? 5 : 3;
            int half = s / 2;
            int y0 = c[3] == 1 ? r.nextInt(half - size + 1) : half + r.nextInt(s - half - size + 1);
            int x0 = r.nextInt(s - size - 2);
            float b = 0.8f + 0.4f * r.nextFloat();
            int o = (i * 2 + c[1]) * area;
            boolean[][] m = masks[i];
            for (int dy = 0; dy < size; dy++) {
                for (int dx = 0; dx < size; dx++) {
                    boolean on = switch (shape) {
                        case 0 -> dy == 0 || dx == 0 || dy == size - 1 || dx == size - 1;
                        case 1 -> dy == size / 2 || dx == size / 2;
                        default -> dy == dx;
                    };
                    if (!on) {
                        continue;
                    }
                    int p = (y0 + dy) * s + x0 + dx;
                    img[o + p] = c[4] == 1 && dy % 2 == 1 ? 0.35f * b : b;
                    m[shape][p] = true;
                    m[3 + c[1]][p] = true;
                    m[c[2] == 1 ? 5 : 6][p] = true;
                    m[c[4] == 1 ? 7 : 8][p] = true;
                }
            }
            for (int p = 0; p < area; p++) {
                m[p / s < half ? 9 : 10][p] = true;
            }
            if (patchRate >= 0 && classes(task) == 2) {
                patch[i] = r.nextDouble() < (labels[i] == 1 ? patchRate : 1 - patchRate);
                if (patch[i]) {
                    for (int p : new int[] {s - 2, s - 1, 2 * s - 2, 2 * s - 1}) {
                        img[i * 2 * area + p] = 1f;
                        img[(i * 2 + 1) * area + p] = 1f;
                        m[PATCH_MASK][p] = true;
                    }
                }
            }
            for (int p = 0; p < 2 * area; p++) {
                img[i * 2 * area + p] += (float) (noise * r.nextGaussian());
            }
        }
        return new Batch(Tensor.of(Shape.of(n, 2 * area), img), labels, concepts, masks, patch);
    }

    /**
     * A boolean pixel mask of the background patch, {@code [2 * side * side]} over both channels,
     * the annotation that explanation-guided training penalises.
     *
     * @return the mask
     */
    public float[] patchAnnotation() {
        int area = side * side;
        float[] a = new float[2 * area];
        for (int p : new int[] {side - 2, side - 1, 2 * side - 2, 2 * side - 1}) {
            a[p] = 1;
            a[area + p] = 1;
        }
        return a;
    }
}
