package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * The multi-scale residual tokenizer of VAR, "visual autoregressive modeling" by next-scale
 * prediction (Tian et al., <a href="https://arxiv.org/abs/2404.02905">2404.02905</a>). An
 * image's feature map (here its {@code patch x patch} pixel tiles on a {@code g x g} grid)
 * is described by token maps of growing size {@code s_1 < s_2 < ... < s_K = g}: map
 * {@code k} quantizes, with its own {@link VectorQuantizer} codebook, the area-downsampled
 * <em>residual</em> that the coarser maps left, and its codes, upsampled back to
 * {@code g x g}, are subtracted before the next scale. The image is the sum of all
 * upsampled maps, so a model that writes the coarse maps first sketches the whole image
 * and then refines it.
 *
 * <p>{@link #contextInputs} gives what a next-scale predictor sees for every token: the
 * reconstruction from all coarser scales, downsampled to the token's own scale (zeros for
 * the first scale). Those inputs depend only on coarser tokens, so all tokens of one scale
 * can be predicted in parallel: generation takes {@code K} network calls.
 */
public final class MultiScaleVqTokenizer {
    private final int side;
    private final int patch;
    private final int grid;
    private final int dim;
    private final int[] scales;
    private final List<VectorQuantizer> codebooks = new ArrayList<>();

    private MultiScaleVqTokenizer(int side, int patch, int[] scales, int codes, long seed) {
        if (side % patch != 0) {
            throw new IllegalArgumentException("patch " + patch + " does not divide side " + side);
        }
        this.side = side;
        this.patch = patch;
        this.grid = side / patch;
        this.dim = patch * patch;
        this.scales = scales.clone();
        if (scales[scales.length - 1] != grid) {
            throw new IllegalArgumentException("the last scale must be the grid size " + grid);
        }
        for (int k = 0; k < scales.length; k++) {
            if (grid % scales[k] != 0 || (k > 0 && scales[k] <= scales[k - 1])) {
                throw new IllegalArgumentException("scales must grow and divide the grid size " + grid);
            }
            codebooks.add(VectorQuantizer.of(codes, dim, 0.25f, seed + k));
        }
    }

    /**
     * Creates an unfitted tokenizer.
     *
     * @param side the image side length
     * @param patch the tile side length
     * @param scales the token map sizes, increasing, each dividing {@code side / patch}, the
     *     last equal to it
     * @param codes the codebook size of every scale
     * @param seed the seed for k-means
     * @return the tokenizer
     */
    public static MultiScaleVqTokenizer of(int side, int patch, int[] scales, int codes, long seed) {
        return new MultiScaleVqTokenizer(side, patch, scales, codes, seed);
    }

    /** @return the token map sizes */
    public int[] scales() {
        return scales.clone();
    }

    /** @return the vocabulary size of every scale */
    public int vocabulary() {
        return codebooks.get(0).codebook().shape().dim(0);
    }

    /** @return the length of a feature vector, {@code patch * patch} */
    public int dim() {
        return dim;
    }

    /** @return the total number of tokens of all scales */
    public int sequenceLength() {
        int n = 0;
        for (int s : scales) {
            n += s * s;
        }
        return n;
    }

    /**
     * Where a scale's tokens start in the flattened sequence.
     *
     * @param scale the scale index
     * @return the offset
     */
    public int offset(int scale) {
        int n = 0;
        for (int k = 0; k < scale; k++) {
            n += scales[k] * scales[k];
        }
        return n;
    }

    private float[] features(float[] image) {
        float[] f = new float[grid * grid * dim];
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                for (int y = 0; y < patch; y++) {
                    for (int x = 0; x < patch; x++) {
                        f[(gy * grid + gx) * dim + y * patch + x] = image[(gy * patch + y) * side + gx * patch + x];
                    }
                }
            }
        }
        return f;
    }

    private float[] image(float[] f) {
        float[] img = new float[side * side];
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                for (int y = 0; y < patch; y++) {
                    for (int x = 0; x < patch; x++) {
                        img[(gy * patch + y) * side + gx * patch + x] = f[(gy * grid + gx) * dim + y * patch + x];
                    }
                }
            }
        }
        return img;
    }

    /** Area-averages a {@code grid x grid} map down to {@code s x s}. */
    private float[] down(float[] map, int s) {
        int block = grid / s;
        float[] out = new float[s * s * dim];
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                int o = ((gy / block) * s + gx / block) * dim;
                for (int d = 0; d < dim; d++) {
                    out[o + d] += map[(gy * grid + gx) * dim + d] / (block * block);
                }
            }
        }
        return out;
    }

    /** Repeats an {@code s x s} map up to {@code grid x grid}. */
    private float[] up(float[] map, int s) {
        int block = grid / s;
        float[] out = new float[grid * grid * dim];
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                System.arraycopy(map, ((gy / block) * s + gx / block) * dim, out, (gy * grid + gx) * dim, dim);
            }
        }
        return out;
    }

    private float[] lookup(int scale, int[] codes) {
        float[] c = codebooks.get(scale).codebook().value().toArray();
        float[] out = new float[codes.length * dim];
        for (int i = 0; i < codes.length; i++) {
            System.arraycopy(c, codes[i] * dim, out, i * dim, dim);
        }
        return out;
    }

    private static void subtract(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] -= b[i];
        }
    }

    /**
     * Fits the codebooks scale by scale with k-means on the residuals the coarser scales leave.
     *
     * @param images training images
     * @param iterations Lloyd iterations per scale
     * @return the mean squared reconstruction error per pixel on those images
     */
    public double fit(List<float[]> images, int iterations) {
        List<float[]> residuals = new ArrayList<>();
        for (float[] img : images) {
            residuals.add(features(img));
        }
        for (int k = 0; k < scales.length; k++) {
            int s = scales[k];
            float[] rows = new float[images.size() * s * s * dim];
            for (int i = 0; i < residuals.size(); i++) {
                System.arraycopy(down(residuals.get(i), s), 0, rows, i * s * s * dim, s * s * dim);
            }
            PatchVqTokenizer.kMeans(codebooks.get(k), rows, dim, iterations);
            for (float[] r : residuals) {
                float[] d = down(r, s);
                int[] codes = codebooks.get(k).nearest(Tensor.of(Shape.of(s * s, dim), d));
                subtract(r, up(lookup(k, codes), s));
            }
        }
        double err = 0;
        for (float[] img : images) {
            float[] back = decode(encode(img));
            for (int i = 0; i < img.length; i++) {
                err += (back[i] - img[i]) * (back[i] - img[i]);
            }
        }
        return err / (images.size() * (double) side * side);
    }

    /**
     * Encodes an image into one token map per scale.
     *
     * @param image {@code side * side} pixels
     * @return {@code tokens[scale]}, each in raster order
     */
    public int[][] encode(float[] image) {
        float[] r = features(image);
        int[][] out = new int[scales.length][];
        for (int k = 0; k < scales.length; k++) {
            int s = scales[k];
            out[k] = codebooks.get(k).nearest(Tensor.of(Shape.of(s * s, dim), down(r, s)));
            subtract(r, up(lookup(k, out[k]), s));
        }
        return out;
    }

    /**
     * Decodes token maps, using only the first {@code tokens.length} scales.
     *
     * @param tokens {@code tokens[scale]}
     * @return the image
     */
    public float[] decode(int[][] tokens) {
        float[] acc = new float[grid * grid * dim];
        for (int k = 0; k < tokens.length; k++) {
            float[] u = up(lookup(k, tokens[k]), scales[k]);
            for (int i = 0; i < acc.length; i++) {
                acc[i] += u[i];
            }
        }
        return image(acc);
    }

    /**
     * Flattens token maps coarse to fine.
     *
     * @param tokens {@code tokens[scale]}
     * @return the token sequence
     */
    public int[] flatten(int[][] tokens) {
        int[] out = new int[sequenceLength()];
        for (int k = 0; k < tokens.length; k++) {
            System.arraycopy(tokens[k], 0, out, offset(k), tokens[k].length);
        }
        return out;
    }

    /**
     * The next-scale predictor's input for every token: the reconstruction from all coarser
     * scales, area-downsampled to the token's scale. Scales from {@code known} on are left
     * as zeros.
     *
     * @param tokens {@code tokens[scale]} for at least the first {@code known - 1} scales
     * @param known how many scales of inputs to fill
     * @return {@code [sequenceLength][dim]}, flattened row-major
     */
    public float[] contextInputs(int[][] tokens, int known) {
        float[] out = new float[sequenceLength() * dim];
        float[] acc = new float[grid * grid * dim];
        for (int k = 0; k < Math.min(known, scales.length); k++) {
            if (k > 0) {
                float[] u = up(lookup(k - 1, tokens[k - 1]), scales[k - 1]);
                for (int i = 0; i < acc.length; i++) {
                    acc[i] += u[i];
                }
                System.arraycopy(down(acc, scales[k]), 0, out, offset(k) * dim, scales[k] * scales[k] * dim);
            }
        }
        return out;
    }
}
