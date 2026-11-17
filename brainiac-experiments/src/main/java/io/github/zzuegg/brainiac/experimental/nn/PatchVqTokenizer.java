package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * An image tokenizer in the style of the VQ tokenizers behind token-based image generators
 * (VQ-VAE, <a href="https://arxiv.org/abs/1711.00937">1711.00937</a>; MaskGIT and Chameleon
 * use learned VQ-GAN encoders): a square grayscale image is cut into non-overlapping
 * {@code patch x patch} tiles, and every tile becomes the index of its nearest entry in a
 * {@link VectorQuantizer} codebook. The codebook is fitted by k-means (k-means++ seeding
 * from {@link VectorQuantizer#initialize}, then Lloyd iterations), which for small images
 * plays the role of the trained encoder. Decoding pastes the code vectors back.
 *
 * <p>The tokens of an image are returned in raster order, {@code (side / patch)^2} of them,
 * each in {@code 0 .. codes - 1}: ready for a transformer.
 */
public final class PatchVqTokenizer {
    private final int side;
    private final int patch;
    private final int grid;
    private final VectorQuantizer quantizer;

    private PatchVqTokenizer(int side, int patch, int codes, long seed) {
        if (side % patch != 0) {
            throw new IllegalArgumentException("patch " + patch + " does not divide side " + side);
        }
        this.side = side;
        this.patch = patch;
        this.grid = side / patch;
        this.quantizer = VectorQuantizer.of(codes, patch * patch, 0.25f, seed);
    }

    /**
     * Creates an unfitted tokenizer.
     *
     * @param side the image side length
     * @param patch the tile side length, dividing {@code side}
     * @param codes the codebook size (the vocabulary)
     * @param seed the seed for k-means
     * @return the tokenizer
     */
    public static PatchVqTokenizer of(int side, int patch, int codes, long seed) {
        return new PatchVqTokenizer(side, patch, codes, seed);
    }

    /** @return the number of tokens per image */
    public int tokens() {
        return grid * grid;
    }

    /** @return the number of tiles per image side */
    public int grid() {
        return grid;
    }

    /** @return the vocabulary size */
    public int vocabulary() {
        return quantizer.codebook().shape().dim(0);
    }

    /** @return the underlying quantizer */
    public VectorQuantizer quantizer() {
        return quantizer;
    }

    /**
     * Cuts an image into tile vectors.
     *
     * @param image {@code side * side} pixels, row-major
     * @return {@code [tokens][patch * patch]}
     */
    public float[][] tiles(float[] image) {
        float[][] out = new float[grid * grid][patch * patch];
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                for (int y = 0; y < patch; y++) {
                    for (int x = 0; x < patch; x++) {
                        out[gy * grid + gx][y * patch + x] = image[(gy * patch + y) * side + gx * patch + x];
                    }
                }
            }
        }
        return out;
    }

    /**
     * Fits the codebook to the tiles of some images.
     *
     * @param images training images
     * @param iterations Lloyd iterations after the k-means++ seeding
     * @return the mean squared reconstruction error per pixel on those images
     */
    public double fit(List<float[]> images, int iterations) {
        int dim = patch * patch;
        int n = images.size() * grid * grid;
        float[] all = new float[n * dim];
        int row = 0;
        for (float[] img : images) {
            for (float[] t : tiles(img)) {
                System.arraycopy(t, 0, all, row++ * dim, dim);
            }
        }
        kMeans(quantizer, all, dim, iterations);
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
     * Fits a quantizer's codebook to rows by k-means: k-means++ seeding, then Lloyd iterations.
     *
     * @param quantizer the quantizer
     * @param rows the data, row-major
     * @param dim the row length
     * @param iterations Lloyd iterations
     */
    static void kMeans(VectorQuantizer quantizer, float[] rows, int dim, int iterations) {
        int n = rows.length / dim;
        Tensor x = Tensor.of(Shape.of(n, dim), rows);
        quantizer.initialize(x);
        int k = quantizer.codebook().shape().dim(0);
        for (int it = 0; it < iterations; it++) {
            int[] codes = quantizer.nearest(x);
            float[] c = quantizer.codebook().value().toArray();
            double[] sum = new double[k * dim];
            int[] count = new int[k];
            for (int i = 0; i < n; i++) {
                count[codes[i]]++;
                for (int j = 0; j < dim; j++) {
                    sum[codes[i] * dim + j] += rows[i * dim + j];
                }
            }
            for (int q = 0; q < k; q++) {
                if (count[q] > 0) {
                    for (int j = 0; j < dim; j++) {
                        c[q * dim + j] = (float) (sum[q * dim + j] / count[q]);
                    }
                }
            }
            quantizer.codebook().assign(Tensor.of(Shape.of(k, dim), c));
        }
    }

    /**
     * Encodes an image.
     *
     * @param image {@code side * side} pixels, row-major
     * @return the tokens in raster order
     */
    public int[] encode(float[] image) {
        float[][] t = tiles(image);
        float[] flat = new float[t.length * patch * patch];
        for (int i = 0; i < t.length; i++) {
            System.arraycopy(t[i], 0, flat, i * patch * patch, patch * patch);
        }
        return quantizer.nearest(Tensor.of(Shape.of(t.length, patch * patch), flat));
    }

    /**
     * Decodes tokens.
     *
     * @param tokens tokens in raster order
     * @return the image, {@code side * side} pixels
     */
    public float[] decode(int[] tokens) {
        float[] c = quantizer.codebook().value().toArray();
        int dim = patch * patch;
        float[] out = new float[side * side];
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                int code = tokens[gy * grid + gx];
                for (int y = 0; y < patch; y++) {
                    for (int x = 0; x < patch; x++) {
                        out[(gy * patch + y) * side + gx * patch + x] = c[code * dim + y * patch + x];
                    }
                }
            }
        }
        return out;
    }
}
