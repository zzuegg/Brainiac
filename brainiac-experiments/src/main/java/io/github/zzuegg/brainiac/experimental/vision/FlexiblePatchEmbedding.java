package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * FlexiViT's patch embedding (Beyer et al., CVPR 2023): one set of patch-embedding weights,
 * stored at a base patch size, used at any patch size.
 *
 * <p>Images {@code [batch, channels, h, w]} become tokens {@code [batch, (h / p) (w / p), dim]}
 * for the current patch size {@code p}. The weights are resized on the fly by the
 * <em>pseudo-inverse resize</em>: with {@code B} the bilinear resize of a base patch to a
 * {@code p x p} patch, the new weights {@code w' = pinv(B^T) w} keep every token unchanged,
 * {@code <B x, w'> = <x, w>}, when the patch itself is resized ({@link Resize#PSEUDO_INVERSE});
 * {@link Resize#BILINEAR} resizes the weights like an image instead (the naive baseline whose
 * tokens change scale and collapse away from the training size). The learned position
 * embeddings live on the base grid and are resized bilinearly to the current grid. During
 * training a random patch size is drawn per step ({@link #samplePatch}).
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2212.08013">arXiv:2212.08013</a>.
 */
public final class FlexiblePatchEmbedding implements Layer {

    /** How the base weights are resized to another patch size. */
    public enum Resize {
        /** FlexiViT's pseudo-inverse resize, which preserves token values. */
        PSEUDO_INVERSE,
        /** Plain bilinear resizing of the weights. */
        BILINEAR
    }

    private final int channels;
    private final int dim;
    private final int basePatch;
    private final int baseGrid;
    private final int[] patchSizes;
    private final Resize resize;
    private final Var weight;
    private final Var bias;
    private final Var positions;
    private final Map<Integer, Tensor> resizers = new HashMap<>();
    private int patch;

    /**
     * Creates the embedding.
     *
     * @param channels image channels
     * @param dim the token width
     * @param basePatch the patch size the weights are stored at
     * @param baseGrid the side of the position-embedding grid
     * @param patchSizes the sizes {@link #samplePatch} draws from
     * @param resize the weight resize
     * @param random the source of initial weights
     */
    public FlexiblePatchEmbedding(int channels, int dim, int basePatch, int baseGrid, int[] patchSizes, Resize resize,
            RandomGenerator random) {
        this.channels = channels;
        this.dim = dim;
        this.basePatch = basePatch;
        this.baseGrid = baseGrid;
        this.patchSizes = patchSizes.clone();
        this.resize = resize;
        int in = channels * basePatch * basePatch;
        weight = Var.param(Tensor.normal(Shape.of(in, dim), 0, (float) Math.sqrt(1.0 / in), random));
        bias = Var.param(Tensor.zeros(Shape.of(dim)));
        positions = Var.param(Tensor.normal(Shape.of(1, dim, baseGrid, baseGrid), 0, 0.02f, random));
        patch = basePatch;
    }

    /** @return the current patch size */
    public int patch() {
        return patch;
    }

    /**
     * Sets the patch size used by {@link #forward}.
     *
     * @param p the patch size
     */
    public void patch(int p) {
        this.patch = p;
    }

    /**
     * Draws a training patch size uniformly from the configured sizes and makes it current.
     *
     * @param random the source
     * @return the size
     */
    public int samplePatch(RandomGenerator random) {
        patch = patchSizes[random.nextInt(patchSizes.length)];
        return patch;
    }

    /** The 1-D bilinear resize matrix {@code [to, from]}. */
    static Tensor resize1d(int from, int to) {
        float[] eye = new float[from * from];
        for (int i = 0; i < from; i++) {
            eye[i * from + i] = 1;
        }
        Tensor r = Var.noGrad(() -> Functional.resize(Var.of(Tensor.of(Shape.of(from, 1, from), eye)), new int[] {to},
                Functional.Interpolation.LINEAR, false)).value();
        return r.reshape(Shape.of(from, to)).transpose();
    }

    /**
     * The matrix {@code M [p^2, base^2]} that maps base weights (per channel) to weights of
     * patch size {@code p}.
     *
     * @param p the patch size
     * @return the resize matrix
     */
    Tensor weightResizer(int p) {
        return resizers.computeIfAbsent(p, size -> {
            Tensor r = resize1d(basePatch, size);             // [p, base]
            float[] rv = r.toArray();
            int n = size * size;
            int m = basePatch * basePatch;
            float[] b = new float[n * m];                     // B = kron(R, R) [p^2, base^2]
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < basePatch; k++) {
                        for (int l = 0; l < basePatch; l++) {
                            b[(i * size + j) * m + k * basePatch + l] = rv[i * basePatch + k] * rv[j * basePatch + l];
                        }
                    }
                }
            }
            Tensor bm = Tensor.of(Shape.of(n, m), b);
            if (resize == Resize.BILINEAR) {
                // resize the weights as an image, rescaled so a constant patch keeps its response
                return bm.mul((float) m / n);
            }
            if (n >= m) {
                // pinv(B^T) = B (B^T B)^-1
                Tensor gram = bm.transpose().matmul(bm).add(Linalg.identity(m, bm.dtype(), bm.device()).mul(1e-6f));
                return bm.matmul(Linalg.inverse(gram));
            }
            // pinv(B^T) = (B B^T)^-1 B
            Tensor gram = bm.matmul(bm.transpose()).add(Linalg.identity(n, bm.dtype(), bm.device()).mul(1e-6f));
            return Linalg.inverse(gram).matmul(bm);
        });
    }

    @Override
    public Var forward(Var images) {
        int b = images.shape().dim(0);
        int h = images.shape().dim(2);
        int w = images.shape().dim(3);
        Var tokens = Functional.patchify(images, patch);                 // [b, n, c p p]
        Var wt = weight;
        if (patch != basePatch) {
            Tensor m = weightResizer(patch);
            Var perChannel = weight.reshape(channels, basePatch * basePatch, dim);
            wt = Var.of(m.reshape(Shape.of(1, patch * patch, basePatch * basePatch))
                    .expand(Shape.of(channels, patch * patch, basePatch * basePatch))).matmul(perChannel)
                    .reshape(channels * patch * patch, dim);
        }
        int gh = h / patch;
        int gw = w / patch;
        Var pos = positions;
        if (gh != baseGrid || gw != baseGrid) {
            pos = Functional.resize(positions, new int[] {gh, gw}, Functional.Interpolation.LINEAR, false);
        }
        pos = pos.reshape(dim, gh * gw).transpose().reshape(1, gh * gw, dim);
        return tokens.matmul(wt).add(bias).add(pos).reshape(b, gh * gw, dim);
    }

    /**
     * Multiply-adds of the embedding for an image.
     *
     * @param h image height
     * @param w image width
     * @return multiply-adds
     */
    public long macs(int h, int w) {
        return (long) h * w * channels * dim;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of((input.dim(1) / patch) * (input.dim(2) / patch), dim);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias, positions);
    }

    @Override
    public String toString() {
        return "FlexiblePatchEmbedding(base " + basePatch + ", now " + patch + ", " + resize + ")";
    }
}
