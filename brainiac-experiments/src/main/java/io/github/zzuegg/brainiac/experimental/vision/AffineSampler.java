package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Differentiable grid generation and bilinear sampling, the two halves of a spatial
 * transformer (Jaderberg et al., 2015): {@link #affineGrid} maps every output pixel through a
 * {@code 2x3} affine matrix to a sampling location, and {@link #sample} reads the input there
 * by bilinear interpolation, with gradients to both the image and the locations. The same
 * sampler warps a frame by an optical flow ({@link #warp}, used by the flow blocks) and
 * shifts a feature map by a sub-pixel amount ({@link #shift}, used by stereo).
 *
 * <p>Normalised coordinates run from {@code -1} (centre of the first pixel) to {@code 1}
 * (centre of the last), as {@code align_corners = true} in other libraries. Samples outside
 * the image read zero.
 *
 * <p>Paper: Jaderberg et al., "Spatial Transformer Networks" (2015),
 * <a href="https://arxiv.org/abs/1506.02025">arXiv:1506.02025</a>.
 */
public final class AffineSampler {

    private AffineSampler() {
    }

    /**
     * The sampling grid of affine transforms: output pixel {@code (x, y)} in normalised
     * coordinates samples the input at {@code theta [x, y, 1]^T}.
     *
     * @param theta {@code [n, 2, 3]} transforms; the identity is {@code [[1, 0, 0], [0, 1, 0]]}
     * @param height the output height
     * @param width the output width
     * @return {@code [n, height, width, 2]} normalised {@code (x, y)} locations
     */
    public static Var affineGrid(Var theta, int height, int width) {
        int n = theta.shape().dim(0);
        float[] base = new float[height * width * 3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                base[i] = width == 1 ? 0 : -1 + 2f * x / (width - 1);
                base[i + 1] = height == 1 ? 0 : -1 + 2f * y / (height - 1);
                base[i + 2] = 1;
            }
        }
        Var b = Var.of(Tensor.of(Shape.of(1, height * width, 3), base));
        // [n, hw, 3] x [n, 3, 2] = [n, hw, 2]
        return b.expand(Shape.of(n, height * width, 3)).matmul(theta.permute(0, 2, 1)).reshape(n, height, width, 2);
    }

    /**
     * The identity transform, repeated.
     *
     * @param n how many
     * @return {@code [n, 2, 3]}
     */
    public static Tensor identity(int n) {
        float[] t = new float[6 * n];
        for (int i = 0; i < n; i++) {
            t[6 * i] = 1;
            t[6 * i + 4] = 1;
        }
        return Tensor.of(Shape.of(n, 2, 3), t);
    }

    /**
     * Samples images bilinearly at normalised locations.
     *
     * @param images {@code [n, c, h, w]}
     * @param grid {@code [n, H, W, 2]} normalised {@code (x, y)} locations
     * @return {@code [n, c, H, W]}
     */
    public static Var sample(Var images, Var grid) {
        int h = images.shape().dim(2);
        int w = images.shape().dim(3);
        return samplePixels(images, grid, (w - 1) / 2f, (h - 1) / 2f, (w - 1) / 2f, (h - 1) / 2f);
    }

    /**
     * Warps images by a dense displacement: output pixel {@code (x, y)} reads the input at
     * {@code (x + u, y + v)}. Backward-warping the second frame by the flow of the first
     * should reproduce the first.
     *
     * @param images {@code [n, c, h, w]}
     * @param flow {@code [n, 2, h, w]}, the displacement {@code (u, v)} in pixels
     * @return {@code [n, c, h, w]}
     */
    public static Var warp(Var images, Var flow) {
        int n = images.shape().dim(0);
        int h = images.shape().dim(2);
        int w = images.shape().dim(3);
        float[] base = new float[h * w * 2];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                base[(y * w + x) * 2] = x;
                base[(y * w + x) * 2 + 1] = y;
            }
        }
        Var grid = flow.permute(0, 2, 3, 1).add(Tensor.of(Shape.of(1, h, w, 2), base));
        if (grid.shape().dim(0) != n) {
            throw new IllegalArgumentException("flow " + flow.shape() + " does not fit images " + images.shape());
        }
        return samplePixels(images, grid, 1, 1, 0, 0);
    }

    /**
     * Shifts images horizontally: output column {@code x} reads input column
     * {@code x - dx}, so {@code dx > 0} moves content right. Fractional shifts interpolate.
     *
     * @param images {@code [n, c, h, w]}
     * @param dx the shift in pixels
     * @return {@code [n, c, h, w]}
     */
    public static Var shift(Var images, float dx) {
        int n = images.shape().dim(0);
        int h = images.shape().dim(2);
        int w = images.shape().dim(3);
        float[] g = new float[n * h * w * 2];
        for (int i = 0; i < n; i++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int k = ((i * h + y) * w + x) * 2;
                    g[k] = x - dx;
                    g[k + 1] = y;
                }
            }
        }
        return samplePixels(images, Var.of(Tensor.of(Shape.of(n, h, w, 2), g)), 1, 1, 0, 0);
    }

    /**
     * Bilinear sampling at {@code pixel = scale * grid + offset}, with gradients to the
     * images and the grid.
     */
    private static Var samplePixels(Var images, Var grid, float sx, float sy, float ox, float oy) {
        int n = images.shape().dim(0);
        int c = images.shape().dim(1);
        int h = images.shape().dim(2);
        int w = images.shape().dim(3);
        int gh = grid.shape().dim(1);
        int gw = grid.shape().dim(2);
        if (grid.shape().dim(0) != n || grid.shape().dim(3) != 2) {
            throw new IllegalArgumentException("grid " + grid.shape() + " does not fit images " + images.shape());
        }
        return Var.custom(in -> {
            float[] img = in[0].toArray();
            float[] gr = in[1].toArray();
            float[] out = new float[n * c * gh * gw];
            for (int b = 0; b < n; b++) {
                for (int p = 0; p < gh * gw; p++) {
                    float x = gr[(b * gh * gw + p) * 2] * sx + ox;
                    float y = gr[(b * gh * gw + p) * 2 + 1] * sy + oy;
                    int x0 = (int) Math.floor(x);
                    int y0 = (int) Math.floor(y);
                    float fx = x - x0;
                    float fy = y - y0;
                    for (int ch = 0; ch < c; ch++) {
                        int plane = (b * c + ch) * h * w;
                        float v00 = at(img, plane, w, h, x0, y0);
                        float v10 = at(img, plane, w, h, x0 + 1, y0);
                        float v01 = at(img, plane, w, h, x0, y0 + 1);
                        float v11 = at(img, plane, w, h, x0 + 1, y0 + 1);
                        out[(b * c + ch) * gh * gw + p] = (1 - fy) * ((1 - fx) * v00 + fx * v10)
                                + fy * ((1 - fx) * v01 + fx * v11);
                    }
                }
            }
            return Tensor.of(Shape.of(n, c, gh, gw), out);
        }, (in, output, grad) -> {
            float[] img = in[0].toArray();
            float[] gr = in[1].toArray();
            float[] g = grad.toArray();
            float[] dImg = new float[img.length];
            float[] dGrid = new float[gr.length];
            for (int b = 0; b < n; b++) {
                for (int p = 0; p < gh * gw; p++) {
                    float x = gr[(b * gh * gw + p) * 2] * sx + ox;
                    float y = gr[(b * gh * gw + p) * 2 + 1] * sy + oy;
                    int x0 = (int) Math.floor(x);
                    int y0 = (int) Math.floor(y);
                    float fx = x - x0;
                    float fy = y - y0;
                    float gx = 0;
                    float gy = 0;
                    for (int ch = 0; ch < c; ch++) {
                        int plane = (b * c + ch) * h * w;
                        float go = g[(b * c + ch) * gh * gw + p];
                        if (go == 0) {
                            continue;
                        }
                        add(dImg, plane, w, h, x0, y0, go * (1 - fx) * (1 - fy));
                        add(dImg, plane, w, h, x0 + 1, y0, go * fx * (1 - fy));
                        add(dImg, plane, w, h, x0, y0 + 1, go * (1 - fx) * fy);
                        add(dImg, plane, w, h, x0 + 1, y0 + 1, go * fx * fy);
                        float v00 = at(img, plane, w, h, x0, y0);
                        float v10 = at(img, plane, w, h, x0 + 1, y0);
                        float v01 = at(img, plane, w, h, x0, y0 + 1);
                        float v11 = at(img, plane, w, h, x0 + 1, y0 + 1);
                        gx += go * ((1 - fy) * (v10 - v00) + fy * (v11 - v01));
                        gy += go * ((1 - fx) * (v01 - v00) + fx * (v11 - v10));
                    }
                    dGrid[(b * gh * gw + p) * 2] = gx * sx;
                    dGrid[(b * gh * gw + p) * 2 + 1] = gy * sy;
                }
            }
            return new Tensor[] {Tensor.of(in[0].shape(), dImg), Tensor.of(in[1].shape(), dGrid)};
        }, images, grid);
    }

    private static float at(float[] img, int plane, int w, int h, int x, int y) {
        return x < 0 || y < 0 || x >= w || y >= h ? 0 : img[plane + y * w + x];
    }

    private static void add(float[] img, int plane, int w, int h, int x, int y, float v) {
        if (x >= 0 && y >= 0 && x < w && y < h) {
            img[plane + y * w + x] += v;
        }
    }
}
