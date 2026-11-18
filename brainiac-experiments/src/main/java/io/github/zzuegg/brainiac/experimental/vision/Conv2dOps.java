package io.github.zzuegg.brainiac.experimental.vision;

/**
 * Public 2-D convolution helpers on plain row-major arrays: zero padding, im2col unrolling,
 * the direct (six-loop) convolution, the im2col-plus-matrix-product convolution and the
 * per-channel 2-D filtering that classical vision code (Sobel, Gaussian smoothing, Harris
 * structure tensors) needs. Wave 10 found that every package kept a package-private copy of
 * these; new code uses this class, and the existing classes keep their local helpers so their
 * measured numbers do not move.
 *
 * <p>Every convolution here is a cross-correlation, as in deep-learning libraries: images are
 * {@code [channels, h, w]}, filters {@code [out, in, k, k]}, outputs
 * {@code [out, (h + 2 pad - k) / stride + 1, ...]}.
 */
public final class Conv2dOps {

    private Conv2dOps() {
    }

    /**
     * Zero-pads every channel.
     *
     * @param x {@code [c, h, w]}
     * @param c channels
     * @param h rows
     * @param w columns
     * @param pad pixels added on every side
     * @return {@code [c, h + 2 pad, w + 2 pad]}
     */
    public static float[] pad(float[] x, int c, int h, int w, int pad) {
        int ph = h + 2 * pad;
        int pw = w + 2 * pad;
        float[] out = new float[c * ph * pw];
        for (int ch = 0; ch < c; ch++) {
            for (int y = 0; y < h; y++) {
                System.arraycopy(x, (ch * h + y) * w, out, (ch * ph + y + pad) * pw + pad, w);
            }
        }
        return out;
    }

    /**
     * @param size input side
     * @param k kernel side
     * @param stride stride
     * @param pad padding
     * @return output side
     */
    public static int outSize(int size, int k, int stride, int pad) {
        return (size + 2 * pad - k) / stride + 1;
    }

    /**
     * Unrolls sliding windows into the columns of a matrix.
     *
     * @param x {@code [c, h, w]}
     * @param c channels
     * @param h rows
     * @param w columns
     * @param k kernel side
     * @param stride stride
     * @param pad zero padding
     * @return {@code [c k k, oh ow]}, row-major
     */
    public static float[] im2col(float[] x, int c, int h, int w, int k, int stride, int pad) {
        int oh = outSize(h, k, stride, pad);
        int ow = outSize(w, k, stride, pad);
        int cols = oh * ow;
        float[] out = new float[c * k * k * cols];
        for (int ch = 0; ch < c; ch++) {
            for (int ky = 0; ky < k; ky++) {
                for (int kx = 0; kx < k; kx++) {
                    int row = (ch * k + ky) * k + kx;
                    for (int oy = 0; oy < oh; oy++) {
                        int y = oy * stride + ky - pad;
                        if (y < 0 || y >= h) {
                            continue;
                        }
                        for (int ox = 0; ox < ow; ox++) {
                            int xx = ox * stride + kx - pad;
                            if (xx >= 0 && xx < w) {
                                out[row * cols + oy * ow + ox] = x[(ch * h + y) * w + xx];
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    /**
     * The direct convolution.
     *
     * @param x {@code [in, h, w]}
     * @param in input channels
     * @param h rows
     * @param w columns
     * @param weight {@code [out, in, k, k]}
     * @param out output channels
     * @param k kernel side
     * @param stride stride
     * @param pad zero padding
     * @return {@code [out, oh, ow]}
     */
    public static float[] direct(float[] x, int in, int h, int w, float[] weight, int out, int k, int stride,
            int pad) {
        int oh = outSize(h, k, stride, pad);
        int ow = outSize(w, k, stride, pad);
        float[] y = new float[out * oh * ow];
        for (int o = 0; o < out; o++) {
            for (int ch = 0; ch < in; ch++) {
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {
                        float wv = weight[((o * in + ch) * k + ky) * k + kx];
                        for (int oy = 0; oy < oh; oy++) {
                            int iy = oy * stride + ky - pad;
                            if (iy < 0 || iy >= h) {
                                continue;
                            }
                            int src = (ch * h + iy) * w;
                            int dst = (o * oh + oy) * ow;
                            for (int ox = 0; ox < ow; ox++) {
                                int ix = ox * stride + kx - pad;
                                if (ix >= 0 && ix < w) {
                                    y[dst + ox] += wv * x[src + ix];
                                }
                            }
                        }
                    }
                }
            }
        }
        return y;
    }

    /**
     * The im2col convolution: unroll, then one matrix product.
     *
     * @param x {@code [in, h, w]}
     * @param in input channels
     * @param h rows
     * @param w columns
     * @param weight {@code [out, in, k, k]}
     * @param out output channels
     * @param k kernel side
     * @param stride stride
     * @param pad zero padding
     * @return {@code [out, oh, ow]}
     */
    public static float[] im2colConv(float[] x, int in, int h, int w, float[] weight, int out, int k, int stride,
            int pad) {
        float[] cols = im2col(x, in, h, w, k, stride, pad);
        int n = outSize(h, k, stride, pad) * outSize(w, k, stride, pad);
        int rows = in * k * k;
        float[] y = new float[out * n];
        for (int o = 0; o < out; o++) {
            for (int r = 0; r < rows; r++) {
                float wv = weight[o * rows + r];
                if (wv == 0) {
                    continue;
                }
                int src = r * n;
                int dst = o * n;
                for (int j = 0; j < n; j++) {
                    y[dst + j] += wv * cols[src + j];
                }
            }
        }
        return y;
    }

    /**
     * Filters one single-channel image with a small kernel, same size, edges replicated.
     *
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param kernel row-major {@code kh x kw}, odd sides
     * @param kh kernel rows
     * @param kw kernel columns
     * @return the filtered image
     */
    public static float[] filter(float[] image, int h, int w, float[] kernel, int kh, int kw) {
        float[] out = new float[h * w];
        int ry = kh / 2;
        int rx = kw / 2;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double s = 0;
                for (int dy = 0; dy < kh; dy++) {
                    int yy = Math.min(h - 1, Math.max(0, y + dy - ry));
                    for (int dx = 0; dx < kw; dx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + dx - rx));
                        s += kernel[dy * kw + dx] * image[yy * w + xx];
                    }
                }
                out[y * w + x] = (float) s;
            }
        }
        return out;
    }

    /**
     * A normalised 1-D Gaussian kernel.
     *
     * @param sigma the standard deviation in pixels
     * @return {@code 2 ceil(3 sigma) + 1} taps summing to one
     */
    public static float[] gaussian(double sigma) {
        int r = Math.max(1, (int) Math.ceil(3 * sigma));
        float[] g = new float[2 * r + 1];
        double sum = 0;
        for (int i = -r; i <= r; i++) {
            double v = Math.exp(-i * i / (2 * sigma * sigma));
            g[i + r] = (float) v;
            sum += v;
        }
        for (int i = 0; i < g.length; i++) {
            g[i] /= (float) sum;
        }
        return g;
    }

    /**
     * Separable Gaussian smoothing with replicated edges.
     *
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param sigma the standard deviation in pixels
     * @return the smoothed image
     */
    public static float[] gaussianBlur(float[] image, int h, int w, double sigma) {
        float[] g = gaussian(sigma);
        return filter(filter(image, h, w, g, 1, g.length), h, w, g, g.length, 1);
    }

    /** The horizontal Sobel kernel (derivative along columns). */
    public static final float[] SOBEL_X = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    /** The vertical Sobel kernel (derivative along rows). */
    public static final float[] SOBEL_Y = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    /** The 4-neighbour Laplacian kernel. */
    public static final float[] LAPLACIAN = {0, 1, 0, 1, -4, 1, 0, 1, 0};

    /**
     * Sobel gradients.
     *
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @return {@code {gx, gy}}
     */
    public static float[][] sobel(float[] image, int h, int w) {
        return new float[][] {filter(image, h, w, SOBEL_X, 3, 3), filter(image, h, w, SOBEL_Y, 3, 3)};
    }

    /**
     * Bilinear sampling with edge clamping.
     *
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param x column coordinate (pixel centres at integers)
     * @param y row coordinate
     * @param outside the value returned outside the image
     * @return the sample
     */
    public static float bilinear(float[] image, int h, int w, double x, double y, float outside) {
        if (x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5) {
            return outside;
        }
        double cx = Math.min(w - 1, Math.max(0, x));
        double cy = Math.min(h - 1, Math.max(0, y));
        int x0 = (int) Math.floor(cx);
        int y0 = (int) Math.floor(cy);
        int x1 = Math.min(w - 1, x0 + 1);
        int y1 = Math.min(h - 1, y0 + 1);
        double fx = cx - x0;
        double fy = cy - y0;
        double top = image[y0 * w + x0] * (1 - fx) + image[y0 * w + x1] * fx;
        double bottom = image[y1 * w + x0] * (1 - fx) + image[y1 * w + x1] * fx;
        return (float) (top * (1 - fy) + bottom * fy);
    }
}
