package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.experimental.restore.Denoisers;
import io.github.zzuegg.brainiac.experimental.restore.ImageDenoiser;
import io.github.zzuegg.brainiac.experimental.restore.PlugAndPlay;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Blind deconvolution of {@code y = k * x + n} with an unknown kernel (circular blur).
 *
 * <p>Naive joint MAP prefers the blurry image and a delta kernel; the methods here instead
 * estimate the kernel with a sparse gradient prior that favours sharp edges, alternating
 * latent-image and kernel updates on an image pyramid (coarse to fine), then hand the kernel to a
 * non-blind solver ({@link PlugAndPlay#hqs} with a TV denoiser prior, {@link #deconvolve}).
 * <ul>
 *   <li>{@link Prior#L0_GRADIENT} (Xu et al. 2013; Pan et al., CVPR 2014): the latent update
 *       minimises {@code |k * x - y|^2 + lambda |grad x|_0} by half-quadratic splitting with an
 *       increasing penalty, hard-thresholding the auxiliary gradients and solving for {@code x}
 *       in the Fourier domain.</li>
 *   <li>{@link Prior#NORMALISED_SPARSITY} (Krishnan, Tay &amp; Fergus, CVPR 2011): the
 *       scale-invariant {@code |grad x|_1 / |grad x|_2}, minimised the same way with soft
 *       thresholding at {@code lambda / |grad x|_2}.</li>
 * </ul>
 * The kernel update is regularised least squares in the gradient domain
 * {@code min |grad x * k - grad y|^2 + gamma |k|^2}, solved in the Fourier domain, then cropped to
 * the kernel support, re-centred on its centroid, small entries zeroed and the rest projected to
 * be non-negative and sum to one (Levin et al. 2011; Cho &amp; Lee 2009).
 */
public final class BlindDeconvolution {

    /** The sparse gradient priors. */
    public enum Prior {
        /** The l0 norm of the gradient. */
        L0_GRADIENT,
        /** The l1/l2 ratio of the gradient. */
        NORMALISED_SPARSITY
    }

    private BlindDeconvolution() {
    }

    private static double[][] fft(double[] x, int side) {
        double[] re = x.clone();
        double[] im = new double[x.length];
        ImageMath.fft2(re, im, side, side, false);
        return new double[][] {re, im};
    }

    private static double[] ifftReal(double[] re, double[] im, int side) {
        double[] r = re.clone();
        double[] i = im.clone();
        ImageMath.fft2(r, i, side, side, true);
        return r;
    }

    private static double[][] gradientTransfers(int side) {
        double[] dx = {0, 0, 0, 1, -1, 0, 0, 0, 0};
        double[] dy = {0, 1, 0, 0, -1, 0, 0, 0, 0};
        double[][] tx = ImageMath.transfer(dx, 3, side);
        double[][] ty = ImageMath.transfer(dy, 3, side);
        return new double[][] {tx[0], tx[1], ty[0], ty[1]};
    }

    /**
     * Estimates the kernel.
     *
     * @param prior the gradient prior
     * @param y the blurred image
     * @param side its side (a power of two)
     * @param kernelSize the kernel side (odd)
     * @param levels pyramid levels
     * @param iterations alternations per level
     * @param lambda the prior weight at the finest level
     * @return the kernel, {@code kernelSize x kernelSize}
     */
    public static double[] estimateKernel(Prior prior, double[] y, int side, int kernelSize, int levels, int iterations,
            double lambda) {
        int lv = levels;
        while (lv > 1 && ((kernelSize >> (lv - 1)) < 3 || (side >> (lv - 1)) < 16)) {
            lv--;
        }
        double[] kernel = null;
        int ks = 0;
        for (int l = lv - 1; l >= 0; l--) {
            int s = side >> l;
            int k = Math.max(3, (kernelSize >> l) | 1);
            double[] yl = l == 0 ? y : downsample(y, side, 1 << l);
            if (kernel == null) {
                kernel = new double[k * k];
                kernel[(k / 2) * k + k / 2] = 1;
            } else {
                kernel = resizeKernel(kernel, ks, k);
            }
            ks = k;
            double lam = lambda * Math.pow(1.5, l);
            for (int it = 0; it < iterations; it++) {
                double[] x = latent(prior, yl, s, kernel, k, lam);
                kernel = kernelStep(x, yl, s, k, 1e-3 * s * s / 256.0);
                lam = Math.max(lam * 0.9, lambda * 0.5);
            }
        }
        return kernel;
    }

    /**
     * The latent-image update with a sparse gradient prior.
     *
     * @param prior the prior
     * @param y the blurred image
     * @param side its side
     * @param kernel the current kernel
     * @param k its side
     * @param lambda the prior weight
     * @return the sharp estimate
     */
    public static double[] latent(Prior prior, double[] y, int side, double[] kernel, int k, double lambda) {
        int n = side * side;
        double[][] kf = ImageMath.transfer(kernel, k, side);
        double[][] g = gradientTransfers(side);
        double[][] yf = fft(y, side);
        double[] x = y.clone();
        double beta = 2 * lambda;
        double l2 = 1;
        while (beta < 1e5) {
            double[] gx = circularGrad(x, side, true);
            double[] gy = circularGrad(x, side, false);
            if (prior == Prior.NORMALISED_SPARSITY) {
                l2 = Math.sqrt(ImageMath.dot(gx, gx) + ImageMath.dot(gy, gy)) + 1e-9;
            }
            for (int i = 0; i < n; i++) {
                if (prior == Prior.L0_GRADIENT) {
                    if (gx[i] * gx[i] + gy[i] * gy[i] < lambda / beta) {
                        gx[i] = 0;
                        gy[i] = 0;
                    }
                } else {
                    double t = lambda / (l2 * beta);
                    gx[i] = Math.signum(gx[i]) * Math.max(0, Math.abs(gx[i]) - t);
                    gy[i] = Math.signum(gy[i]) * Math.max(0, Math.abs(gy[i]) - t);
                }
            }
            double[][] gxf = fft(gx, side);
            double[][] gyf = fft(gy, side);
            double[] re = new double[n];
            double[] im = new double[n];
            for (int i = 0; i < n; i++) {
                double kr = kf[0][i];
                double ki = kf[1][i];
                double nr = kr * yf[0][i] + ki * yf[1][i];
                double ni = kr * yf[1][i] - ki * yf[0][i];
                nr += beta * (g[0][i] * gxf[0][i] + g[1][i] * gxf[1][i] + g[2][i] * gyf[0][i] + g[3][i] * gyf[1][i]);
                ni += beta * (g[0][i] * gxf[1][i] - g[1][i] * gxf[0][i] + g[2][i] * gyf[1][i] - g[3][i] * gyf[0][i]);
                double den = kr * kr + ki * ki + beta * (g[0][i] * g[0][i] + g[1][i] * g[1][i] + g[2][i] * g[2][i]
                        + g[3][i] * g[3][i]) + 1e-8;
                re[i] = nr / den;
                im[i] = ni / den;
            }
            x = ifftReal(re, im, side);
            beta *= 2;
        }
        return x;
    }

    private static double[] circularGrad(double[] x, int side, boolean horizontal) {
        double[] g = new double[x.length];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int j = horizontal ? r * side + (c + 1) % side : ((r + 1) % side) * side + c;
                g[r * side + c] = x[j] - x[r * side + c];
            }
        }
        return g;
    }

    /**
     * The kernel update in the gradient domain.
     *
     * @param x the latent estimate
     * @param y the blurred image
     * @param side the side
     * @param k the kernel side
     * @param gamma the Tikhonov weight
     * @return the projected kernel
     */
    public static double[] kernelStep(double[] x, double[] y, int side, int k, double gamma) {
        int n = side * side;
        double[] numR = new double[n];
        double[] numI = new double[n];
        double[] den = new double[n];
        for (boolean h : new boolean[] {true, false}) {
            double[][] xf = fft(circularGrad(x, side, h), side);
            double[][] yf = fft(circularGrad(y, side, h), side);
            for (int i = 0; i < n; i++) {
                numR[i] += xf[0][i] * yf[0][i] + xf[1][i] * yf[1][i];
                numI[i] += xf[0][i] * yf[1][i] - xf[1][i] * yf[0][i];
                den[i] += xf[0][i] * xf[0][i] + xf[1][i] * xf[1][i];
            }
        }
        for (int i = 0; i < n; i++) {
            numR[i] /= den[i] + gamma * n;
            numI[i] /= den[i] + gamma * n;
        }
        double[] full = ifftReal(numR, numI, side);
        double[] kernel = new double[k * k];
        int c = k / 2;
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                kernel[ky * k + kx] = full[Math.floorMod(ky - c, side) * side + Math.floorMod(kx - c, side)];
            }
        }
        return project(recentre(kernel, k), k);
    }

    private static double[] project(double[] kernel, int k) {
        double max = 0;
        for (double v : kernel) {
            max = Math.max(max, v);
        }
        double s = 0;
        double[] out = new double[kernel.length];
        for (int i = 0; i < kernel.length; i++) {
            out[i] = kernel[i] >= 0.05 * max ? kernel[i] : 0;
            s += out[i];
        }
        if (s <= 0) {
            out = new double[kernel.length];
            out[(k / 2) * k + k / 2] = 1;
            return out;
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= s;
        }
        return out;
    }

    private static double[] recentre(double[] kernel, int k) {
        double s = 0;
        double cy = 0;
        double cx = 0;
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                double v = Math.max(0, kernel[ky * k + kx]);
                s += v;
                cy += v * ky;
                cx += v * kx;
            }
        }
        if (s <= 0) {
            return kernel;
        }
        int dy = (int) Math.round(cy / s) - k / 2;
        int dx = (int) Math.round(cx / s) - k / 2;
        double[] out = new double[kernel.length];
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                int sy = ky + dy;
                int sx = kx + dx;
                if (sy >= 0 && sx >= 0 && sy < k && sx < k) {
                    out[ky * k + kx] = kernel[sy * k + sx];
                }
            }
        }
        return out;
    }

    private static double[] downsample(double[] x, int side, int f) {
        int s = side / f;
        double[] out = new double[s * s];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                out[(r / f) * s + c / f] += x[r * side + c] / (f * f);
            }
        }
        return out;
    }

    private static double[] resizeKernel(double[] kernel, int from, int to) {
        double[] out = new double[to * to];
        double scale = (from - 1) / (double) Math.max(1, to - 1);
        double s = 0;
        for (int y = 0; y < to; y++) {
            for (int x = 0; x < to; x++) {
                double sy = y * scale;
                double sx = x * scale;
                int y0 = (int) Math.floor(sy);
                int x0 = (int) Math.floor(sx);
                double fy = sy - y0;
                double fx = sx - x0;
                double v = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int yy = Math.min(from - 1, y0 + dy);
                        int xx = Math.min(from - 1, x0 + dx);
                        v += (dy == 0 ? 1 - fy : fy) * (dx == 0 ? 1 - fx : fx) * kernel[yy * from + xx];
                    }
                }
                out[y * to + x] = Math.max(0, v);
                s += out[y * to + x];
            }
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= s;
        }
        return out;
    }

    /**
     * A TV-denoiser prior for the plug-and-play solvers ({@link PrimalDual} TV-L2 with weight equal
     * to the noise level).
     *
     * @return the denoiser
     */
    public static ImageDenoiser tvDenoiser() {
        return Denoisers.of((im, side, sigma) -> ImageMath.toFloat(PrimalDual.denoise(PrimalDual.Model.TV_L2,
                ImageMath.toDouble(im), side, sigma, 40)));
    }

    /**
     * Non-blind deconvolution with a given kernel by {@link PlugAndPlay#hqs} and the TV denoiser.
     *
     * @param y the blurred image
     * @param side the side
     * @param kernel the kernel
     * @param k its side
     * @param noise the noise level
     * @return the estimate
     */
    public static double[] deconvolve(double[] y, int side, double[] kernel, int k, double noise) {
        Tensor transfer = PlugAndPlay.transfer(ImageMath.toFloat(kernel), k, side, side);
        Tensor yt = Tensor.of(Shape.of(1, 1, side, side), ImageMath.toFloat(y));
        Tensor out = PlugAndPlay.hqs(yt, PlugAndPlay.stackTransfers(java.util.List.of(transfer)), (float) noise,
                tvDenoiser(), 8, 0.23f);
        return ImageMath.toDouble(out.toArray());
    }

    /**
     * Kernel similarity: the maximum normalised cross-correlation over shifts.
     *
     * @param a one kernel
     * @param b another of the same size
     * @param k the side
     * @return a value in [0, 1] for non-negative kernels
     */
    public static double kernelSimilarity(double[] a, double[] b, int k) {
        double na = Math.sqrt(ImageMath.dot(a, a));
        double nb = Math.sqrt(ImageMath.dot(b, b));
        double best = 0;
        for (int dy = -k / 2; dy <= k / 2; dy++) {
            for (int dx = -k / 2; dx <= k / 2; dx++) {
                double s = 0;
                for (int y = 0; y < k; y++) {
                    for (int x = 0; x < k; x++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy >= 0 && xx >= 0 && yy < k && xx < k) {
                            s += a[y * k + x] * b[yy * k + xx];
                        }
                    }
                }
                best = Math.max(best, s / (na * nb));
            }
        }
        return best;
    }
}
