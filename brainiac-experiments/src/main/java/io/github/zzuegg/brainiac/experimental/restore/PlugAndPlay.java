package io.github.zzuegg.brainiac.experimental.restore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Non-blind deblurring of circularly blurred images {@code y = k * x + n} with learned priors.
 *
 * <ul>
 *   <li>The data step {@link #dataStep}: the closed-form minimiser of
 *       {@code |k * x - y|^2 + alpha |x - z|^2}, {@code X = (conj(K) Y + alpha Z) / (|K|^2 + alpha)}
 *       in the 2-D Fourier domain (core {@link Var#fft}), differentiable in {@code z} and
 *       {@code alpha}; with {@code z = 0} it is Wiener deconvolution ({@link #wiener}).</li>
 *   <li>Plug-and-play half-quadratic splitting ({@link #hqs}, the DPIR recipe of Zhang et al.,
 *       TPAMI 2021) and ADMM ({@link #admm}, Venkatakrishnan et al., 2013): alternate the
 *       data step with any {@link ImageDenoiser} standing in for the prior's proximal operator,
 *       with the denoiser's noise level decreasing log-uniformly from 49/255 to the noise level
 *       and {@code alpha_k = lambda sigma^2 / sigma_k^2}.</li>
 *   <li>Deep unrolling ({@link Unrolled}): a fixed number of HQS iterations trained end to end,
 *       each with its own small residual CNN denoiser and learned {@code alpha_k}.</li>
 *   <li>Classical baselines: {@link #richardsonLucy} and total-variation-regularised
 *       deconvolution by gradient descent ({@link #totalVariation}).</li>
 * </ul>
 * Images are square with a power-of-two side. Papers:
 * <a href="https://doi.org/10.1109/GlobalSIP.2013.6737048">doi:10.1109/GlobalSIP.2013.6737048</a>,
 * <a href="https://arxiv.org/abs/2008.13751">arXiv:2008.13751</a>.
 */
public final class PlugAndPlay {

    private PlugAndPlay() {
    }

    // ---------------------------------------------------------------- FFT helpers

    /**
     * The 2-D DFT of complex images {@code [b, h, w, 2]}.
     *
     * @param c the complex images
     * @param inverse whether to invert (with the {@code 1/(h w)} factor)
     * @return the transform
     */
    public static Var fft2(Var c, boolean inverse) {
        Var rows = c.fft(inverse);
        return rows.permute(0, 2, 1, 3).fft(inverse).permute(0, 2, 1, 3);
    }

    private static Var complex(Var real) {
        Shape s = real.shape();
        return Var.concat(3, real.reshape(s.dim(0), s.dim(1), s.dim(2), 1),
                Var.of(Tensor.zeros(Shape.of(s.dim(0), s.dim(1), s.dim(2), 1))));
    }

    /**
     * The transfer function of a centred blur kernel on an {@code h x w} grid (circular
     * convolution as {@link DegradedImages#blur}).
     *
     * @param kernel the kernel, {@code k x k}
     * @param k its side
     * @param h the grid height
     * @param w the grid width
     * @return {@code [h, w, 2]}
     */
    public static Tensor transfer(float[] kernel, int k, int h, int w) {
        float[] shifted = new float[h * w];
        int c = k / 2;
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                shifted[Math.floorMod(ky - c, h) * w + Math.floorMod(kx - c, w)] += kernel[ky * k + kx];
            }
        }
        return Var.noGrad(() -> fft2(complex(Var.of(Tensor.of(Shape.of(1, h, w), shifted))), false))
                .value().reshape(Shape.of(h, w, 2));
    }

    /**
     * Stacks transfer functions into a batch.
     *
     * @param transfers per image {@code [h, w, 2]}
     * @return {@code [b, h, w, 2]}
     */
    public static Tensor stackTransfers(List<Tensor> transfers) {
        Tensor[] parts = new Tensor[transfers.size()];
        for (int i = 0; i < parts.length; i++) {
            Shape s = transfers.get(i).shape();
            parts[i] = transfers.get(i).reshape(Shape.of(1, s.dim(0), s.dim(1), 2));
        }
        return Tensor.concat(0, parts);
    }

    /**
     * The data step {@code argmin_x |k * x - y|^2 + alpha |x - z|^2}.
     *
     * @param y the blurred images {@code [b, 1, h, w]}
     * @param transfer the kernels' transfer functions {@code [b, h, w, 2]}
     * @param z the prior estimate {@code [b, 1, h, w]}, or {@code null} for zero
     * @param alpha the weight: a scalar Var or {@code [b, 1, 1, 1]}
     * @return the minimiser {@code [b, 1, h, w]}
     */
    public static Var dataStep(Var y, Tensor transfer, Var z, Var alpha) {
        Shape s = y.shape();
        int b = s.dim(0);
        int h = s.dim(2);
        int w = s.dim(3);
        Var yf = fft2(complex(y.reshape(b, h, w)), false);
        Tensor kr = transfer.slice(3, 0, 1);
        Tensor ki = transfer.slice(3, 1, 2);
        Var yr = yf.slice(3, 0, 1);
        Var yi = yf.slice(3, 1, 2);
        Var re = yr.mul(kr).add(yi.mul(ki));
        Var im = yi.mul(kr).sub(yr.mul(ki));
        Var a = alpha.reshape(alpha.shape().size() == 1L ? new int[] {1, 1, 1, 1} : new int[] {b, 1, 1, 1});
        if (z != null) {
            Var zf = fft2(complex(z.reshape(b, h, w)), false);
            re = re.add(zf.slice(3, 0, 1).mul(a));
            im = im.add(zf.slice(3, 1, 2).mul(a));
        }
        Var den = Var.of(kr.mul(kr).add(ki.mul(ki))).add(a);
        Var x = fft2(Var.concat(3, re.div(den), im.div(den)), true);
        return x.slice(3, 0, 1).reshape(b, 1, h, w);
    }

    private static Var scalar(float v) {
        return Var.of(Tensor.scalar(v));
    }

    /**
     * Wiener deconvolution with a constant noise-to-signal ratio.
     *
     * @param y the blurred images {@code [b, 1, h, w]}
     * @param transfer {@code [b, h, w, 2]}
     * @param nsr the noise-to-signal power ratio
     * @return the estimates
     */
    public static Tensor wiener(Tensor y, Tensor transfer, float nsr) {
        return Var.noGrad(() -> dataStep(Var.of(y), transfer, null, scalar(nsr))).value();
    }

    private static float[] sigmas(int iterations, float sigma1, float sigmaEnd) {
        float[] s = new float[iterations];
        for (int i = 0; i < iterations; i++) {
            double t = iterations == 1 ? 1 : i / (double) (iterations - 1);
            s[i] = (float) Math.exp(Math.log(sigma1) * (1 - t) + Math.log(sigmaEnd) * t);
        }
        return s;
    }

    /**
     * Plug-and-play half-quadratic splitting (DPIR): {@code x = data(y, z, alpha_k)},
     * {@code z = D(x, sigma_k)}.
     *
     * @param y the blurred images {@code [b, 1, h, w]}
     * @param transfer {@code [b, h, w, 2]}
     * @param noise the noise standard deviation of {@code y}
     * @param denoiser the prior
     * @param iterations the number of iterations (DPIR uses 8)
     * @param lambda the trade-off (DPIR uses about 0.23)
     * @return the estimates
     */
    public static Tensor hqs(Tensor y, Tensor transfer, float noise, ImageDenoiser denoiser, int iterations,
            float lambda) {
        return Var.noGrad(() -> {
            float[] s = sigmas(iterations, 49f / 255, Math.max(noise, 0.005f));
            Var z = Var.of(y);
            for (int k = 0; k < iterations; k++) {
                float alpha = lambda * Math.max(noise, 0.005f) * Math.max(noise, 0.005f) / (s[k] * s[k]);
                Var x = dataStep(Var.of(y), transfer, z, scalar(alpha));
                z = denoiser.denoise(x, s[k]);
            }
            return z;
        }).value();
    }

    /**
     * Plug-and-play ADMM: {@code x = data(y, z - u)}, {@code z = D(x + u, sigma_k)},
     * {@code u += x - z}, on the same schedule as {@link #hqs}.
     *
     * @param y the blurred images
     * @param transfer {@code [b, h, w, 2]}
     * @param noise the noise standard deviation
     * @param denoiser the prior
     * @param iterations the number of iterations
     * @param lambda the trade-off
     * @return the estimates
     */
    public static Tensor admm(Tensor y, Tensor transfer, float noise, ImageDenoiser denoiser, int iterations,
            float lambda) {
        return Var.noGrad(() -> {
            float[] s = sigmas(iterations, 49f / 255, Math.max(noise, 0.005f));
            Var z = Var.of(y);
            Var u = Var.of(Tensor.zeros(y.shape()));
            for (int k = 0; k < iterations; k++) {
                float alpha = lambda * Math.max(noise, 0.005f) * Math.max(noise, 0.005f) / (s[k] * s[k]);
                Var x = dataStep(Var.of(y), transfer, z.sub(u), scalar(alpha));
                z = denoiser.denoise(x.add(u), s[k]);
                u = u.add(x).sub(z);
            }
            return z;
        }).value();
    }

    /**
     * Richardson-Lucy deconvolution (Poisson maximum likelihood), {@code x <- x * k^T(y / (k *
     * x))}, started from the blurred image.
     *
     * @param y the blurred image
     * @param side its side
     * @param kernel the kernel
     * @param k its side
     * @param iterations the iterations
     * @return the estimate
     */
    public static float[] richardsonLucy(float[] y, int side, float[] kernel, int k, int iterations) {
        float[] flipped = flip(kernel, k);
        float[] x = new float[y.length];
        float[] obs = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            obs[i] = Math.max(1e-3f, y[i]);
            x[i] = obs[i];
        }
        for (int it = 0; it < iterations; it++) {
            float[] est = DegradedImages.blur(x, side, side, kernel, k);
            float[] ratio = new float[y.length];
            for (int i = 0; i < y.length; i++) {
                ratio[i] = obs[i] / Math.max(1e-4f, est[i]);
            }
            float[] corr = DegradedImages.blur(ratio, side, side, flipped, k);
            for (int i = 0; i < y.length; i++) {
                x[i] *= corr[i];
            }
        }
        return x;
    }

    /**
     * Total-variation-regularised deconvolution: gradient descent on
     * {@code 1/2 |k * x - y|^2 + lambda sum sqrt(|grad x|^2 + eps^2)}.
     *
     * @param y the blurred image
     * @param side its side
     * @param kernel the kernel
     * @param k its side
     * @param lambda the TV weight
     * @param steps gradient steps
     * @param rate the step size
     * @return the estimate
     */
    public static float[] totalVariation(float[] y, int side, float[] kernel, int k, float lambda, int steps,
            float rate) {
        float[] flipped = flip(kernel, k);
        float[] x = y.clone();
        float eps = 0.01f;
        for (int it = 0; it < steps; it++) {
            float[] res = DegradedImages.blur(x, side, side, kernel, k);
            for (int i = 0; i < res.length; i++) {
                res[i] -= y[i];
            }
            float[] grad = DegradedImages.blur(res, side, side, flipped, k);
            float[] px = new float[x.length];
            float[] py = new float[x.length];
            for (int r = 0; r < side; r++) {
                for (int c = 0; c < side; c++) {
                    int i = r * side + c;
                    float gx = x[r * side + Math.floorMod(c + 1, side)] - x[i];
                    float gy = x[Math.floorMod(r + 1, side) * side + c] - x[i];
                    float norm = (float) Math.sqrt(gx * gx + gy * gy + eps * eps);
                    px[i] = gx / norm;
                    py[i] = gy / norm;
                }
            }
            for (int r = 0; r < side; r++) {
                for (int c = 0; c < side; c++) {
                    int i = r * side + c;
                    float div = px[i] - px[r * side + Math.floorMod(c - 1, side)] + py[i]
                            - py[Math.floorMod(r - 1, side) * side + c];
                    x[i] -= rate * (grad[i] - lambda * div);
                }
            }
        }
        return x;
    }

    private static float[] flip(float[] kernel, int k) {
        float[] f = new float[kernel.length];
        for (int i = 0; i < kernel.length; i++) {
            f[kernel.length - 1 - i] = kernel[i];
        }
        return f;
    }

    /**
     * Deep unrolled HQS: {@code iterations} data steps, each followed by its own residual CNN
     * (3x3, 3x3, 3x3 convolutions) with a learned {@code alpha_k = softplus(a_k)}; trained end to
     * end on blurred-sharp pairs with their kernels.
     */
    public static final class Unrolled {
        private final List<RestoreOps.Conv[]> denoisers = new ArrayList<>();
        private final Var alphas;

        /**
         * Creates the network.
         *
         * @param iterations the unrolled iterations
         * @param channels the width of each denoiser
         * @param random the source of initial weights
         */
        public Unrolled(int iterations, int channels, RandomGenerator random) {
            float[] a = new float[iterations];
            for (int i = 0; i < iterations; i++) {
                // softplus^-1 of alphas decreasing from 0.1 to 0.005
                double alpha = 0.1 * Math.pow(0.05, i / (double) Math.max(1, iterations - 1));
                a[i] = (float) Math.log(Math.expm1(alpha));
                denoisers.add(new RestoreOps.Conv[] {new RestoreOps.Conv(1, channels, 3, random),
                    new RestoreOps.Conv(channels, channels, 3, random),
                    new RestoreOps.Conv(channels, 1, 3, 1e-2f, random)});
            }
            alphas = Var.param(Tensor.of(Shape.of(iterations), a));
        }

        /**
         * Restores images.
         *
         * @param y the blurred images {@code [b, 1, h, w]}
         * @param transfer {@code [b, h, w, 2]}
         * @return the estimates
         */
        public Var forward(Var y, Tensor transfer) {
            Var z = y;
            for (int k = 0; k < denoisers.size(); k++) {
                Var alpha = alphas.slice(0, k, k + 1).softplus();
                Var x = dataStep(y, transfer, z, alpha);
                RestoreOps.Conv[] d = denoisers.get(k);
                Var h = d[0].forward(x).relu();
                h = d[1].forward(h).relu();
                z = x.sub(d[2].forward(h));
            }
            return z;
        }

        /** @return the trainable parameters */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            p.add(alphas);
            for (RestoreOps.Conv[] d : denoisers) {
                for (RestoreOps.Conv c : d) {
                    p.addAll(c.parameters());
                }
            }
            return p;
        }
    }
}
