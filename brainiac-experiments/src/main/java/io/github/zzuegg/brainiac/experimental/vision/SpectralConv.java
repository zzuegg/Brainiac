package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Convolution and pooling in the Fourier domain.
 *
 * <ul>
 *   <li>{@link #fftConv}: stride-1 cross-correlation as a pointwise product of spectra
 *       (Mathieu, Henaff and LeCun): every input channel and every filter is transformed
 *       once, products are summed over input channels, and one inverse transform per output
 *       channel gives the result. The cost is {@code O(N^2 log N)} per channel pair instead
 *       of {@code O(N^2 k^2)}, so it wins for large kernels.
 *   <li>{@link #conv}: the same as a differentiable {@link Var} operation whose backward pass
 *       is also computed by FFT (the input gradient is a full convolution with the filters,
 *       the filter gradient a correlation of the input with the output gradient).
 *   <li>{@link #spectralPool}: downsampling by keeping only the lowest frequencies (Rippel,
 *       Snoek and Adams), implemented as the separable real linear map
 *       {@code DFT, crop, inverse DFT} so it is differentiable through a matrix product.
 *   <li>{@link #spectralFilters}: a spectral parametrisation of {@code k x k} filters, a
 *       real basis of the 2-D DFT so that gradient steps move frequencies rather than taps.
 * </ul>
 *
 * <p>Papers: Mathieu, Henaff, LeCun, "Fast Training of Convolutional Networks through FFTs"
 * (ICLR 2014), <a href="https://arxiv.org/abs/1312.5851">arXiv:1312.5851</a>; Rippel, Snoek,
 * Adams, "Spectral Representations for Convolutional Neural Networks" (NeurIPS 2015),
 * <a href="https://arxiv.org/abs/1506.03767">arXiv:1506.03767</a>.
 */
public final class SpectralConv {

    private SpectralConv() {
    }

    /**
     * A stride-1 cross-correlation by FFT, numerically equal to
     * {@link Conv2dOps#direct} with stride 1.
     *
     * @param x {@code [in, h, w]}
     * @param in input channels
     * @param h rows
     * @param w columns
     * @param weight {@code [out, in, k, k]}
     * @param out output channels
     * @param k kernel side
     * @param pad zero padding
     * @return {@code [out, h + 2 pad - k + 1, w + 2 pad - k + 1]}
     */
    public static float[] fftConv(float[] x, int in, int h, int w, float[] weight, int out, int k, int pad) {
        int ph = h + 2 * pad;
        int pw = w + 2 * pad;
        int nh = Fft2d.nextPowerOfTwo(ph);
        int nw = Fft2d.nextPowerOfTwo(pw);
        int oh = ph - k + 1;
        int ow = pw - k + 1;
        int size = nh * nw;
        double[][] xr = new double[in][];
        double[][] xi = new double[in][];
        for (int c = 0; c < in; c++) {
            double[] re = new double[size];
            for (int y = 0; y < h; y++) {
                for (int xx = 0; xx < w; xx++) {
                    re[(y + pad) * nw + xx + pad] = x[(c * h + y) * w + xx];
                }
            }
            double[] im = new double[size];
            Fft2d.fft2d(re, im, nh, nw, false);
            xr[c] = re;
            xi[c] = im;
        }
        float[] y = new float[out * oh * ow];
        double[] fr = new double[size];
        double[] fi = new double[size];
        double[] accR = new double[size];
        double[] accI = new double[size];
        for (int o = 0; o < out; o++) {
            java.util.Arrays.fill(accR, 0);
            java.util.Arrays.fill(accI, 0);
            for (int c = 0; c < in; c++) {
                java.util.Arrays.fill(fr, 0);
                java.util.Arrays.fill(fi, 0);
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {
                        fr[ky * nw + kx] = weight[((o * in + c) * k + ky) * k + kx];
                    }
                }
                Fft2d.fft2d(fr, fi, nh, nw, false);
                double[] ar = xr[c];
                double[] ai = xi[c];
                for (int j = 0; j < size; j++) {
                    // X * conj(W): correlation
                    accR[j] += ar[j] * fr[j] + ai[j] * fi[j];
                    accI[j] += ai[j] * fr[j] - ar[j] * fi[j];
                }
            }
            Fft2d.fft2d(accR, accI, nh, nw, true);
            for (int yy = 0; yy < oh; yy++) {
                for (int xx = 0; xx < ow; xx++) {
                    y[(o * oh + yy) * ow + xx] = (float) accR[yy * nw + xx];
                }
            }
        }
        return y;
    }

    /** Circular correlation of {@code a} ({@code ah x aw}) with {@code b}, read at {@code rh x rw}. */
    private static float[] correlate(float[] a, int ah, int aw, float[] b, int bh, int bw, int rh, int rw,
            boolean flipB) {
        int nh = Fft2d.nextPowerOfTwo(Math.max(ah, bh + rh));
        int nw = Fft2d.nextPowerOfTwo(Math.max(aw, bw + rw));
        double[] ar = new double[nh * nw];
        double[] ai = new double[nh * nw];
        double[] br = new double[nh * nw];
        double[] bi = new double[nh * nw];
        for (int y = 0; y < ah; y++) {
            for (int x = 0; x < aw; x++) {
                ar[y * nw + x] = a[y * aw + x];
            }
        }
        for (int y = 0; y < bh; y++) {
            for (int x = 0; x < bw; x++) {
                br[y * nw + x] = b[y * bw + x];
            }
        }
        Fft2d.fft2d(ar, ai, nh, nw, false);
        Fft2d.fft2d(br, bi, nh, nw, false);
        for (int j = 0; j < ar.length; j++) {
            double re;
            double im;
            if (flipB) {
                // true convolution: A * B
                re = ar[j] * br[j] - ai[j] * bi[j];
                im = ar[j] * bi[j] + ai[j] * br[j];
            } else {
                re = ar[j] * br[j] + ai[j] * bi[j];
                im = ai[j] * br[j] - ar[j] * bi[j];
            }
            ar[j] = re;
            ai[j] = im;
        }
        Fft2d.fft2d(ar, ai, nh, nw, true);
        float[] out = new float[rh * rw];
        for (int y = 0; y < rh; y++) {
            for (int x = 0; x < rw; x++) {
                out[y * rw + x] = (float) ar[y * nw + x];
            }
        }
        return out;
    }

    /**
     * A differentiable stride-1 convolution computed by FFT in both passes.
     *
     * @param x {@code [batch, in, h, w]}
     * @param weight {@code [out, in, k, k]}
     * @param pad zero padding on every side
     * @return {@code [batch, out, h + 2 pad - k + 1, w + 2 pad - k + 1]}
     */
    public static Var conv(Var x, Var weight, int pad) {
        Shape xs = x.shape();
        Shape ws = weight.shape();
        int b = xs.dim(0);
        int in = xs.dim(1);
        int h = xs.dim(2);
        int w = xs.dim(3);
        int out = ws.dim(0);
        int k = ws.dim(2);
        int ph = h + 2 * pad;
        int pw = w + 2 * pad;
        int oh = ph - k + 1;
        int ow = pw - k + 1;
        return Var.custom(v -> {
            float[] xv = v[0].toArray();
            float[] wv = v[1].toArray();
            float[] y = new float[b * out * oh * ow];
            for (int n = 0; n < b; n++) {
                float[] img = java.util.Arrays.copyOfRange(xv, n * in * h * w, (n + 1) * in * h * w);
                float[] r = fftConv(img, in, h, w, wv, out, k, pad);
                System.arraycopy(r, 0, y, n * out * oh * ow, r.length);
            }
            return Tensor.of(Shape.of(b, out, oh, ow), y);
        }, (v, output, grad) -> {
            float[] xv = v[0].toArray();
            float[] wv = v[1].toArray();
            float[] g = grad.toArray();
            float[] dx = new float[b * in * h * w];
            float[] dw = new float[out * in * k * k];
            for (int n = 0; n < b; n++) {
                float[] xp = Conv2dOps.pad(java.util.Arrays.copyOfRange(xv, n * in * h * w, (n + 1) * in * h * w), in,
                        h, w, pad);
                for (int o = 0; o < out; o++) {
                    float[] go = java.util.Arrays.copyOfRange(g, (n * out + o) * oh * ow, (n * out + o + 1) * oh * ow);
                    for (int c = 0; c < in; c++) {
                        float[] xc = java.util.Arrays.copyOfRange(xp, c * ph * pw, (c + 1) * ph * pw);
                        float[] dwc = correlate(xc, ph, pw, go, oh, ow, k, k, false);
                        for (int j = 0; j < k * k; j++) {
                            dw[(o * in + c) * k * k + j] += dwc[j];
                        }
                        float[] wc = java.util.Arrays.copyOfRange(wv, (o * in + c) * k * k, (o * in + c + 1) * k * k);
                        float[] full = correlate(go, oh, ow, wc, k, k, ph, pw, true);
                        for (int y = 0; y < h; y++) {
                            for (int xx = 0; xx < w; xx++) {
                                dx[((n * in + c) * h + y) * w + xx] += full[(y + pad) * pw + xx + pad];
                            }
                        }
                    }
                }
            }
            return new Tensor[] {Tensor.of(xs, dx), Tensor.of(ws, dw)};
        }, x, weight);
    }

    /**
     * The real {@code m x n} matrix that keeps the {@code m} lowest frequencies of a length-
     * {@code n} signal: DFT, keep frequencies {@code |f| <= m / 2} (the shared Nyquist term
     * split symmetrically for even {@code m}), inverse DFT of length {@code m}, scaled so a
     * constant signal stays constant.
     *
     * @param n input length
     * @param m output length, at most {@code n}
     * @return {@code [m, n]}, row-major
     */
    public static float[] lowPassMatrix(int n, int m) {
        float[] a = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int f = -(m / 2); f <= m / 2; f++) {
                    double weight = (m % 2 == 0 && Math.abs(f) == m / 2) ? 0.5 : 1;
                    // inverse DFT at output i of the frequency f coefficient of input j
                    s += weight * Math.cos(2 * Math.PI * f * (i / (double) m - j / (double) n));
                }
                a[i * n + j] = (float) (s / n);
            }
        }
        return a;
    }

    /**
     * Spectral pooling: keeps the lowest {@code oh x ow} frequencies of every channel.
     *
     * @param x {@code [batch, c, h, w]}
     * @param oh output rows
     * @param ow output columns
     * @return {@code [batch, c, oh, ow]}
     */
    public static Var spectralPool(Var x, int oh, int ow) {
        int h = x.shape().dim(2);
        int w = x.shape().dim(3);
        Tensor rowsT = Tensor.of(Shape.of(h, oh), transpose(lowPassMatrix(h, oh), oh, h));
        Tensor cols = Tensor.of(Shape.of(w, ow), transpose(lowPassMatrix(w, ow), ow, w));
        Var y = x.matmul(Var.of(cols));                     // [b, c, h, ow]
        return y.permute(0, 1, 3, 2).matmul(Var.of(rowsT)).permute(0, 1, 3, 2);
    }

    private static float[] transpose(float[] a, int rows, int cols) {
        float[] t = new float[a.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t[j * rows + i] = a[i * cols + j];
            }
        }
        return t;
    }

    /**
     * The real 2-D Fourier basis of {@code k x k} filters: {@code k^2} rows, each a cosine or
     * sine pattern, orthonormal. Filters are {@code theta B} for spectral coefficients
     * {@code theta [filters, k^2]}.
     *
     * @param k the kernel side
     * @return {@code [k^2, k^2]}
     */
    public static Tensor fourierBasis(int k) {
        int n = k * k;
        double[][] rows = new double[n][];
        int r = 0;
        for (int u = 0; u < k && r < n; u++) {
            for (int v = 0; v < k && r < n; v++) {
                for (int kind = 0; kind < 2 && r < n; kind++) {
                    double[] row = new double[n];
                    for (int y = 0; y < k; y++) {
                        for (int x = 0; x < k; x++) {
                            double ang = 2 * Math.PI * (u * y + v * x) / k;
                            row[y * k + x] = kind == 0 ? Math.cos(ang) : Math.sin(ang);
                        }
                    }
                    // Gram-Schmidt against earlier rows
                    for (int p = 0; p < r; p++) {
                        double dot = 0;
                        for (int j = 0; j < n; j++) {
                            dot += row[j] * rows[p][j];
                        }
                        for (int j = 0; j < n; j++) {
                            row[j] -= dot * rows[p][j];
                        }
                    }
                    double norm = 0;
                    for (double z : row) {
                        norm += z * z;
                    }
                    if (norm > 1e-8) {
                        norm = Math.sqrt(norm);
                        for (int j = 0; j < n; j++) {
                            row[j] /= norm;
                        }
                        rows[r++] = row;
                    }
                }
            }
        }
        float[] out = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                out[i * n + j] = (float) rows[i][j];
            }
        }
        return Tensor.of(Shape.of(n, n), out);
    }

    /**
     * Filters from spectral coefficients.
     *
     * @param theta {@code [out, in * k * k]} coefficients, {@code k^2} per input channel
     * @param out output channels
     * @param in input channels
     * @param k kernel side
     * @return filters {@code [out, in, k, k]}
     */
    public static Var spectralFilters(Var theta, int out, int in, int k) {
        Var basis = Var.of(fourierBasis(k));
        return theta.reshape(out * in, k * k).matmul(basis).reshape(out, in, k, k);
    }
}
