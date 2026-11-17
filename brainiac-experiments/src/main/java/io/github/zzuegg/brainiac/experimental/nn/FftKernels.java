package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A small host-side FFT and the FFT-based operations built on it: causal long convolution
 * (Hyena, S4) and the truncated spectral convolution of a Fourier Neural Operator. Package
 * private on purpose: it stands in for a core FFT tensor operation.
 */
final class FftKernels {

    private FftKernels() {
    }

    /** The smallest power of two at least {@code n}. */
    static int powerOfTwo(int n) {
        int p = 1;
        while (p < n) {
            p <<= 1;
        }
        return p;
    }

    /**
     * In-place iterative radix-2 FFT. {@code sign = -1} computes {@code X_k = sum x_s e^{-2 pi i k s / n}},
     * {@code sign = +1} the same with {@code +} (no 1/n scaling).
     */
    static void fft(double[] re, double[] im, int sign) {
        int n = re.length;
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("FFT length must be a power of two, got " + n);
        }
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double t = re[i];
                re[i] = re[j];
                re[j] = t;
                t = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double angle = sign * 2 * Math.PI / len;
            double wr = Math.cos(angle);
            double wi = Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                double cr = 1;
                double ci = 0;
                for (int k = 0; k < len / 2; k++) {
                    int a = i + k;
                    int b = a + len / 2;
                    double xr = re[b] * cr - im[b] * ci;
                    double xi = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                    double nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
    }

    /** Causal convolution of real sequences: {@code y[t] = sum_{s <= t} h[t - s] u[s]}, by FFT. */
    static double[] causalConvolve(double[] u, double[] h) {
        int length = u.length;
        int n = powerOfTwo(2 * length);
        double[] ar = new double[n];
        double[] ai = new double[n];
        double[] br = new double[n];
        double[] bi = new double[n];
        System.arraycopy(u, 0, ar, 0, length);
        System.arraycopy(h, 0, br, 0, Math.min(h.length, length));
        fft(ar, ai, -1);
        fft(br, bi, -1);
        for (int k = 0; k < n; k++) {
            double r = ar[k] * br[k] - ai[k] * bi[k];
            double i = ar[k] * bi[k] + ai[k] * br[k];
            ar[k] = r;
            ai[k] = i;
        }
        fft(ar, ai, 1);
        double[] y = new double[length];
        for (int t = 0; t < length; t++) {
            y[t] = ar[t] / n;
        }
        return y;
    }

    private static double[] reverse(double[] x) {
        double[] r = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            r[i] = x[x.length - 1 - i];
        }
        return r;
    }

    /**
     * Differentiable causal long convolution along the last axis: {@code u [rows..., channels, L]}
     * with one filter per channel {@code h [channels, L]}, by FFT in {@code O(L log L)}.
     */
    static Var causalConvolution(Var u, Var h) {
        return Var.apply(new CausalConvolution(), u, h);
    }

    /** The op behind {@link #causalConvolution}. */
    static final class CausalConvolution implements Op {

        @Override
        public Tensor forward(Tensor... in) {
            return run(in[0], in[1], false);
        }

        /** {@code reversed}: computes {@code rev(conv(rev(u), h))}, the adjoint in {@code u}. */
        private static Tensor run(Tensor u, Tensor h, boolean reversed) {
            Shape s = u.shape();
            int length = s.dim(-1);
            int channels = s.dim(-2);
            if (!h.shape().equals(Shape.of(channels, length))) {
                throw new IllegalArgumentException("filter " + h.shape() + " does not fit input " + s);
            }
            float[] uv = u.toArray();
            float[] hv = h.toArray();
            float[] out = new float[uv.length];
            int rows = uv.length / length;
            for (int r = 0; r < rows; r++) {
                int c = r % channels;
                double[] a = new double[length];
                double[] f = new double[length];
                for (int t = 0; t < length; t++) {
                    a[t] = uv[r * length + t];
                    f[t] = hv[c * length + t];
                }
                double[] y = reversed ? reverse(causalConvolve(reverse(a), f)) : causalConvolve(a, f);
                for (int t = 0; t < length; t++) {
                    out[r * length + t] = (float) y[t];
                }
            }
            return Tensor.of(s, out);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
            Tensor du = run(grad, in[1], true);
            // dh[c] = sum over rows of channel c of rev(conv(rev(g), u))
            Shape s = in[0].shape();
            int length = s.dim(-1);
            int channels = s.dim(-2);
            float[] g = grad.toArray();
            float[] uv = in[0].toArray();
            double[] dh = new double[channels * length];
            int rows = g.length / length;
            for (int r = 0; r < rows; r++) {
                int c = r % channels;
                double[] gr = new double[length];
                double[] ur = new double[length];
                for (int t = 0; t < length; t++) {
                    gr[t] = g[r * length + length - 1 - t];
                    ur[t] = uv[r * length + t];
                }
                double[] y = causalConvolve(gr, ur);
                for (int k = 0; k < length; k++) {
                    dh[c * length + k] += y[length - 1 - k];
                }
            }
            float[] dhf = new float[dh.length];
            for (int i = 0; i < dh.length; i++) {
                dhf[i] = (float) dh[i];
            }
            return new Tensor[] {du, Tensor.of(Shape.of(channels, length), dhf)};
        }

        @Override
        public String name() {
            return "fftCausalConvolution";
        }
    }

    /**
     * Differentiable truncated spectral convolution (the Fourier layer of an FNO): the lowest
     * {@code modes} Fourier coefficients of every input channel are mixed by complex weights,
     * the rest dropped, and the result transformed back.
     *
     * @param u {@code [batch, inChannels, n]}, {@code n} a power of two
     * @param wRe real parts {@code [modes, inChannels, outChannels]}
     * @param wIm imaginary parts, same shape
     * @return {@code [batch, outChannels, n]}
     */
    static Var spectralConvolution(Var u, Var wRe, Var wIm) {
        return Var.apply(new SpectralConvolution(), u, wRe, wIm);
    }

    /** The op behind {@link #spectralConvolution}. */
    static final class SpectralConvolution implements Op {

        @Override
        public Tensor forward(Tensor... in) {
            Shape s = in[0].shape();
            int batch = s.dim(0);
            int cin = s.dim(1);
            int n = s.dim(2);
            Shape w = in[1].shape();
            int modes = w.dim(0);
            int cout = w.dim(2);
            if (modes > n / 2 || w.dim(1) != cin) {
                throw new IllegalArgumentException("weights " + w + " do not fit input " + s);
            }
            double[][][] ur = new double[batch][cin][];
            double[][][] ui = new double[batch][cin][];
            spectra(in[0].toArray(), batch, cin, n, ur, ui);
            float[] wr = in[1].toArray();
            float[] wi = in[2].toArray();
            float[] out = new float[batch * cout * n];
            for (int b = 0; b < batch; b++) {
                for (int o = 0; o < cout; o++) {
                    double[] yr = new double[n];
                    double[] yi = new double[n];
                    for (int k = 0; k < modes; k++) {
                        double sr = 0;
                        double si = 0;
                        for (int c = 0; c < cin; c++) {
                            int idx = (k * cin + c) * cout + o;
                            sr += wr[idx] * ur[b][c][k] - wi[idx] * ui[b][c][k];
                            si += wr[idx] * ui[b][c][k] + wi[idx] * ur[b][c][k];
                        }
                        double scale = k == 0 ? 1 : 2;
                        yr[k] = scale * sr;
                        yi[k] = scale * si;
                    }
                    fft(yr, yi, 1);
                    for (int t = 0; t < n; t++) {
                        out[(b * cout + o) * n + t] = (float) (yr[t] / n);
                    }
                }
            }
            return Tensor.of(Shape.of(batch, cout, n), out);
        }

        private static void spectra(float[] v, int batch, int channels, int n, double[][][] re, double[][][] im) {
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < channels; c++) {
                    double[] r = new double[n];
                    double[] i = new double[n];
                    for (int t = 0; t < n; t++) {
                        r[t] = v[(b * channels + c) * n + t];
                    }
                    fft(r, i, -1);
                    re[b][c] = r;
                    im[b][c] = i;
                }
            }
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
            Shape s = in[0].shape();
            int batch = s.dim(0);
            int cin = s.dim(1);
            int n = s.dim(2);
            int modes = in[1].shape().dim(0);
            int cout = in[1].shape().dim(2);
            double[][][] ur = new double[batch][cin][];
            double[][][] ui = new double[batch][cin][];
            spectra(in[0].toArray(), batch, cin, n, ur, ui);
            double[][][] gr = new double[batch][cout][];
            double[][][] gi = new double[batch][cout][];
            spectra(grad.toArray(), batch, cout, n, gr, gi);
            float[] wr = in[1].toArray();
            float[] wi = in[2].toArray();
            float[] dwr = new float[wr.length];
            float[] dwi = new float[wi.length];
            float[] du = new float[batch * cin * n];
            for (int b = 0; b < batch; b++) {
                double[][] dur = new double[cin][n];
                double[][] dui = new double[cin][n];
                for (int k = 0; k < modes; k++) {
                    double scale = (k == 0 ? 1.0 : 2.0) / n;
                    for (int o = 0; o < cout; o++) {
                        // dL/dY_k = scale * G_k
                        double yr = scale * gr[b][o][k];
                        double yi = scale * gi[b][o][k];
                        for (int c = 0; c < cin; c++) {
                            int idx = (k * cin + c) * cout + o;
                            // dU = conj(W) dY, dW = conj(U) dY
                            dur[c][k] += wr[idx] * yr + wi[idx] * yi;
                            dui[c][k] += wr[idx] * yi - wi[idx] * yr;
                            dwr[idx] += (float) (ur[b][c][k] * yr + ui[b][c][k] * yi);
                            dwi[idx] += (float) (ur[b][c][k] * yi - ui[b][c][k] * yr);
                        }
                    }
                }
                for (int c = 0; c < cin; c++) {
                    // du[s] = Re(sum_k dU_k e^{+i w_k s})
                    fft(dur[c], dui[c], 1);
                    for (int t = 0; t < n; t++) {
                        du[(b * cin + c) * n + t] = (float) dur[c][t];
                    }
                }
            }
            return new Tensor[] {Tensor.of(s, du), Tensor.of(in[1].shape(), dwr), Tensor.of(in[2].shape(), dwi)};
        }

        @Override
        public String name() {
            return "spectralConvolution";
        }
    }
}
