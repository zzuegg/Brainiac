package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.random.RandomGenerator;

/** Small shared building blocks for this package: parameters, head reshapes, norms. */
final class MixerOps {

    private MixerOps() {
    }

    /** A weight matrix {@code [in, out]} drawn uniformly with the Xavier/Glorot range. */
    static Var weight(RandomGenerator random, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, random));
    }

    /** A weight tensor of any shape drawn uniformly from {@code [-scale, scale]}. */
    static Var uniform(RandomGenerator random, float scale, int... dims) {
        return Var.param(Tensor.uniform(Shape.of(dims), -scale, scale, random));
    }

    /** A parameter filled with one value. */
    static Var filled(float value, int... dims) {
        return Var.param(Tensor.full(Shape.of(dims), value));
    }

    /** Softmax over the last axis. */
    static Var softmax(Var x) {
        return Activation.SOFTMAX.forward(x);
    }

    /** RMS norm over the last axis without a learned scale. */
    static Var rmsNorm(Var x) {
        return x.div(x.square().mean(-1, true).add(1e-6f).sqrt());
    }

    /** Projects to the unit sphere along the last axis. */
    static Var unit(Var x) {
        return x.div(x.square().sum(-1, true).add(1e-8f).sqrt());
    }

    /** {@code [b, t, h * d] -> [b, h, t, d]}. */
    static Var split(Var x, int heads) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int d = x.shape().dim(2) / heads;
        return x.reshape(b, t, heads, d).permute(0, 2, 1, 3);
    }

    /** {@code [b, h, t, d] -> [b, t, h * d]}. */
    static Var merge(Var x) {
        int b = x.shape().dim(0);
        int h = x.shape().dim(1);
        int t = x.shape().dim(2);
        int d = x.shape().dim(3);
        return x.permute(0, 2, 1, 3).reshape(b, t, h * d);
    }

    /** A causal mask bias {@code [t, t]}. */
    static Tensor causal(int t) {
        return Functional.causalMask(t, t, Tensor.zeros(Shape.of(1)).device());
    }

    /** {@code [b, t, d]} input check. */
    static void requireSequence(Var x, int features, String layer) {
        Shape s = x.shape();
        if (s.rank() != 3 || s.dim(2) != features) {
            throw new IllegalArgumentException(layer + " expects [batch, steps, " + features + "], got " + s);
        }
    }

    /** A two-layer SiLU MLP applied to the last axis. */
    static Var mlp(Var x, Var w1, Var w2) {
        return Activation.SILU.forward(x.matmul(w1)).matmul(w2);
    }

    /** A spec for a layer that keeps its input shape; {@code make} gets the sample shape (no batch). */
    static LayerSpec sameShape(BiFunction<Shape, SplittableRandom, Layer> make) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return make.apply(input, random);
            }
        };
    }

    /** Whole-number ids as a float tensor, for gather and scatter. */
    static Tensor ids(int[] values) {
        float[] f = new float[values.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = values[i];
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /** Indices of the {@code k} largest entries of {@code row[from, from + n)}, best first (core {@link Tensor#topk}). */
    static int[] topK(float[] row, int from, int n, int k) {
        float[] idx = Tensor.of(Shape.of(n), java.util.Arrays.copyOfRange(row, from, from + n)).topk(k)[1].toArray();
        int[] best = new int[k];
        for (int j = 0; j < k; j++) {
            best[j] = (int) idx[j];
        }
        return best;
    }

    /**
     * Depthwise causal convolution along steps: {@code y[t] = sum_j w[j] x[t - j]}.
     *
     * @param x {@code [b, t, d]}
     * @param weight {@code [kernel, d]}
     */
    static Var causalDepthwise(Var x, Var weight) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int d = x.shape().dim(2);
        int kernel = weight.shape().dim(0);
        Var y = x.mul(weight.slice(0, 0, 1));
        for (int j = 1; j < kernel && j < t; j++) {
            Var shifted = Var.concat(1, Var.of(Tensor.zeros(Shape.of(b, j, d))), x.slice(1, 0, t - j));
            y = y.add(shifted.mul(weight.slice(0, j, j + 1)));
        }
        return y;
    }

    /** Lower-triangular ones including the diagonal, {@code [t, t]}. */
    static Tensor lowerOnes(int t) {
        float[] m = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j <= i; j++) {
                m[i * t + j] = 1;
            }
        }
        return Tensor.of(Shape.of(t, t), m);
    }

    /**
     * {@code exp(c_i - c_j)} for {@code j <= i} and 0 above the diagonal, along the last axis
     * of a cumulative log-decay {@code c [..., t]}: the decay between step {@code j} and
     * {@code i}. The exponent is clamped at 0 above the diagonal, so it never overflows.
     */
    static Var segmentDecay(Var c) {
        Shape s = c.shape();
        int t = s.dim(-1);
        int[] row = s.toArray();
        int[] col = s.toArray();
        int[] ci = java.util.Arrays.copyOf(row, row.length + 1);
        ci[ci.length - 2] = t;
        ci[ci.length - 1] = 1;
        int[] cj = java.util.Arrays.copyOf(col, col.length + 1);
        cj[cj.length - 2] = 1;
        cj[cj.length - 1] = t;
        Tensor mask = lowerOnes(t);
        Var diff = c.reshape(ci).sub(c.reshape(cj));   // [..., t, t]
        return diff.mul(mask).exp().mul(mask);
    }

    /**
     * Causal long convolution along the last axis with one filter per channel, by FFT:
     * {@code y[t] = sum_{s <= t} h[t - s] u[s]} in {@code O(L log L)}.
     *
     * @param u {@code [..., channels, L]}
     * @param h {@code [channels, L]}
     * @return {@code [..., channels, L]}
     */
    static Var causalConvolution(Var u, Var h) {
        int length = u.shape().dim(-1);
        int n = 1;
        while (n < 2 * length) {
            n <<= 1;
        }
        Var uf = padLast(u, n).rfft();
        Var hf = padLast(h, n).rfft();
        return Complex.mul(uf, hf).irfft(n).slice(-1, 0, length);
    }

    /** Zero-pads the last axis to {@code n}. */
    static Var padLast(Var x, int n) {
        int[] dims = x.shape().toArray();
        int length = dims[dims.length - 1];
        if (length == n) {
            return x;
        }
        dims[dims.length - 1] = n - length;
        return Var.concat(-1, x, Var.of(Tensor.zeros(Shape.of(dims))));
    }

    /**
     * Truncated spectral convolution, the Fourier layer of an FNO: the lowest {@code modes}
     * Fourier coefficients of every input channel are mixed by complex weights per mode, the
     * others dropped, and the result transformed back.
     *
     * @param u {@code [batch, inChannels, n]}
     * @param wRe real parts {@code [modes, inChannels, outChannels]}
     * @param wIm imaginary parts, same shape
     * @return {@code [batch, outChannels, n]}
     */
    static Var spectralConvolution(Var u, Var wRe, Var wIm) {
        int b = u.shape().dim(0);
        int n = u.shape().dim(2);
        int modes = wRe.shape().dim(0);
        int cout = wRe.shape().dim(2);
        int half = n / 2 + 1;
        if (modes > half) {
            throw new IllegalArgumentException(modes + " modes do not fit length " + n);
        }
        Var spec = u.rfft().slice(2, 0, modes);                    // [b, cin, M, 2]
        Var ur = Complex.real(spec).permute(2, 0, 1);
        Var ui = Complex.imag(spec).permute(2, 0, 1);              // [M, b, cin]
        Var yr = ur.matmul(wRe).sub(ui.matmul(wIm));               // [M, b, cout]
        Var yi = ur.matmul(wIm).add(ui.matmul(wRe));
        Var y = Complex.of(yr.permute(1, 2, 0), yi.permute(1, 2, 0));   // [b, cout, M, 2]
        if (modes < half) {
            y = Var.concat(2, y, Var.of(Tensor.zeros(Shape.of(b, cout, half - modes, 2))));
        }
        return y.irfft(n);
    }
}
