package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** Small building blocks shared by the audio generators. */
final class AudioNets {

    private AudioNets() {
    }

    /** A 1-D convolution kernel {@code [out, in, k]} with fan-in scaling. */
    static Var kernel(RandomGenerator r, int out, int in, int k) {
        return Var.param(Tensor.normal(Shape.of(out, in, k), 0, (float) Math.sqrt(1.0 / (in * k)), r));
    }

    /** A 2-D convolution kernel {@code [out, in, k, 1]} for the period discriminators. */
    static Var kernel2(RandomGenerator r, int out, int in, int k) {
        return Var.param(Tensor.normal(Shape.of(out, in, k, 1), 0, (float) Math.sqrt(1.0 / (in * k)), r));
    }

    /** A dense matrix {@code [in, out]}. */
    static Var dense(RandomGenerator r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
    }

    static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /**
     * A 1-D convolution with "same" padding at stride 1 (and {@code (k-1)/2} on both sides
     * otherwise).
     */
    static Var conv(Var x, Var w, Var b, int stride, int dilation) {
        int k = w.shape().dim(2);
        int pad = (k - 1) * dilation / 2;
        return Functional.conv(x, w, b, new int[] {stride}, new int[] {pad}, new int[] {pad}, new int[] {dilation}, 1);
    }

    /** A 2-D convolution along the first spatial axis only, kernel {@code [out, in, k, 1]}. */
    static Var conv2(Var x, Var w, Var b, int stride) {
        int k = w.shape().dim(2);
        int pad = (k - 1) / 2;
        return Functional.conv(x, w, b, new int[] {stride, 1}, new int[] {pad, 0}, new int[] {pad, 0},
                new int[] {1, 1}, 1);
    }

    /** Leaky ReLU with slope 0.2 (HiFi-GAN uses 0.1 in the generator, 0.2 is close enough here). */
    static Var leaky(Var x) {
        return x.maximum(x.mul(0.2f));
    }

    /** Sinusoidal features of scalars {@code [n]} (scaled by {@code scale}) followed by nothing. */
    static Tensor sinusoid(float[] values, float scale, int dim) {
        float[] v = new float[values.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = values[i] * scale;
        }
        return Functional.timestepEmbedding(Tensor.of(Shape.of(v.length), v), dim, 10000f);
    }

    static Tensor matrix(float[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows[i], 0, f, i * d, d);
        }
        return Tensor.of(Shape.of(n, d), f);
    }

    static Tensor cube(float[][][] x) {
        int n = x.length;
        int a = x[0].length;
        int b = x[0][0].length;
        float[] f = new float[n * a * b];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < a; j++) {
                System.arraycopy(x[i][j], 0, f, (i * a + j) * b, b);
            }
        }
        return Tensor.of(Shape.of(n, a, b), f);
    }

    static float[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = (int) (t.shape().size() / n);
        float[] f = t.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f, i * d, out[i], 0, d);
        }
        return out;
    }
}
