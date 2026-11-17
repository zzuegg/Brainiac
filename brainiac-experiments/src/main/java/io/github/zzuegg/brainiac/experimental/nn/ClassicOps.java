package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/** Shared helpers of the classic deep learning blocks (regularisers, CNN, RNN and set blocks). */
final class ClassicOps {

    private ClassicOps() {
    }

    /** A He-normal convolution filter bank {@code [out, in, k, k]}. */
    static Var convWeight(RandomGenerator random, int out, int in, int kernel) {
        return Var.param(Tensor.normal(Shape.of(out, in, kernel, kernel), 0,
                (float) Math.sqrt(2.0 / (in * kernel * kernel)), random));
    }

    /** A zero vector parameter. */
    static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /** A same-padded, stride-1 2-D convolution. */
    static Var conv(Var x, Var weight, Var bias, int groups) {
        int k = weight.shape().dim(2);
        int p = k / 2;
        return Functional.conv(x, weight, bias, new int[] {1, 1}, new int[] {p, p}, new int[] {k - 1 - p, k - 1 - p},
                new int[] {1, 1}, groups);
    }

    /** 0/1 values, each 1 with probability {@code keep}. */
    static Tensor bernoulli(Shape shape, float keep, RandomGenerator random, Device device) {
        int n = Math.toIntExact(shape.size());
        float[] m = new float[n];
        for (int i = 0; i < n; i++) {
            m[i] = random.nextFloat() < keep ? 1f : 0f;
        }
        return Tensor.of(shape, m, device);
    }

    /** Uniform values in {@code [low, high)} as a tensor. */
    static Tensor uniform(Shape shape, float low, float high, RandomGenerator random, Device device) {
        int n = Math.toIntExact(shape.size());
        float[] m = new float[n];
        for (int i = 0; i < n; i++) {
            m[i] = low + (high - low) * random.nextFloat();
        }
        return Tensor.of(shape, m, device);
    }

    /** Requires {@code [batch, channels, height, width]} with a given channel count (0: any). */
    static void requireImages(Var x, int channels, Object layer) {
        if (x.shape().rank() != 4 || (channels > 0 && x.shape().dim(1) != channels)) {
            throw new IllegalArgumentException(layer + " expects [batch, " + (channels > 0 ? channels : "c")
                    + ", h, w], got " + x.shape());
        }
    }

    /** Mean over the spatial axes: {@code [b, c, h, w] -> [b, c]}. */
    static Var spatialMean(Var x) {
        return x.mean(3, false).mean(2, false);
    }

    /** A spec whose output shape is computed by a function. */
    static LayerSpec spec(Function<Shape, Shape> shape, BiFunction<Shape, SplittableRandom, Layer> factory) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return shape.apply(input);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return factory.apply(input, random);
            }
        };
    }

    /** Replaces the last dimension of a shape. */
    static Shape withLast(Shape s, int last) {
        int[] d = s.toArray();
        d[d.length - 1] = last;
        return Shape.of(d);
    }

    /** Token ids as a {@code [rows, cols]} tensor. */
    static Tensor ids(int[][] ids) {
        int r = ids.length;
        int c = ids[0].length;
        float[] f = new float[r * c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                f[i * c + j] = ids[i][j];
            }
        }
        return Tensor.of(Shape.of(r, c), f);
    }

    /** Token ids as a {@code [n]} tensor. */
    static Tensor ids(int[] ids) {
        float[] f = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            f[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), f);
    }

    /** Argmax of each row of a {@code [rows, cols]} tensor. */
    static int[] argmax(Tensor t) {
        int rows = t.shape().dim(0);
        int cols = t.shape().dim(1);
        float[] v = t.toArray();
        int[] out = new int[rows];
        for (int r = 0; r < rows; r++) {
            int best = 0;
            for (int c = 1; c < cols; c++) {
                if (v[r * cols + c] > v[r * cols + best]) {
                    best = c;
                }
            }
            out[r] = best;
        }
        return out;
    }

    /** A Xavier-uniform matrix parameter. */
    static Var matrix(RandomGenerator random, int rows, int cols) {
        float a = (float) Math.sqrt(6.0 / (rows + cols));
        return Var.param(Tensor.uniform(Shape.of(rows, cols), -a, a, random));
    }
}
