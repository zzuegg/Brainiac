package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.random.RandomGenerator;

/** Small shared helpers for the normalization, position, vision and generative blocks. */
final class TechniqueOps {

    /** Large enough to zero a softmax weight, small enough to stay finite. */
    static final float MASKED = -1e9f;

    private TechniqueOps() {
    }

    /** A Xavier-uniform {@code [rows, cols]} parameter. */
    static Var weight(RandomGenerator random, int rows, int cols) {
        float a = (float) Math.sqrt(6.0 / (rows + cols));
        return Var.param(Tensor.uniform(Shape.of(rows, cols), -a, a, random));
    }

    /** A normal {@code N(0, std)} parameter of any shape. */
    static Var normal(RandomGenerator random, float std, int... dims) {
        return Var.param(Tensor.normal(Shape.of(dims), 0, std, random));
    }

    /** A parameter filled with one value. */
    static Var filled(float value, int... dims) {
        return Var.param(Tensor.full(Shape.of(dims), value));
    }

    /** {@code x w + b} over the last axis; {@code b} may be {@code null}. */
    static Var linear(Var x, Var w, Var b) {
        Var y = x.matmul(w);
        return b == null ? y : y.add(b);
    }

    /** {@code [batch, steps, heads * dh] -> [batch, heads, steps, dh]}. */
    static Var splitHeads(Var x, int heads) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int d = x.shape().dim(2);
        return x.reshape(b, t, heads, d / heads).permute(0, 2, 1, 3);
    }

    /** {@code [batch, heads, steps, dh] -> [batch, steps, heads * dh]}. */
    static Var mergeHeads(Var x) {
        int b = x.shape().dim(0);
        int h = x.shape().dim(1);
        int t = x.shape().dim(2);
        int dh = x.shape().dim(3);
        return x.permute(0, 2, 1, 3).reshape(b, t, h * dh);
    }

    /** RMS-normalizes the last axis without a learned scale. */
    static Var rms(Var x, float eps) {
        return x.div(x.square().mean(-1, true).add(eps).sqrt());
    }

    /** Layer-normalizes the last axis without learned parameters. */
    static Var standardize(Var x, float eps) {
        Var c = x.sub(x.mean(-1, true));
        return c.div(c.square().mean(-1, true).add(eps).sqrt());
    }

    /** A {@code [queries, keys]} causal mask: 0 where allowed, very negative elsewhere. */
    static Tensor causalMask(int steps, Device device) {
        float[] m = new float[steps * steps];
        for (int i = 0; i < steps; i++) {
            for (int j = i + 1; j < steps; j++) {
                m[i * steps + j] = MASKED;
            }
        }
        return Tensor.of(Shape.of(steps, steps), m, device);
    }

    /**
     * Softmax attention where the score bias may itself be learned:
     * {@code softmax(q k^T * scale + bias + mask) v}.
     *
     * @param q {@code [batch, heads, queries, d]}
     * @param k {@code [batch, heads, keys, d]}
     * @param v {@code [batch, heads, keys, dv]}
     * @param bias a differentiable score bias broadcast to the scores, or {@code null}
     * @param mask a constant score bias (such as a causal mask), or {@code null}
     * @param scale the score scale
     * @return {@code [batch, heads, queries, dv]}
     */
    static Var attend(Var q, Var k, Var v, Var bias, Tensor mask, float scale) {
        Var scores = q.matmul(k.transpose()).mul(scale);
        if (bias != null) {
            scores = scores.add(bias);
        }
        if (mask != null) {
            scores = scores.add(mask);
        }
        return Activation.SOFTMAX.forward(scores).matmul(v);
    }

    /**
     * Picks one entry per index from each row: {@code out[r, j] = x[r, idx[r][j]]}, with
     * gradients flowing back into {@code x}.
     *
     * @param x {@code [rows, width]}
     * @param idx {@code rows x count} indices into each row
     * @return {@code [rows, count]}
     */
    static Var takeAlong(Var x, int[][] idx) {
        int rows = x.shape().dim(0);
        int width = x.shape().dim(1);
        int count = idx[0].length;
        float[] flat = new float[rows * count];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < count; j++) {
                flat[r * count + j] = (float) r * width + idx[r][j];
            }
        }
        Tensor indices = Tensor.of(Shape.of(rows * count), flat, x.value().device());
        return x.reshape(rows * width).gather(0, indices).reshape(rows, count);
    }

    /** A spec for a layer whose output has the input's shape. */
    static LayerSpec sameShape(BiFunction<Shape, SplittableRandom, Layer> factory) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return factory.apply(input, random);
            }
        };
    }

    /** Reads a whole tensor as doubles. */
    static double[] doubles(Tensor t) {
        float[] f = t.toArray();
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }
}
