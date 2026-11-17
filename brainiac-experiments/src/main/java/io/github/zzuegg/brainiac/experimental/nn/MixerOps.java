package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
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

    /** Upper-triangular ones {@code U[s, t] = 1 for s <= t}: {@code x.matmul(U)} is a cumulative sum over the last axis. */
    static Tensor cumsumMatrix(int t) {
        float[] m = new float[t * t];
        for (int s = 0; s < t; s++) {
            for (int j = s; j < t; j++) {
                m[s * t + j] = 1;
            }
        }
        return Tensor.of(Shape.of(t, t), m);
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

    /**
     * Differentiable scatter-add, the adjoint of {@link Var#gather}: adds rows of {@code values}
     * into a zero tensor at {@code indices} along {@code axis}.
     */
    static Var scatterAdd(Var values, int axis, Tensor indices, int length) {
        return Var.apply(new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[0].scatterAdd(axis, indices, length);
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
                return new Tensor[] {grad.gather(axis, indices)};
            }

            @Override
            public String name() {
                return "scatterAdd";
            }
        }, values);
    }

    /** Indices of the {@code k} largest entries of {@code row[from, from + n)}, best first. */
    static int[] topK(float[] row, int from, int n, int k) {
        int[] best = new int[k];
        boolean[] taken = new boolean[n];
        for (int j = 0; j < k; j++) {
            int arg = -1;
            for (int i = 0; i < n; i++) {
                if (!taken[i] && (arg < 0 || row[from + i] > row[from + arg])) {
                    arg = i;
                }
            }
            taken[arg] = true;
            best[j] = arg;
        }
        return best;
    }
}
