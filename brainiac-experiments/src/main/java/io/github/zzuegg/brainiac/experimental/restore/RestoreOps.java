package io.github.zzuegg.brainiac.experimental.restore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/** Shared convolution helpers of the restoration blocks. */
final class RestoreOps {

    private RestoreOps() {
    }

    /** A same-padded 2-D convolution: weight and bias, with its multiply-add count per pixel. */
    static final class Conv {
        final Var weight;
        final Var bias;
        final int in;
        final int out;
        final int kernel;

        Conv(int in, int out, int kernel, RandomGenerator random) {
            this(in, out, kernel, (float) Math.sqrt(2.0 / (in * kernel * kernel)), random);
        }

        Conv(int in, int out, int kernel, float std, RandomGenerator random) {
            this.in = in;
            this.out = out;
            this.kernel = kernel;
            weight = Var.param(Tensor.normal(Shape.of(out, in, kernel, kernel), 0, std, random));
            bias = Var.param(Tensor.zeros(Shape.of(out)));
        }

        Var forward(Var x) {
            int p = kernel / 2;
            return Functional.conv(x, weight, bias, new int[] {1, 1}, new int[] {p, p}, new int[] {p, p},
                    new int[] {1, 1}, 1);
        }

        long macsPerPixel() {
            return (long) in * out * kernel * kernel;
        }

        List<Var> parameters() {
            return List.of(weight, bias);
        }
    }

    static long count(List<Var> params) {
        long n = 0;
        for (Var v : params) {
            n += v.shape().size();
        }
        return n;
    }

    static Var leaky(Var x, float slope) {
        return x.relu().add(x.neg().relu().mul(-slope));
    }

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
}
