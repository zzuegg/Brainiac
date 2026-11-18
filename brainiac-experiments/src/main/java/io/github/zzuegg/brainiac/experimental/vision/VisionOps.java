package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Small convolution helpers shared by the dense-prediction, detection and video blocks of
 * this package: He-initialised filter banks, same-padded convolutions with a stride or a
 * dilation, 2x2 max pooling and nearest-neighbour upsampling. They are thin wrappers over
 * core {@link Functional#conv} and {@link Functional#pool}, so every block runs on any
 * backend and differentiates through autograd.
 */
public final class VisionOps {

    private VisionOps() {
    }

    /**
     * A He-normal filter bank.
     *
     * @param random the source of weights
     * @param out output channels
     * @param in input channels
     * @param kernel the square kernel side
     * @return a parameter {@code [out, in, kernel, kernel]}
     */
    public static Var convWeight(RandomGenerator random, int out, int in, int kernel) {
        return Var.param(Tensor.normal(Shape.of(out, in, kernel, kernel), 0,
                (float) Math.sqrt(2.0 / (in * kernel * kernel)), random));
    }

    /**
     * A zero bias.
     *
     * @param n its length
     * @return a parameter {@code [n]}
     */
    public static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /**
     * A same-padded convolution with a stride and a dilation.
     *
     * @param x {@code [batch, in, h, w]}
     * @param weight {@code [out, in, k, k]}
     * @param bias {@code [out]} or {@code null}
     * @param stride the stride
     * @param dilation the dilation
     * @return {@code [batch, out, ceil(h / stride), ceil(w / stride)]}
     */
    public static Var conv(Var x, Var weight, Var bias, int stride, int dilation) {
        int k = weight.shape().dim(2);
        int span = dilation * (k - 1);
        int before = span / 2;
        int h = x.shape().dim(2);
        int outH = (h + stride - 1) / stride;
        int after = Math.max(0, (outH - 1) * stride + span + 1 - h - before);
        return Functional.conv(x, weight, bias, new int[] {stride, stride}, new int[] {before, before},
                new int[] {after, after}, new int[] {dilation, dilation}, 1);
    }

    /**
     * A same-padded, stride-1 convolution.
     *
     * @param x {@code [batch, in, h, w]}
     * @param weight {@code [out, in, k, k]}
     * @param bias {@code [out]} or {@code null}
     * @return {@code [batch, out, h, w]}
     */
    public static Var conv(Var x, Var weight, Var bias) {
        return conv(x, weight, bias, 1, 1);
    }

    /**
     * 2x2 max pooling with stride 2.
     *
     * @param x {@code [batch, c, h, w]} with even sides
     * @return {@code [batch, c, h / 2, w / 2]}
     */
    public static Var maxPool2(Var x) {
        return Functional.pool(x, Functional.PoolKind.MAX, new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0},
                new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
    }

    /**
     * Nearest-neighbour upsampling to a size.
     *
     * @param x {@code [batch, c, h, w]}
     * @param height the new height
     * @param width the new width
     * @return {@code [batch, c, height, width]}
     */
    public static Var upsample(Var x, int height, int width) {
        return Functional.resize(x, new int[] {height, width}, Functional.Interpolation.NEAREST, false);
    }

    /**
     * The number of multiply-adds of a convolution layer.
     *
     * @param in input channels
     * @param out output channels
     * @param kernel the kernel side
     * @param outHeight output rows
     * @param outWidth output columns
     * @return {@code in * out * kernel^2 * outHeight * outWidth}
     */
    public static long convMacs(int in, int out, int kernel, int outHeight, int outWidth) {
        return (long) in * out * kernel * kernel * outHeight * outWidth;
    }

    /**
     * The number of trainable scalars in a list of parameters.
     *
     * @param params the parameters
     * @return their total size
     */
    public static long count(java.util.List<Var> params) {
        long n = 0;
        for (Var p : params) {
            n += p.value().size();
        }
        return n;
    }
}
