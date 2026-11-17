package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A deformable 2-D convolution (v1, "lite": stride 1, same padding, one offset group): a
 * small ordinary convolution predicts, at every output pixel, a 2-D offset for each of the
 * {@code k x k} kernel taps, and the main convolution samples the input at those shifted,
 * fractional positions with bilinear interpolation. The receptive field thus bends to the
 * object's shape instead of being a fixed square. The offsets start at zero, so the layer
 * starts as a plain convolution; gradients reach the offsets through the bilinear weights.
 *
 * <p>Paper: Dai et al., "Deformable Convolutional Networks" (2017),
 * <a href="https://arxiv.org/abs/1703.06211">arXiv:1703.06211</a>.
 */
public final class DeformableConv2d implements Layer {

    private final int channels;
    private final int filters;
    private final int kernel;
    private final Var weight;
    private final Var bias;
    private final Var offsetWeight;
    private final Var offsetBias;

    /**
     * Creates the layer.
     *
     * @param channels input channels
     * @param filters output channels
     * @param kernel the kernel side, odd
     * @param random the random source
     */
    public DeformableConv2d(int channels, int filters, int kernel, RandomGenerator random) {
        if (kernel % 2 == 0) {
            throw new IllegalArgumentException("kernel must be odd, got " + kernel);
        }
        this.channels = channels;
        this.filters = filters;
        this.kernel = kernel;
        int taps = kernel * kernel;
        this.weight = Var.param(Tensor.normal(Shape.of(channels * taps, filters), 0,
                (float) Math.sqrt(2.0 / (channels * taps)), random));
        this.bias = Var.param(Tensor.zeros(Shape.of(filters)));
        this.offsetWeight = Var.param(Tensor.zeros(Shape.of(2 * taps, channels, kernel, kernel)));
        this.offsetBias = Var.param(Tensor.zeros(Shape.of(2 * taps)));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)} on {@code [channels, h, w]}.
     *
     * @param filters output channels
     * @param kernel the kernel side
     * @return the spec
     */
    public static LayerSpec spec(int filters, int kernel) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(filters, input.dim(1), input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DeformableConv2d(input.dim(0), filters, kernel, random);
            }
        };
    }

    /**
     * The offsets the layer predicts for an input.
     *
     * @param input {@code [batch, channels, h, w]}
     * @return {@code [batch, 2 * k * k, h, w]}: for tap {@code t}, channel {@code 2t} is the
     *     row offset and {@code 2t + 1} the column offset
     */
    public Var offsets(Var input) {
        int p = kernel / 2;
        return Functional.conv(input, offsetWeight, offsetBias, new int[] {1, 1}, new int[] {p, p}, new int[] {p, p},
                new int[] {1, 1}, 1);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 4 || s.dim(1) != channels) {
            throw new IllegalArgumentException(this + " expects [batch, " + channels + ", h, w], got " + s);
        }
        return sample(input, offsets(input));
    }

    /**
     * Runs the deformable convolution with given offsets.
     *
     * @param input {@code [batch, channels, h, w]}
     * @param offsets {@code [batch, 2 * k * k, h, w]}
     * @return {@code [batch, filters, h, w]}
     */
    public Var sample(Var input, Var offsets) {
        int b = input.shape().dim(0);
        int h = input.shape().dim(2);
        int w = input.shape().dim(3);
        int hw = h * w;
        int taps = kernel * kernel;
        int n = taps * hw;
        int p = kernel / 2;
        Var off = offsets.reshape(b, taps, 2, hw);
        Var dy = off.slice(2, 0, 1).reshape(b, n);
        Var dx = off.slice(2, 1, 2).reshape(b, n);
        float[] oy = dy.value().toArray();
        float[] ox = dx.value().toArray();
        float[] baseY = new float[b * n];
        float[] baseX = new float[b * n];
        int[][][] idx = new int[4][b][n];
        float[][] valid = new float[4][b * n];
        for (int s = 0; s < b; s++) {
            for (int t = 0; t < taps; t++) {
                int ky = t / kernel - p;
                int kx = t % kernel - p;
                for (int q = 0; q < hw; q++) {
                    int e = t * hw + q;
                    int f = s * n + e;
                    float by = q / w + ky;
                    float bx = q % w + kx;
                    double py = by + oy[f];
                    double px = bx + ox[f];
                    int y0 = (int) Math.floor(py);
                    int x0 = (int) Math.floor(px);
                    baseY[f] = by - y0;   // fy = py - y0 = (by - y0) + dy
                    baseX[f] = bx - x0;
                    for (int c = 0; c < 4; c++) {
                        int yy = y0 + (c >> 1);
                        int xx = x0 + (c & 1);
                        boolean in = yy >= 0 && yy < h && xx >= 0 && xx < w;
                        idx[c][s][e] = in ? yy * w + xx : 0;
                        valid[c][f] = in ? 1f : 0f;
                    }
                }
            }
        }
        var device = input.value().device();
        Var fy = dy.add(Tensor.of(Shape.of(b, n), baseY, device)).reshape(b, 1, n);
        Var fx = dx.add(Tensor.of(Shape.of(b, n), baseX, device)).reshape(b, 1, n);
        Var gy = fy.neg().add(1f);
        Var gx = fx.neg().add(1f);
        Var[] cornerWeights = {gy.mul(gx), gy.mul(fx), fy.mul(gx), fy.mul(fx)};
        Var rows = input.reshape(b * channels, hw);
        Var sampled = null;
        for (int c = 0; c < 4; c++) {
            int[][] rowIdx = new int[b * channels][];
            for (int s = 0; s < b; s++) {
                for (int ch = 0; ch < channels; ch++) {
                    rowIdx[s * channels + ch] = idx[c][s];
                }
            }
            Var v = TechniqueOps.takeAlong(rows, rowIdx).reshape(b, channels, n);
            Var term = v.mul(cornerWeights[c].mul(Tensor.of(Shape.of(b, 1, n), valid[c], device)));
            sampled = sampled == null ? term : sampled.add(term);
        }
        Var cols = sampled.reshape(b, channels * taps, hw).permute(0, 2, 1);      // [b, hw, c * taps]
        return cols.matmul(weight).add(bias).permute(0, 2, 1).reshape(b, filters, h, w);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(filters, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias, offsetWeight, offsetBias);
    }

    @Override
    public String toString() {
        return "DeformableConv2d(" + channels + " -> " + filters + ", " + kernel + "x" + kernel + ")";
    }
}
