package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** The small layers Chatterbox's parts share, read from PyTorch module weights. */
final class ChatterboxNets {

    private ChatterboxNets() {
    }

    /** A linear layer, its weight transposed to {@code [in, out]}. */
    record Lin(Var w, Var b) {
        Var apply(Var x) {
            Var y = x.matmul(w);
            return b == null ? y : y.add(b);
        }

        int out() {
            return w.shape().dim(1);
        }
    }

    /** A convolution (weight normalization folded in). */
    record Conv(Var w, Var b) {
        int kernel() {
            return w.shape().dim(w.shape().rank() - 1);
        }

        int out() {
            return w.shape().dim(0);
        }

        /** A 1-D convolution with zero padding. */
        Var apply(Var x, int stride, int padBefore, int padAfter, int dilation) {
            int groups = x.shape().dim(1) / w.shape().dim(1);
            return Functional.conv(x, w, b, new int[] {stride}, new int[] {padBefore}, new int[] {padAfter},
                    new int[] {dilation}, groups);
        }

        /** A 1-D convolution padded to keep the length ({@code (k - 1) / 2 * dilation} each side). */
        Var same(Var x, int dilation) {
            int p = (kernel() - 1) / 2 * dilation;
            return apply(x, 1, p, p, dilation);
        }

        /** A causal 1-D convolution: {@code k - 1} zeros before. */
        Var causal(Var x) {
            return apply(x, 1, kernel() - 1, 0, 1);
        }

        /** A 2-D convolution with the same padding on both sides of each axis. */
        Var apply2d(Var x, int strideH, int strideW, int pad) {
            return Functional.conv(x, w, b, new int[] {strideH, strideW}, new int[] {pad, pad},
                    new int[] {pad, pad}, new int[] {1, 1}, 1);
        }
    }

    /** A layer normalization's scale and shift. */
    record Norm(Var w, Var b) {
    }

    /** An inference batch normalization folded into a per-channel scale and shift. */
    record BatchNorm(float[] scale, float[] shift) {
        /** Normalizes axis 1 of {@code x}. */
        Var apply(Var x) {
            int rank = x.shape().rank();
            int c = scale.length;
            int[] dims = new int[rank - 1];
            java.util.Arrays.fill(dims, 1);
            dims[0] = c;
            Shape s = Shape.of(dims);
            var device = x.value().device();
            Tensor a = Tensor.of(s, scale, device).to(x.dtype());
            Tensor m = Tensor.of(s, shift, device).to(x.dtype());
            return x.mul(a).add(m);
        }
    }

    static Lin lin(Weights w, String name) {
        return new Lin(w.linear(name + ".weight"), w.optional(name + ".bias"));
    }

    static Conv conv(Weights w, String name) {
        return new Conv(w.conv(name), w.optional(name + ".bias"));
    }

    static Norm norm(Weights w, String name) {
        return new Norm(w.read(name + ".weight"), w.read(name + ".bias"));
    }

    static BatchNorm batchNorm(Weights w, String name) {
        float[] mean = w.floats(name + ".running_mean");
        float[] var = w.floats(name + ".running_var");
        float[] g = w.contains(name + ".weight") ? w.floats(name + ".weight") : null;
        float[] b = w.contains(name + ".bias") ? w.floats(name + ".bias") : null;
        float[] scale = new float[mean.length];
        float[] shift = new float[mean.length];
        for (int i = 0; i < mean.length; i++) {
            double s = 1 / Math.sqrt(var[i] + 1e-5);
            if (g != null) {
                s *= g[i];
            }
            scale[i] = (float) s;
            shift[i] = (float) ((b == null ? 0 : b[i]) - mean[i] * s);
        }
        return new BatchNorm(scale, shift);
    }

    /** Layer normalization over the last axis. */
    static Var layerNorm(Var x, Norm n, float eps) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        Var y = centred.div(centred.square().mean(last, true).add(eps).sqrt());
        return n == null ? y : y.mul(n.w()).add(n.b());
    }

    /** Layer normalization over the channels of {@code [batch, channels, time]}. */
    static Var channelNorm(Var x, Norm n, float eps) {
        return layerNorm(x.permute(0, 2, 1), n, eps).permute(0, 2, 1);
    }

    /** {@code [batch, steps, heads * d]} as {@code [batch, heads, steps, d]}. */
    static Var heads(Var x, int heads) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1), heads, s.dim(2) / heads).permute(0, 2, 1, 3);
    }

    /** {@code [batch, heads, steps, d]} back to {@code [batch, steps, heads * d]}. */
    static Var merge(Var x) {
        Shape s = x.shape();
        return x.permute(0, 2, 1, 3).reshape(s.dim(0), s.dim(2), s.dim(1) * s.dim(3));
    }

    /** Plain multi-head attention, every key visible. */
    static Var attend(Var q, Var k, Var v, int heads) {
        int dk = q.shape().dim(2) / heads;
        return merge(Functional.attention(ChatterboxNets.heads(q, heads), ChatterboxNets.heads(k, heads),
                ChatterboxNets.heads(v, heads), false, (float) (1 / Math.sqrt(dk))));
    }

    /** Rows of an embedding table. */
    static Var rows(Var table, int... ids) {
        float[] idx = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            idx[i] = ids[i];
        }
        return table.gather(0, Tensor.of(Shape.of(ids.length), idx, table.value().device()));
    }

    /** Rows {@code 0..n-1} of an embedding table. */
    static Var firstRows(Var table, int n) {
        return table.slice(0, 0, n);
    }
}
