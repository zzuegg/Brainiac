package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/** The inference-only layers the diffusion models are built from, on plain tensors. */
final class Nn {

    private Nn() {
    }

    /**
     * The most values one convolution unrolls at once ({@code im2col}); larger images are
     * convolved in bands of rows so that a 1024 x 1024 VAE decode stays within GPU memory.
     */
    static final long MAX_COLUMNS = Long.getLong("brainiac.diffusion.maxColumns", 1L << 27);

    /** The most attention scores computed at once where the backend has no fused attention. */
    static final long MAX_SCORES = Long.getLong("brainiac.diffusion.maxScores", 1L << 26);

    /** {@code x W + b} with {@code W} as {@code [in, out]}. */
    record Linear(Tensor weight, Tensor bias) {

        Tensor apply(Tensor x) {
            Tensor y = x.matmul(weight);
            return bias == null ? y : y.add(bias);
        }

        int out() {
            return weight.shape().dim(1);
        }
    }

    /**
     * A square 2-d convolution on {@code [batch, channels, height, width]}, as unrolled windows
     * times a matrix, with {@code pad} zeros before and {@code padEnd} after the image.
     */
    record Conv2d(Tensor weight, Tensor bias, int in, int out, int kernel, int stride, int pad, int padEnd) {

        Tensor apply(Tensor x) {
            Shape s = x.shape();
            int b = s.dim(0);
            int h = s.dim(2);
            int w = s.dim(3);
            if (s.dim(1) != in) {
                throw new IllegalArgumentException("convolution expects " + in + " channels, got " + s);
            }
            Tensor y;
            if (kernel == 1 && stride == 1 && pad == 0 && padEnd == 0) {
                y = x.reshape(b, in, h * w).transpose().matmul(weight).transpose().reshape(b, out, h, w);
            } else {
                int ho = (h + pad + padEnd - kernel) / stride + 1;
                int wo = (w + pad + padEnd - kernel) / stride + 1;
                long perRow = (long) b * wo * in * kernel * kernel;
                int band = (int) Math.max(1, Math.min(ho, MAX_COLUMNS / perRow));
                if (band >= ho) {
                    y = rows(x, 0, ho, wo);
                } else {
                    List<Tensor> parts = new ArrayList<>();
                    for (int y0 = 0; y0 < ho; y0 += band) {
                        Tensor part = rows(x, y0, Math.min(ho, y0 + band), wo);
                        Tensor.evaluate(List.of(part));
                        parts.add(part);
                    }
                    y = Tensor.concat(2, parts.toArray(Tensor[]::new));
                }
            }
            return bias == null ? y : y.add(bias);
        }

        /** Output rows {@code [y0, y1)}. */
        private Tensor rows(Tensor x, int y0, int y1, int wo) {
            Shape s = x.shape();
            int b = s.dim(0);
            int h = s.dim(2);
            int w = s.dim(3);
            int first = y0 * stride - pad;
            int last = (y1 - 1) * stride - pad + kernel;
            int a = Math.max(0, first);
            int e = Math.min(h, last);
            Tensor part = a == 0 && e == h ? x : x.slice(2, a, e);
            ConvGeometry g = new ConvGeometry(b, in, new int[] {e - a, w}, new int[] {kernel, kernel},
                    new int[] {stride, stride}, new int[] {a - first, pad}, new int[] {1, 1},
                    new int[] {y1 - y0, wo});
            Tensor cols = part.im2col(g, 0f);
            return cols.matmul(weight).reshape(b, (y1 - y0) * wo, out).transpose().reshape(b, out, y1 - y0, wo);
        }
    }

    /** Group normalization over {@code [batch, channels, ...]}, computed in float32. */
    record GroupNorm(Tensor weight, Tensor bias, int groups, float eps) {

        Tensor apply(Tensor x) {
            Shape s = x.shape();
            int b = s.dim(0);
            int c = s.dim(1);
            long rest = s.size() / ((long) b * c);
            Tensor f = x.to(DType.FLOAT32);
            Tensor r = f.reshape(b, groups, Math.toIntExact(s.size() / ((long) b * groups)));
            Tensor d = r.sub(r.mean(2, true));
            Tensor n = d.div(d.square().mean(2, true).add(eps).sqrt());
            Tensor y = n.reshape(b, c, Math.toIntExact(rest)).mul(weight.reshape(c, 1)).add(bias.reshape(c, 1));
            return y.reshape(s).to(x.dtype());
        }
    }

    /** Layer normalization over the last axis, computed in float32. */
    record LayerNorm(Tensor weight, Tensor bias, float eps) {

        Tensor apply(Tensor x) {
            Tensor f = x.to(DType.FLOAT32);
            Tensor d = f.sub(f.mean(-1, true));
            Tensor n = d.div(d.square().mean(-1, true).add(eps).sqrt());
            return n.mul(weight).add(bias).to(x.dtype());
        }
    }

    static Tensor silu(Tensor x) {
        return x.activation(ActivationOp.SILU, 0);
    }

    static Tensor gelu(Tensor x) {
        return x.activation(ActivationOp.GELU, 0);
    }

    /** CLIP's {@code x * sigmoid(1.702 x)}. */
    static Tensor quickGelu(Tensor x) {
        return x.mul(x.mul(1.702f).sigmoid());
    }

    /** Nearest-neighbour upsampling by 2 of {@code [batch, channels, height, width]}. */
    static Tensor upsample2x(Tensor x) {
        Shape s = x.shape();
        int b = s.dim(0);
        int c = s.dim(1);
        int h = s.dim(2);
        int w = s.dim(3);
        return x.reshape(b, c, h, 1, w, 1).expand(Shape.of(b, c, h, 2, w, 2)).reshape(b, c, 2 * h, 2 * w);
    }

    /** {@code [batch, channels, h, w]} as tokens {@code [batch, h * w, channels]}. */
    static Tensor tokens(Tensor x) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1), s.dim(2) * s.dim(3)).transpose();
    }

    /** Tokens {@code [batch, h * w, channels]} back as {@code [batch, channels, h, w]}. */
    static Tensor image(Tensor x, int h, int w) {
        Shape s = x.shape();
        return x.transpose().reshape(s.dim(0), s.dim(2), h, w);
    }

    /**
     * Multi-head scaled dot-product attention of {@code [batch, tq, channels]} queries over
     * {@code [batch, tk, channels]} keys and values.
     */
    static Tensor attention(Tensor q, Tensor k, Tensor v, int heads, boolean causal) {
        Shape qs = q.shape();
        int b = qs.dim(0);
        int tq = qs.dim(1);
        int tk = k.shape().dim(1);
        int c = qs.dim(2);
        int d = c / heads;
        Tensor qh = q.reshape(b, tq, heads, d).permute(0, 2, 1, 3);
        Tensor kh = k.reshape(b, tk, heads, d).permute(0, 2, 1, 3);
        Tensor vh = v.reshape(b, tk, heads, d).permute(0, 2, 1, 3);
        float scale = (float) (1 / Math.sqrt(d));
        Tensor o;
        if (!causal || tq == tk) {
            if (Tensor.hasAttention(qh, kh, vh, false)) {
                o = Tensor.attention(qh, kh, vh, scale, causal)[0];
            } else {
                o = slowAttention(qh, kh, vh, scale, causal);
            }
        } else {
            throw new IllegalArgumentException("causal attention needs as many queries as keys");
        }
        return o.permute(0, 2, 1, 3).reshape(b, tq, c);
    }

    /** Attention as two products and a softmax in float32, a band of queries at a time. */
    private static Tensor slowAttention(Tensor q, Tensor k, Tensor v, float scale, boolean causal) {
        DType type = q.dtype();
        Shape s = q.shape();
        int tq = s.dim(2);
        int tk = k.shape().dim(2);
        long perRow = (long) s.dim(0) * s.dim(1) * tk;
        int band = (int) Math.max(1, Math.min(tq, MAX_SCORES / perRow));
        Tensor kt = k.to(DType.FLOAT32).transpose();
        Tensor vf = v.to(DType.FLOAT32);
        Tensor qf = q.to(DType.FLOAT32);
        List<Tensor> parts = new ArrayList<>();
        for (int r0 = 0; r0 < tq; r0 += band) {
            int r1 = Math.min(tq, r0 + band);
            Tensor qb = band >= tq ? qf : qf.slice(2, r0, r1);
            Tensor scores = qb.matmul(kt).mul(scale);
            if (causal) {
                float[] mask = new float[(r1 - r0) * tk];
                for (int i = r0; i < r1; i++) {
                    for (int j = i + 1; j < tk; j++) {
                        mask[(i - r0) * tk + j] = Float.NEGATIVE_INFINITY;
                    }
                }
                scores = scores.add(Tensor.of(Shape.of(r1 - r0, tk), mask, q.device()));
            }
            Tensor e = scores.sub(scores.max(-1, true)).exp();
            Tensor p = e.div(e.sum(-1, true));
            Tensor part = p.matmul(vf);
            if (band < tq) {
                Tensor.evaluate(List.of(part));
            }
            parts.add(part);
        }
        Tensor out = parts.size() == 1 ? parts.get(0) : Tensor.concat(2, parts.toArray(Tensor[]::new));
        return out.to(type);
    }

    /**
     * Runs a step of a forward pass and frees every tensor it made except its result, the
     * weights and the given inputs, so that memory stays bounded over a deep network.
     */
    static Tensor scoped(Collection<Tensor> weights, Supplier<Tensor> step, Tensor... inputs) {
        Tensor[] out = new Tensor[1];
        List<Tensor> made = Tensor.created(() -> out[0] = step.get());
        Tensor.evaluate(List.of(out[0]));
        Set<Tensor> keepSet = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        keepSet.addAll(weights);
        keepSet.add(out[0]);
        for (Tensor t : inputs) {
            if (t != null) {
                keepSet.add(t);
            }
        }
        Tensor.release(made, keepSet);
        return out[0];
    }
}
