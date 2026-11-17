package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The math behind the image layers as plain functions on {@link Var}s, for building
 * custom layers or experimenting without a model. Images have the layout
 * {@code [batch, channels, d1, d2, ...]} with any number of spatial dimensions.
 *
 * <p>Everything here is differentiable.
 */
public final class Functional {

    private Functional() {
    }

    // ---------------------------------------------------------------- padding

    /**
     * Pads (or, with negative amounts, crops) the spatial dimensions of images.
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param before the cells to add before each spatial dimension
     * @param after the cells to add after each spatial dimension
     * @param mode how new cells are filled
     * @param value the fill value for {@link PadMode#CONSTANT}
     * @return the padded images
     */
    public static Var pad(Var x, int[] before, int[] after, PadMode mode, float value) {
        int spatial = x.shape().rank() - 2;
        if (before.length != spatial || after.length != spatial) {
            throw new IllegalArgumentException("pad needs " + spatial + " amounts per side for " + x.shape());
        }
        Var out = x;
        for (int d = 0; d < spatial; d++) {
            out = padAxis(out, d + 2, before[d], after[d], mode, value);
        }
        return out;
    }

    /**
     * Pads one axis of any tensor.
     *
     * @param x the tensor
     * @param axis the axis
     * @param before cells before (negative crops)
     * @param after cells after (negative crops)
     * @param mode how new cells are filled
     * @param value the fill value for {@link PadMode#CONSTANT}
     * @return the padded tensor
     */
    public static Var padAxis(Var x, int axis, int before, int after, PadMode mode, float value) {
        int length = x.shape().dim(axis);
        if (before < 0 || after < 0) {
            int start = Math.max(-before, 0);
            int end = length - Math.max(-after, 0);
            if (end <= start) {
                throw new IllegalArgumentException("cropping removes all of axis " + axis + " of " + x.shape());
            }
            x = x.slice(axis, start, end);
            return padAxis(x, axis, Math.max(before, 0), Math.max(after, 0), mode, value);
        }
        if (before == 0 && after == 0) {
            return x;
        }
        Device device = x.value().device();
        if (mode == PadMode.CONSTANT) {
            int[] dims = x.shape().toArray();
            Var[] parts = new Var[3];
            int n = 0;
            if (before > 0) {
                dims[axis] = before;
                parts[n++] = Var.of(Tensor.full(Shape.of(dims), value, device));
            }
            parts[n++] = x;
            if (after > 0) {
                dims[axis] = after;
                parts[n++] = Var.of(Tensor.full(Shape.of(dims), value, device));
            }
            return Var.concat(axis, Arrays.copyOf(parts, n));
        }
        float[] idx = new float[before + length + after];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = mode.source(i - before, length);
        }
        return x.gather(axis, Tensor.of(Shape.of(idx.length), idx, device));
    }

    // ---------------------------------------------------------------- convolution

    /**
     * Convolves images with filters (cross-correlation, as in every deep learning
     * library).
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param weight the filters, {@code [filters, channels / groups, kernel...]}
     * @param bias one value per filter, or {@code null}
     * @param stride the step per spatial dimension
     * @param padBefore zero padding before each spatial dimension
     * @param padAfter zero padding after each spatial dimension
     * @param dilation the spacing of kernel cells per spatial dimension
     * @param groups the number of channel groups; each filter sees only its group's channels
     * @return {@code [batch, filters, out...]}
     */
    public static Var conv(Var x, Var weight, Var bias, int[] stride, int[] padBefore, int[] padAfter,
            int[] dilation, int groups) {
        Shape xs = x.shape();
        Shape ws = weight.shape();
        int r = xs.rank() - 2;
        if (r < 1 || ws.rank() != r + 2) {
            throw new IllegalArgumentException("conv needs images [batch, channels, spatial...] and filters"
                    + " [filters, channels / groups, kernel...] of the same spatial rank, got " + xs + " and " + ws);
        }
        int n = xs.dim(0);
        int c = xs.dim(1);
        int f = ws.dim(0);
        if (groups < 1 || c % groups != 0 || f % groups != 0 || ws.dim(1) != c / groups) {
            throw new IllegalArgumentException("conv with " + groups + " groups cannot take " + c + " channels and "
                    + f + " filters of shape " + ws);
        }
        int[] kernel = Arrays.copyOfRange(ws.toArray(), 2, r + 2);
        ConvGeometry g = ConvGeometry.sliding(n, c, Arrays.copyOfRange(xs.toArray(), 2, r + 2), kernel, stride,
                padBefore, padAfter, dilation, false);
        int p = g.positionCount();
        int k = g.kernelSize();
        int cg = c / groups;
        int fg = f / groups;
        Var cols = x.im2col(g, 0f);
        Var y;
        if (groups == 1) {
            y = cols.matmul(weight.reshape(f, c * k).transpose());
        } else {
            Var grouped = cols.reshape(n * p, groups, cg * k).permute(1, 0, 2);
            Var w = weight.reshape(groups, fg, cg * k).permute(0, 2, 1);
            y = grouped.matmul(w).permute(1, 0, 2).reshape(n * p, f);
        }
        int[] outDims = imageDims(n, f, g.positions());
        y = y.reshape(n, p, f).permute(0, 2, 1).reshape(outDims);
        return bias == null ? y : y.add(bias.reshape(channelShape(f, r)));
    }

    /**
     * Transposed convolution ("deconvolution"): the gradient of {@link #conv} with
     * respect to its input, used to upsample. Output size per dimension is
     * {@code (in - 1) * stride - padBefore - padAfter + dilation * (kernel - 1) + 1 + outputPadding}.
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param weight the filters, {@code [channels, filters / groups, kernel...]}
     * @param bias one value per output filter, or {@code null}
     * @param stride the step per spatial dimension
     * @param padBefore cells removed before each output dimension
     * @param padAfter cells removed after each output dimension
     * @param outputPadding cells added after each output dimension, to pick among the sizes a stride allows
     * @param dilation the spacing of kernel cells
     * @param groups the number of channel groups
     * @return {@code [batch, filters, out...]}
     */
    public static Var convTranspose(Var x, Var weight, Var bias, int[] stride, int[] padBefore, int[] padAfter,
            int[] outputPadding, int[] dilation, int groups) {
        Shape xs = x.shape();
        Shape ws = weight.shape();
        int r = xs.rank() - 2;
        if (r < 1 || ws.rank() != r + 2) {
            throw new IllegalArgumentException("transposed conv needs images [batch, channels, spatial...] and"
                    + " filters [channels, filters / groups, kernel...], got " + xs + " and " + ws);
        }
        int n = xs.dim(0);
        int c = xs.dim(1);
        if (groups < 1 || c % groups != 0 || ws.dim(0) != c) {
            throw new IllegalArgumentException("transposed conv with " + groups + " groups cannot take " + c
                    + " channels and filters of shape " + ws);
        }
        int fg = ws.dim(1);
        int f = fg * groups;
        int cg = c / groups;
        int[] kernel = Arrays.copyOfRange(ws.toArray(), 2, r + 2);
        int[] in = Arrays.copyOfRange(xs.toArray(), 2, r + 2);
        int[] out = new int[r];
        for (int i = 0; i < r; i++) {
            out[i] = (in[i] - 1) * stride[i] - padBefore[i] - padAfter[i] + dilation[i] * (kernel[i] - 1) + 1
                    + outputPadding[i];
            if (out[i] < 1) {
                throw new IllegalArgumentException("transposed conv output would be empty in dimension " + i);
            }
        }
        ConvGeometry g = new ConvGeometry(n, f, out, kernel, stride, padBefore, dilation, in);
        int p = g.positionCount();
        int k = g.kernelSize();
        Var rows = x.reshape(n, c, p).permute(0, 2, 1).reshape(n * p, c);
        Var cols;
        if (groups == 1) {
            cols = rows.matmul(weight.reshape(c, f * k));
        } else {
            Var grouped = rows.reshape(n * p, groups, cg).permute(1, 0, 2);
            cols = grouped.matmul(weight.reshape(groups, cg, fg * k)).permute(1, 0, 2).reshape(n * p, f * k);
        }
        Var y = cols.col2im(g);
        return bias == null ? y : y.add(bias.reshape(channelShape(f, r)));
    }

    // ---------------------------------------------------------------- pooling

    /** What a pooling window computes. */
    public enum PoolKind {
        /** the largest value; the gradient goes to the first largest */
        MAX,
        /** the mean */
        AVG,
        /** the power mean {@code (sum |x|^p)^(1/p)} */
        LP,
        /** the smallest value */
        MIN,
        /** the sum */
        SUM
    }

    /**
     * Pools sliding windows of each channel.
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param kind what each window computes
     * @param kernel the window size per spatial dimension
     * @param stride the step per spatial dimension
     * @param padBefore padding before each spatial dimension (ignored by max and min)
     * @param padAfter padding after each spatial dimension
     * @param dilation the spacing of window cells
     * @param ceil whether a partial window at the end still produces an output
     * @param countPadding for averages: whether padding cells count in the divisor
     * @param p the power for {@link PoolKind#LP}
     * @return {@code [batch, channels, out...]}
     */
    public static Var pool(Var x, PoolKind kind, int[] kernel, int[] stride, int[] padBefore, int[] padAfter,
            int[] dilation, boolean ceil, boolean countPadding, float p) {
        Shape xs = x.shape();
        int r = xs.rank() - 2;
        if (r < 1 || kernel.length != r) {
            throw new IllegalArgumentException("pooling with a " + kernel.length + "-d window needs images [batch,"
                    + " channels, spatial...] with " + kernel.length + " spatial dimensions, got " + xs);
        }
        int n = xs.dim(0);
        int c = xs.dim(1);
        int[] image = Arrays.copyOfRange(xs.toArray(), 2, r + 2);
        ConvGeometry g = ConvGeometry.sliding(n * c, 1, image, kernel, stride, padBefore, padAfter, dilation, ceil);
        int[] planes = new int[r + 2];
        planes[0] = n * c;
        planes[1] = 1;
        System.arraycopy(image, 0, planes, 2, r);
        Var flat = x.reshape(planes);
        Var y = switch (kind) {
            case MAX -> windowMax(flat.im2col(g, Float.NEGATIVE_INFINITY));
            case MIN -> windowMax(flat.neg().im2col(g, Float.NEGATIVE_INFINITY)).neg();
            case SUM -> flat.im2col(g, 0f).sum(1, false);
            case LP -> flat.abs().pow(p).im2col(g, 0f).sum(1, false).add(1e-12f).pow(1 / p);
            case AVG -> {
                Var sums = flat.im2col(g, 0f).sum(1, false);
                if (countPadding) {
                    // windows past the padded image (ceil mode) still divide by their real cells
                    yield sums.div(windowCounts(g, x.value().device(), padBefore, padAfter));
                }
                yield sums.div(windowCounts(g, x.value().device(), null, null));
            }
        };
        return y.reshape(imageDims(n, c, g.positions()));
    }

    /** The number of cells each window covers, counting padding inside the given bounds. */
    private static Tensor windowCounts(ConvGeometry g, Device device, int[] padBefore, int[] padAfter) {
        int[] image = g.image();
        int r = image.length;
        if (padBefore == null) {
            Tensor ones = Tensor.ones(Shape.of(imageDims(1, 1, image))).to(device);
            return ones.im2col(g.with(1, 1), 0f).sum(1, false).reshape(g.positionCount()).expand(
                    Shape.of(g.batch(), g.positionCount())).reshape(g.batch() * g.positionCount());
        }
        int[] padded = new int[r];
        int[] noPad = new int[r];
        for (int i = 0; i < r; i++) {
            padded[i] = image[i] + padBefore[i] + padAfter[i];
        }
        ConvGeometry inPadded = new ConvGeometry(1, 1, padded, g.kernel(), g.stride(), noPad, g.dilation(),
                g.positions());
        Tensor ones = Tensor.ones(Shape.of(imageDims(1, 1, padded))).to(device);
        return ones.im2col(inPadded, 0f).sum(1, false).reshape(g.positionCount())
                .expand(Shape.of(g.batch(), g.positionCount())).reshape(g.batch() * g.positionCount());
    }

    private static final Op WINDOW_MAX = new Op() {
        @Override
        public Tensor forward(Tensor... in) {
            return in[0].max(1, false);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            Tensor cols = in[0];
            int rows = cols.shape().dim(0);
            int k = cols.shape().dim(1);
            Tensor first = cols.argmax(1).reshape(rows, 1);
            Tensor hot = first.equalTo(Tensor.arange(k).to(cols.device()).reshape(1, k));
            return new Tensor[] {hot.mul(grad.reshape(rows, 1))};
        }

        @Override
        public String name() {
            return "max";
        }
    };

    /**
     * The maximum of each row, with the whole gradient going to the first largest value
     * (so ties, such as several zeros after a ReLU, do not multiply it).
     *
     * @param rows a matrix
     * @return the row maxima
     */
    public static Var windowMax(Var rows) {
        return Var.apply(rows.requiresGrad() ? new WindowMax() : WINDOW_MAX, rows);
    }

    /**
     * {@link #WINDOW_MAX} that finds the first largest value along with the maximum: on devices
     * that defer reductions, both are then computed by one kernel instead of two.
     */
    private static final class WindowMax implements Op {

        private Tensor first;

        @Override
        public Tensor forward(Tensor... in) {
            first = in[0].argmax(1);
            return in[0].max(1, false);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            Tensor cols = in[0];
            int rows = cols.shape().dim(0);
            int k = cols.shape().dim(1);
            Tensor hot = first.reshape(rows, 1).equalTo(Tensor.arange(k).to(cols.device()).reshape(1, k));
            return new Tensor[] {hot.mul(grad.reshape(rows, 1))};
        }

        @Override
        public String name() {
            return "max";
        }
    }

    /**
     * Pools over all spatial positions.
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param kind {@link PoolKind#AVG}, {@link PoolKind#MAX}, {@link PoolKind#MIN} or {@link PoolKind#SUM}
     * @param keepDims whether to keep the spatial dimensions as size 1
     * @return {@code [batch, channels]}, or {@code [batch, channels, 1, ...]}
     */
    public static Var globalPool(Var x, PoolKind kind, boolean keepDims) {
        Shape xs = x.shape();
        int n = xs.dim(0);
        int c = xs.dim(1);
        int p = Math.toIntExact(xs.size() / ((long) n * c));
        Var rows = x.reshape(n * c, p);
        Var y = switch (kind) {
            case AVG -> rows.mean(1, false);
            case SUM -> rows.sum(1, false);
            case MAX -> windowMax(rows);
            case MIN -> windowMax(rows.neg()).neg();
            case LP -> throw new IllegalArgumentException("use pool(...) for LP pooling");
        };
        if (!keepDims) {
            return y.reshape(n, c);
        }
        int[] dims = new int[xs.rank()];
        Arrays.fill(dims, 1);
        dims[0] = n;
        dims[1] = c;
        return y.reshape(dims);
    }

    /**
     * Pools each channel to a fixed output size, whatever the input size: output cell
     * {@code i} of a dimension covers input cells {@code floor(i * in / out)} to
     * {@code ceil((i + 1) * in / out)}.
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param kind {@link PoolKind#AVG}, {@link PoolKind#MAX}, {@link PoolKind#MIN} or {@link PoolKind#SUM}
     * @param size the output size per spatial dimension
     * @return {@code [batch, channels, size...]}
     */
    public static Var adaptivePool(Var x, PoolKind kind, int[] size) {
        int r = x.shape().rank() - 2;
        if (size.length != r) {
            throw new IllegalArgumentException("adaptive pooling needs " + r + " output sizes for " + x.shape());
        }
        Var out = x;
        for (int d = 0; d < r; d++) {
            int axis = d + 2;
            int in = out.shape().dim(axis);
            int target = size[d];
            if (target < 1) {
                throw new IllegalArgumentException("output size must be at least 1, got " + target);
            }
            if (in == target) {
                continue;
            }
            if (kind == PoolKind.AVG || kind == PoolKind.SUM) {
                // averages are separable: one matrix per dimension
                float[] m = new float[in * target];
                for (int o = 0; o < target; o++) {
                    int start = Math.floorDiv(o * in, target);
                    int end = Math.ceilDiv((o + 1) * in, target);
                    for (int i = start; i < end; i++) {
                        m[i * target + o] = kind == PoolKind.AVG ? 1f / (end - start) : 1f;
                    }
                }
                out = alongAxis(out, axis, Tensor.of(Shape.of(in, target), m, x.value().device()));
            } else {
                Var[] cells = new Var[target];
                for (int o = 0; o < target; o++) {
                    int start = Math.floorDiv(o * in, target);
                    int end = Math.ceilDiv((o + 1) * in, target);
                    Var window = moveAxisLast(out.slice(axis, start, end), axis);
                    Shape ws = window.shape();
                    int rows = Math.toIntExact(ws.size() / ws.dim(-1));
                    Var flat = window.reshape(rows, ws.dim(-1));
                    Var m = kind == PoolKind.MAX ? windowMax(flat) : windowMax(flat.neg()).neg();
                    int[] dims = out.shape().toArray();
                    dims[axis] = 1;
                    int[] permuted = moveLast(dims, axis);
                    cells[o] = moveAxisBack(m.reshape(permuted), axis);
                }
                out = Var.concat(axis, cells);
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- resampling

    /** How {@link #resize} computes new cells. */
    public enum Interpolation {
        /** copy the nearest cell */
        NEAREST,
        /** linear along each dimension: bilinear for images, trilinear for volumes */
        LINEAR,
        /** cubic convolution along each dimension (Keys, a = -0.75): bicubic for images */
        CUBIC
    }

    /**
     * Resizes the spatial dimensions of images.
     *
     * @param x the images, {@code [batch, channels, spatial...]}
     * @param size the new size per spatial dimension
     * @param interpolation how new cells are computed
     * @param alignCorners for linear and cubic: map the corner cells onto each other
     *     instead of the cell edges
     * @return the resized images
     */
    public static Var resize(Var x, int[] size, Interpolation interpolation, boolean alignCorners) {
        int r = x.shape().rank() - 2;
        if (size.length != r) {
            throw new IllegalArgumentException("resize needs " + r + " sizes for " + x.shape());
        }
        Var out = x;
        Device device = x.value().device();
        for (int d = 0; d < r; d++) {
            int axis = d + 2;
            int in = out.shape().dim(axis);
            int target = size[d];
            if (in == target && interpolation == Interpolation.NEAREST) {
                continue;
            }
            if (interpolation == Interpolation.NEAREST) {
                float[] idx = new float[target];
                for (int o = 0; o < target; o++) {
                    idx[o] = Math.min((int) Math.floor(o * (double) in / target), in - 1);
                }
                out = out.gather(axis, Tensor.of(Shape.of(target), idx, device));
            } else {
                out = alongAxis(out, axis, Tensor.of(Shape.of(in, target),
                        interpolationMatrix(in, target, interpolation, alignCorners), device));
            }
        }
        return out;
    }

    private static float[] interpolationMatrix(int in, int out, Interpolation kind, boolean alignCorners) {
        float[] m = new float[in * out];
        for (int o = 0; o < out; o++) {
            double src;
            if (alignCorners) {
                src = out == 1 ? 0 : o * (in - 1.0) / (out - 1.0);
            } else {
                src = Math.max((o + 0.5) * in / out - 0.5, kind == Interpolation.LINEAR ? 0 : -1e9);
            }
            int base = (int) Math.floor(src);
            double t = src - base;
            if (kind == Interpolation.LINEAR) {
                int i0 = Math.clamp(base, 0, in - 1);
                int i1 = Math.clamp(base + 1, 0, in - 1);
                m[i0 * out + o] += (float) (1 - t);
                m[i1 * out + o] += (float) t;
            } else {
                for (int j = -1; j <= 2; j++) {
                    int i = Math.clamp(base + j, 0, in - 1);
                    m[i * out + o] += (float) cubic(j - t);
                }
            }
        }
        return m;
    }

    private static double cubic(double x) {
        double a = -0.75;
        double ax = Math.abs(x);
        if (ax <= 1) {
            return ((a + 2) * ax - (a + 3)) * ax * ax + 1;
        }
        if (ax < 2) {
            return ((a * ax - 5 * a) * ax + 8 * a) * ax - 4 * a;
        }
        return 0;
    }

    /**
     * Moves values from channels into space: {@code [batch, channels * r^d, s...]} becomes
     * {@code [batch, channels, s * r...]} (sub-pixel convolution).
     *
     * @param x the images
     * @param factor the upscale factor {@code r} per spatial dimension
     * @return the rearranged images
     */
    public static Var pixelShuffle(Var x, int factor) {
        Shape xs = x.shape();
        int r = xs.rank() - 2;
        int n = xs.dim(0);
        int scale = (int) Math.pow(factor, r);
        if (factor < 1 || xs.dim(1) % scale != 0) {
            throw new IllegalArgumentException("pixel shuffle by " + factor + " needs channels divisible by " + scale
                    + ", got " + xs);
        }
        int c = xs.dim(1) / scale;
        // [n, c, f, f, ..., s1, s2, ...] -> [n, c, s1, f, s2, f, ...]
        int[] split = new int[2 + 2 * r];
        split[0] = n;
        split[1] = c;
        for (int i = 0; i < r; i++) {
            split[2 + i] = factor;
            split[2 + r + i] = xs.dim(2 + i);
        }
        int[] perm = new int[2 + 2 * r];
        perm[0] = 0;
        perm[1] = 1;
        for (int i = 0; i < r; i++) {
            perm[2 + 2 * i] = 2 + r + i;
            perm[3 + 2 * i] = 2 + i;
        }
        int[] out = new int[2 + r];
        out[0] = n;
        out[1] = c;
        for (int i = 0; i < r; i++) {
            out[2 + i] = xs.dim(2 + i) * factor;
        }
        return x.reshape(split).permute(perm).reshape(out);
    }

    /**
     * The inverse of {@link #pixelShuffle}: {@code [batch, channels, s * r...]} becomes
     * {@code [batch, channels * r^d, s...]}.
     *
     * @param x the images
     * @param factor the downscale factor per spatial dimension
     * @return the rearranged images
     */
    public static Var pixelUnshuffle(Var x, int factor) {
        Shape xs = x.shape();
        int r = xs.rank() - 2;
        int n = xs.dim(0);
        int c = xs.dim(1);
        int[] split = new int[2 + 2 * r];
        split[0] = n;
        split[1] = c;
        for (int i = 0; i < r; i++) {
            if (factor < 1 || xs.dim(2 + i) % factor != 0) {
                throw new IllegalArgumentException("pixel unshuffle by " + factor + " needs spatial sizes divisible"
                        + " by it, got " + xs);
            }
            split[2 + 2 * i] = xs.dim(2 + i) / factor;
            split[3 + 2 * i] = factor;
        }
        int[] perm = new int[2 + 2 * r];
        perm[0] = 0;
        perm[1] = 1;
        for (int i = 0; i < r; i++) {
            perm[2 + i] = 3 + 2 * i;
            perm[2 + r + i] = 2 + 2 * i;
        }
        int[] out = new int[2 + r];
        out[0] = n;
        out[1] = c * (int) Math.pow(factor, r);
        for (int i = 0; i < r; i++) {
            out[2 + i] = xs.dim(2 + i) / factor;
        }
        return x.reshape(split).permute(perm).reshape(out);
    }

    // ---------------------------------------------------------------- cross-entropy

    /**
     * The value of {@code forward} with the gradient of {@code backward}: the
     * straight-through estimator, for rounding, sampling or other steps without a useful
     * gradient.
     *
     * @param forward what the result is
     * @param backward what the gradient flows through; same shape
     * @return {@code backward + (forward - backward)}, the difference held constant
     */
    public static Var straightThrough(Var forward, Var backward) {
        return backward.add(forward.sub(backward).detach());
    }

    /**
     * Samples from the Gumbel-softmax (concrete) distribution over the last axis:
     * {@code softmax((logits + gumbel noise) / temperature)}. With {@code hard} the result
     * is the one-hot of the sample, with the soft sample's gradient (straight-through).
     *
     * @param logits {@code [..., classes]}
     * @param temperature positive; lower is closer to one-hot
     * @param hard whether to return one-hot samples
     * @param random the noise source
     * @return {@code [..., classes]}
     */
    public static Var gumbelSoftmax(Var logits, float temperature, boolean hard,
            java.util.random.RandomGenerator random) {
        if (!(temperature > 0)) {
            throw new IllegalArgumentException("temperature must be positive: " + temperature);
        }
        Tensor u = Tensor.uniform(logits.shape(), 1e-10f, 1, random).to(logits.dtype()).to(logits.value().device());
        Tensor gumbel = u.log().neg().log().neg();
        Var soft = Activation.SOFTMAX.forward(logits.add(gumbel).div(temperature));
        if (!hard) {
            return soft;
        }
        int classes = logits.shape().dim(-1);
        Tensor oneHot = soft.value().argmax(-1).reshape(appendOne(soft.shape()))
                .equalTo(Tensor.arange(classes).to(logits.value().device())).to(logits.dtype());
        return straightThrough(Var.of(oneHot), soft);
    }

    private static int[] appendOne(Shape s) {
        int[] dims = s.toArray();
        dims[dims.length - 1] = 1;
        return dims;
    }

    /**
     * Sinusoidal embeddings of continuous times or noise levels, one row per example, as
     * diffusion models feed their timestep (Ho et al.; the cos half first, as in DDPM code).
     *
     * @param times {@code [batch]}, such as diffusion steps or flow times scaled up
     * @param dim the embedding width, even
     * @param maxPeriod the longest wavelength, usually 10000
     * @return {@code [batch, dim]}
     */
    public static Tensor timestepEmbedding(Tensor times, int dim, float maxPeriod) {
        if (dim % 2 != 0) {
            throw new IllegalArgumentException("dim must be even, got " + dim);
        }
        int half = dim / 2;
        float[] freq = new float[half];
        for (int i = 0; i < half; i++) {
            freq[i] = (float) Math.exp(-Math.log(maxPeriod) * i / half);
        }
        int batch = times.shape().dim(0);
        Tensor args = times.reshape(batch, 1).mul(Tensor.of(Shape.of(1, half), freq, times.device()).to(times.dtype()));
        return Tensor.concat(1, args.cos(), args.sin());
    }

    /**
     * Feature-wise linear modulation (FiLM, Perez et al., 2018): {@code x * (1 + scale) + shift},
     * with one {@code scale} and {@code shift} per example and feature broadcast over the
     * positions in between, for conditioning on a class, a timestep or text.
     *
     * @param x {@code [batch, ..., features]}
     * @param scale {@code [batch, features]}
     * @param shift {@code [batch, features]}
     * @return modulated {@code x}
     */
    public static Var film(Var x, Var scale, Var shift) {
        int[] dims = new int[x.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = x.shape().dim(0);
        dims[dims.length - 1] = x.shape().dim(-1);
        return x.mul(scale.reshape(dims).add(1)).add(shift.reshape(dims));
    }

    /**
     * The modulation of an adaLN-Zero block (DiT, Peebles and Xie, 2023): layer normalization
     * without its own scale and offset, then {@link #film}. The conditioning projection that
     * makes {@code scale}, {@code shift} and the block's output gate starts at zero, so each
     * block starts as the identity; see {@link #gate}.
     *
     * @param x {@code [batch, ..., features]}
     * @param scale {@code [batch, features]}
     * @param shift {@code [batch, features]}
     * @param eps added to the variance
     * @return normalized and modulated {@code x}
     */
    public static Var adaLayerNorm(Var x, Var scale, Var shift, float eps) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        Var normed = centred.div(centred.square().mean(last, true).add(eps).sqrt());
        return film(normed, scale, shift);
    }

    /**
     * Scales a residual branch by a per-example, per-feature gate before it is added back,
     * as adaLN-Zero blocks do: {@code residual + gate * branch}.
     *
     * @param residual {@code [batch, ..., features]}
     * @param branch the block's output, same shape
     * @param gate {@code [batch, features]}
     * @return the sum
     */
    public static Var gate(Var residual, Var branch, Var gate) {
        int[] dims = new int[branch.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = branch.shape().dim(0);
        dims[dims.length - 1] = branch.shape().dim(-1);
        return residual.add(branch.mul(gate.reshape(dims)));
    }

    /**
     * The InfoNCE (contrastive) loss of paired embeddings, as in CLIP and SimCLR: row
     * {@code i} of {@code a} should be most similar to row {@code i} of {@code b} among all
     * rows of {@code b}, and the other way round. Rows are normalized first, and the other
     * rows of the batch serve as negatives.
     *
     * @param a embeddings {@code [batch, dim]}
     * @param b their partners {@code [batch, dim]}
     * @param temperature divides the cosine similarities; CLIP learns it, 0.07 is common
     * @return the loss averaged over both directions, a scalar
     */
    public static Var infoNce(Var a, Var b, float temperature) {
        if (!a.shape().equals(b.shape()) || a.shape().rank() != 2) {
            throw new IllegalArgumentException("infoNce needs two [batch, dim] inputs, got " + a.shape() + " and "
                    + b.shape());
        }
        Var na = a.div(a.square().sum(1, true).add(1e-12f).sqrt());
        Var nb = b.div(b.square().sum(1, true).add(1e-12f).sqrt());
        Var logits = na.matmul(nb.transpose()).div(temperature);
        int n = a.shape().dim(0);
        Tensor ids = Tensor.arange(n).to(a.value().device());
        return crossEntropy(logits, ids).add(crossEntropy(logits.transpose(), ids)).mul(0.5f);
    }

    /**
     * Sampled softmax cross-entropy (Jean et al., 2015) for very large output layers, such as
     * recommendation catalogues: the logits are computed only for the rows' own classes and
     * {@code negatives} classes drawn uniformly, shared by the batch. Every row's class is a
     * negative for the others. Uniform draws need no correction; the loss estimates full
     * cross-entropy and its gradient, not exactly.
     *
     * @param hidden {@code [rows, size]}
     * @param weight {@code [size, classes]}
     * @param ids {@code [rows]} class ids
     * @param negatives how many classes to draw
     * @param random the source of the draws
     * @return a scalar
     */
    public static Var sampledSoftmax(Var hidden, Var weight, Tensor ids, int negatives,
            java.util.random.RandomGenerator random) {
        int rows = hidden.shape().dim(0);
        int classes = weight.shape().dim(1);
        float[] target = ids.toArray();
        java.util.LinkedHashMap<Integer, Integer> column = new java.util.LinkedHashMap<>();
        for (float t : target) {
            column.putIfAbsent((int) t, column.size());
        }
        for (int i = 0; i < negatives && column.size() < classes; i++) {
            column.putIfAbsent(random.nextInt(classes), column.size());
        }
        float[] picked = new float[column.size()];
        column.forEach((c, at) -> picked[at] = c);
        float[] remapped = new float[rows];
        for (int r = 0; r < rows; r++) {
            remapped[r] = column.get((int) target[r]);
        }
        Device device = hidden.value().device();
        Var w = weight.transpose().gather(0, Tensor.of(Shape.of(picked.length), picked, device));
        Var logits = hidden.matmul(w.transpose());
        return crossEntropy(logits, Tensor.of(Shape.of(rows), remapped, device));
    }

    /**
     * Spectral normalization (Miyato et al., 2018): the weight divided by an estimate of its
     * largest singular value, so a GAN discriminator stays 1-Lipschitz. One power iteration
     * per call refines {@code u}, carried between steps.
     *
     * @param weight {@code [rows, cols]}, or more axes flattened after the first
     * @param u {@code [rows]} the running left singular vector estimate; updated in place
     * @return the normalized weight, same shape
     */
    public static Var spectralNorm(Var weight, float[] u) {
        int rows = weight.shape().dim(0);
        int cols = Math.toIntExact(weight.shape().size() / rows);
        float[] w = weight.value().to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
        if (u.length != rows) {
            throw new IllegalArgumentException("u needs " + rows + " values, got " + u.length);
        }
        if (norm(u) == 0) {
            java.util.Arrays.fill(u, 1);
        }
        float[] v = new float[cols];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                v[c] += w[r * cols + c] * u[r];
            }
        }
        float nv = norm(v);
        for (int c = 0; c < cols; c++) {
            v[c] /= Math.max(nv, 1e-12f);
        }
        java.util.Arrays.fill(u, 0);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                u[r] += w[r * cols + c] * v[c];
            }
        }
        float nu = norm(u);
        for (int r = 0; r < rows; r++) {
            u[r] /= Math.max(nu, 1e-12f);
        }
        // sigma = u^T W v, differentiable in W with u and v held fixed
        Device device = weight.value().device();
        Tensor uv = Tensor.of(Shape.of(rows, cols), outer(u, v), device).to(weight.value().dtype());
        Var sigma = weight.reshape(rows, cols).mul(uv).sum();
        return weight.div(sigma);
    }

    private static float norm(float[] x) {
        double s = 0;
        for (float f : x) {
            s += f * f;
        }
        return (float) Math.sqrt(s);
    }

    private static float[] outer(float[] u, float[] v) {
        float[] o = new float[u.length * v.length];
        for (int r = 0; r < u.length; r++) {
            for (int c = 0; c < v.length; c++) {
                o[r * v.length + c] = u[r] * v[c];
            }
        }
        return o;
    }

    /**
     * The R1 gradient penalty (Mescheder et al., 2018) for GAN discriminators:
     * {@code gamma / 2 * E[|grad_x D(x)|^2]} over real samples, differentiable in the
     * discriminator's weights (double backward).
     *
     * @param discriminator scores a batch {@code [batch, ...] -> [batch, 1]}
     * @param real the real samples
     * @param gamma the penalty weight
     * @return a scalar to add to the discriminator's loss
     */
    public static Var r1Penalty(java.util.function.Function<Var, Var> discriminator, Tensor real, float gamma) {
        Var x = Var.param(real);
        Var scores = discriminator.apply(x).sum();
        Var g = Var.gradients(scores, java.util.List.of(x), true).getFirst();
        int batch = real.shape().dim(0);
        return g.square().sum().mul(gamma / 2 / batch);
    }

    /**
     * Cross-entropy from logits: the mean over rows of {@code logsumexp(logits) - logits[id]},
     * skipping rows whose id is negative (padding, as PyTorch's {@code ignore_index}).
     *
     * @param logits {@code [rows, classes]}
     * @param ids {@code [rows]} class ids, negative to skip a row
     * @return a scalar
     */
    public static Var crossEntropy(Var logits, Tensor ids) {
        int rows = logits.shape().dim(0);
        int classes = logits.shape().dim(1);
        Tensor valid = ids.greater(Tensor.full(Shape.of(), -0.5f, ids.device()));
        float count = Math.max(1, valid.sum().item());
        Tensor target = ids.reshape(rows, 1).equalTo(Tensor.arange(classes).to(ids.device()));
        Var logp = Activation.LOG_SOFTMAX.forward(logits);
        return logp.mul(target).sum().mul(-1f / count);
    }

    /**
     * Cross-entropy of a linear output layer, {@code crossEntropy(hidden x weight, ids)}, that
     * never holds the {@code [rows, classes]} logits: it computes them {@code chunk} rows at a
     * time and again for the gradient. For language models, whose vocabularies make the
     * logits the largest array of a training step.
     *
     * @param hidden {@code [rows, size]}
     * @param weight {@code [size, classes]}
     * @param ids {@code [rows]} class ids, negative to skip a row
     * @param chunk the rows of logits computed at a time
     * @return a scalar, the mean over the rows that count
     */
    public static Var linearCrossEntropy(Var hidden, Var weight, Tensor ids, int chunk) {
        if (hidden.shape().rank() != 2 || weight.shape().rank() != 2 || hidden.shape().dim(1) != weight.shape().dim(0)
                || ids.size() != hidden.shape().dim(0)) {
            throw new IllegalArgumentException("linearCrossEntropy takes [rows, size], [size, classes] and [rows] ids, "
                    + "got " + hidden.shape() + ", " + weight.shape() + " and " + ids.shape());
        }
        if (chunk < 1) {
            throw new IllegalArgumentException("chunk < 1");
        }
        return Var.apply(new LinearCrossEntropy(ids.reshape(Math.toIntExact(ids.size())).to(DType.FLOAT32), chunk),
                hidden, weight);
    }

    /** Keeps each row's log-sum-exp between the passes; recomputes the logits. */
    private static final class LinearCrossEntropy implements Op {
        private final Tensor ids;
        private final int chunk;
        private Tensor lse;
        private float count;

        LinearCrossEntropy(Tensor ids, int chunk) {
            this.ids = ids;
            this.chunk = chunk;
        }

        @Override
        public Tensor forward(Tensor... in) {
            Tensor h = in[0];
            Tensor w = in[1];
            int rows = h.shape().dim(0);
            Tensor valid = ids.greater(Tensor.full(Shape.of(), -0.5f, ids.device()));
            count = Math.max(1, valid.sum().item());
            List<Tensor> lses = new ArrayList<>();
            List<Tensor> losses = new ArrayList<>();
            for (int start = 0; start < rows; start += chunk) {
                int from = start;
                int to = Math.min(rows, from + chunk);
                Tensor[] kept = new Tensor[2];
                List<Tensor> made = Tensor.created(() -> {
                    Tensor logits = h.slice(0, from, to).matmul(w).to(DType.FLOAT32);
                    Tensor max = logits.max(1, true);
                    Tensor l = logits.sub(max).exp().sum(1, true).log().add(max);   // [n, 1]
                    Tensor picked = logits.mul(target(from, to, logits.shape().dim(1))).sum(1, true);
                    kept[0] = l.reshape(to - from);
                    kept[1] = l.sub(picked).mul(valid.slice(0, from, to).reshape(to - from, 1)).sum();
                    Tensor.evaluate(List.of(kept[0], kept[1]));
                });
                Tensor.release(made, List.of(kept));
                lses.add(kept[0]);
                losses.add(kept[1]);
            }
            lse = lses.size() == 1 ? lses.getFirst() : Tensor.concat(0, lses.toArray(Tensor[]::new));
            Tensor total = losses.getFirst();
            for (int i = 1; i < losses.size(); i++) {
                total = total.add(losses.get(i));
            }
            return total.mul(1 / count).to(h.dtype());
        }

        private Tensor target(int from, int to, int classes) {
            return ids.slice(0, from, to).reshape(to - from, 1).equalTo(Tensor.arange(classes).to(ids.device()));
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
            Tensor h = in[0];
            Tensor w = in[1];
            int rows = h.shape().dim(0);
            Tensor scale = grad.to(DType.FLOAT32).reshape(1, 1).mul(1 / count);
            Tensor valid = ids.greater(Tensor.full(Shape.of(), -0.5f, ids.device()));
            Tensor wt = w.transpose();
            List<Tensor> dh = new ArrayList<>();
            Tensor dw = null;
            for (int start = 0; start < rows; start += chunk) {
                int from = start;
                int to = Math.min(rows, from + chunk);
                Tensor before = dw;
                Tensor[] kept = new Tensor[2];
                List<Tensor> made = Tensor.created(() -> {
                    Tensor hc = h.slice(0, from, to);
                    Tensor logits = hc.matmul(w).to(DType.FLOAT32);
                    Tensor p = logits.sub(lse.slice(0, from, to).reshape(to - from, 1)).exp();
                    // d loss / d logits = (softmax - one-hot) for the rows that count
                    Tensor g = p.sub(target(from, to, logits.shape().dim(1)))
                            .mul(valid.slice(0, from, to).reshape(to - from, 1)).mul(scale).to(h.dtype());
                    kept[0] = g.matmul(wt);
                    Tensor part = hc.transpose().matmul(g);
                    kept[1] = before == null ? part : before.add(part);
                    Tensor.evaluate(List.of(kept[0], kept[1]));
                });
                List<Tensor> dead = new ArrayList<>(made);
                if (before != null) {
                    dead.add(before);
                }
                Tensor.release(dead, List.of(kept));
                dh.add(kept[0]);
                dw = kept[1];
            }
            Tensor dhAll = dh.size() == 1 ? dh.getFirst() : Tensor.concat(0, dh.toArray(Tensor[]::new));
            return new Tensor[] {dhAll, dw};
        }

        @Override
        public String name() {
            return "linearCrossEntropy";
        }
    }

    // ---------------------------------------------------------------- attention

    /** Large enough to zero a softmax weight, small enough to stay finite. */
    static final float MASKED = -1e9f;

    /**
     * Scaled dot-product attention: {@code softmax(q k^T * scale + bias) v}, per batch and
     * head. Key/value heads may be fewer than query heads (grouped-query or multi-query
     * attention); each serves {@code heads / kvHeads} query heads.
     *
     * @param q queries {@code [batch, heads, queries, dk]}
     * @param k keys {@code [batch, kvHeads, keys, dk]}
     * @param v values {@code [batch, kvHeads, keys, dv]}
     * @param bias added to the scores before the softmax, broadcast to
     *     {@code [batch, heads, queries, keys]}, or {@code null}; use {@link #causalMask},
     *     {@link #keyMask} and {@link #alibi} or your own
     * @param scale the score scale, usually {@code 1 / sqrt(dk)}
     * @param dropout the share of attention weights dropped while training
     * @param random the random source for dropout, or {@code null} when not dropping
     * @return {@code [batch, heads, queries, dv]}
     */
    public static Var attention(Var q, Var k, Var v, Tensor bias, float scale, float dropout,
            java.util.random.RandomGenerator random) {
        return attention(q, k, v, bias, scale, dropout, random, null);
    }

    /**
     * As {@link #attention(Var, Var, Var, Tensor, float, float, java.util.random.RandomGenerator)},
     * also handing out the attention weights, for interpretability or eviction policies. The
     * weights are built then, so the fused kernel is not used.
     *
     * @param q queries {@code [batch, heads, queries, dk]}
     * @param k keys {@code [batch, kvHeads, keys, dk]}
     * @param v values {@code [batch, kvHeads, keys, dv]}
     * @param bias added to the scores before the softmax, or {@code null}
     * @param scale the score scale
     * @param dropout the share of attention weights dropped while training
     * @param random the random source for dropout, or {@code null}
     * @param weightsOut when not {@code null}, {@code weightsOut[0]} receives the weights
     *     {@code [batch, heads, queries, keys]} (after dropout)
     * @return {@code [batch, heads, queries, dv]}
     */
    public static Var attention(Var q, Var k, Var v, Tensor bias, float scale, float dropout,
            java.util.random.RandomGenerator random, Tensor[] weightsOut) {
        int heads = q.shape().dim(1);
        int kvHeads = k.shape().dim(1);
        if (heads % kvHeads != 0 || v.shape().dim(1) != kvHeads) {
            throw new IllegalArgumentException("query heads " + heads + " must be a multiple of key/value heads "
                    + kvHeads + " (and keys and values need the same heads)");
        }
        if (FUSED && bias != null && weightsOut == null && (dropout == 0 || random == null)
                && !(q.requiresGrad() || k.requiresGrad() || v.requiresGrad())
                && q.value().dtype() == k.value().dtype() && q.value().dtype() == v.value().dtype()
                && Tensor.hasAttentionWithBias(q.value(), k.value(), v.value())) {
            try {
                // inference: the fused kernel adds the bias itself and never builds the weights
                return Var.of(Tensor.attention(q.value(), k.value(), v.value(), bias, scale, false)[0]);
            } catch (UnsupportedOperationException | IllegalArgumentException e) {
                // this backend, type, size or bias shape has no fused kernel
            }
        }
        if (kvHeads != heads) {
            k = repeatHeads(k, heads / kvHeads);
            v = repeatHeads(v, heads / kvHeads);
        }
        Var scores = q.matmul(k.transpose()).mul(scale);
        if (bias != null) {
            scores = scores.add(bias);
        }
        Var weights = Activation.SOFTMAX.forward(scores);
        if (weightsOut != null) {
            weightsOut[0] = weights.value();
        }
        if (dropout > 0 && random != null) {
            Tensor keep = Tensor.random(weights.shape(), random.nextLong(), weights.value().device())
                    .greater(Tensor.full(Shape.of(), dropout, weights.value().device()));
            weights = weights.mul(keep.div(1 - dropout));
        }
        return weights.matmul(v);
    }

    /** Whether {@link #attention(Var, Var, Var, boolean, float)} may use a backend's fused kernel. */
    private static final boolean FUSED = !"false".equals(System.getProperty("brainiac.fusedAttention"));

    /**
     * Scaled dot-product attention without extra biases or dropout, optionally causal (see
     * {@link #causalMask}). Uses the backend's fused kernel when it has one, which never
     * builds the {@code [queries, keys]} weights; otherwise the same as
     * {@link #attention(Var, Var, Var, Tensor, float, float, java.util.random.RandomGenerator)}
     * with a causal mask. {@code -Dbrainiac.fusedAttention=false} turns the kernel off.
     *
     * @param q queries {@code [batch, heads, queries, dk]}
     * @param k keys {@code [batch, kvHeads, keys, dk]}
     * @param v values {@code [batch, kvHeads, keys, dv]}
     * @param causal whether query {@code i} sees only keys up to {@code i + keys - queries}
     * @param scale the score scale
     * @return {@code [batch, heads, queries, dv]}
     */
    public static Var attention(Var q, Var k, Var v, boolean causal, float scale) {
        if (FUSED && q.value().dtype() == k.value().dtype() && q.value().dtype() == v.value().dtype()
                && Tensor.hasAttention(q.value(), k.value(), v.value(),
                        q.requiresGrad() || k.requiresGrad() || v.requiresGrad())) {
            try {
                return Var.apply(new FusedAttention(scale, causal), q, k, v);
            } catch (UnsupportedOperationException e) {
                // this backend, type or size has no fused kernel
            }
        }
        Tensor bias = causal ? causalMask(q.shape().dim(2), k.shape().dim(2), q.value().device()) : null;
        return attention(q, k, v, bias, scale, 0, null);
    }

    /** The fused kernel; keeps the log-sum-exp of each query's scores for the gradient. */
    private static final class FusedAttention implements Op {
        private final float scale;
        private final boolean causal;
        private Tensor lse;

        FusedAttention(float scale, boolean causal) {
            this.scale = scale;
            this.causal = causal;
        }

        @Override
        public Tensor forward(Tensor... in) {
            Tensor[] r = Tensor.attention(in[0], in[1], in[2], scale, causal);
            lse = r[1];
            return r[0];
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
            return Tensor.attentionBackward(in[0], in[1], in[2], output, lse, grad, scale, causal);
        }

        @Override
        public String name() {
            return "attention";
        }
    }

    private static Var repeatHeads(Var x, int times) {
        int[] s = x.shape().toArray();
        Var e = x.reshape(s[0], s[1], 1, s[2], s[3]).expand(Shape.of(s[0], s[1], times, s[2], s[3]));
        return e.reshape(s[0], s[1] * times, s[2], s[3]);
    }

    /**
     * A causal mask: query {@code i} sees keys up to {@code i + keys - queries}, so the
     * last query sees every key (which also fits a cache of earlier keys).
     *
     * @param queries the number of queries
     * @param keys the number of keys
     * @param device where to create it
     * @return {@code [queries, keys]} with 0 where allowed and a large negative value elsewhere
     */
    public static Tensor causalMask(int queries, int keys, Device device) {
        float[] m = new float[queries * keys];
        int offset = keys - queries;
        for (int i = 0; i < queries; i++) {
            for (int j = i + offset + 1; j < keys; j++) {
                m[i * keys + j] = MASKED;
            }
        }
        return Tensor.of(Shape.of(queries, keys), m, device);
    }

    /**
     * Turns a padding mask into a score bias.
     *
     * @param keep {@code [batch, keys]}, 1 for real keys and 0 for padding
     * @return {@code [batch, 1, 1, keys]}, to add to attention scores
     */
    public static Tensor keyMask(Tensor keep) {
        int n = keep.shape().dim(0);
        int keys = keep.shape().dim(-1);
        return keep.reshape(n, 1, 1, keys).sub(1f).mul(-MASKED);
    }

    /**
     * ALiBi position biases: each head penalizes distant keys linearly, with slopes
     * {@code 2^(-8 h / heads)}, instead of position embeddings.
     *
     * @param heads the number of heads
     * @param queries the number of queries
     * @param keys the number of keys
     * @param device where to create it
     * @return {@code [heads, queries, keys]}
     */
    public static Tensor alibi(int heads, int queries, int keys, Device device) {
        float[] m = new float[heads * queries * keys];
        int offset = keys - queries;
        for (int h = 0; h < heads; h++) {
            double slope = Math.pow(2, -8.0 * (h + 1) / heads);
            for (int i = 0; i < queries; i++) {
                for (int j = 0; j < keys; j++) {
                    m[(h * queries + i) * keys + j] = (float) (-slope * Math.abs(i + offset - j));
                }
            }
        }
        return Tensor.of(Shape.of(heads, queries, keys), m, device);
    }

    /**
     * Rotary position embedding (RoPE): rotates pairs of features by an angle that grows
     * with the position, so dot products depend on relative positions. Uses the
     * half-split layout: feature {@code i} pairs with {@code i + d/2}.
     *
     * @param x {@code [..., steps, d]} with even {@code d}
     * @param base the frequency base, usually 10000
     * @param offset the position of the first step
     * @return the rotated values
     */
    public static Var rotary(Var x, float base, int offset) {
        Shape s = x.shape();
        int steps = s.dim(-2);
        int d = s.dim(-1);
        if (d % 2 != 0) {
            throw new IllegalArgumentException("rotary embedding needs an even feature size, got " + d);
        }
        int half = d / 2;
        float[] cos = new float[steps * half];
        float[] sin = new float[steps * half];
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < half; i++) {
                double angle = (t + offset) * Math.pow(base, -2.0 * i / d);
                cos[t * half + i] = (float) Math.cos(angle);
                sin[t * half + i] = (float) Math.sin(angle);
            }
        }
        Device device = x.value().device();
        Tensor c = Tensor.of(Shape.of(steps, half), cos, device);
        Tensor sn = Tensor.of(Shape.of(steps, half), sin, device);
        Var x1 = x.slice(-1, 0, half);
        Var x2 = x.slice(-1, half, d);
        return Var.concat(-1, x1.mul(c).sub(x2.mul(sn)), x2.mul(c).add(x1.mul(sn)));
    }

    /**
     * Fixed sinusoidal position encodings from "Attention Is All You Need".
     *
     * @param steps the number of positions
     * @param features the feature size
     * @param base the frequency base, usually 10000
     * @param device where to create them
     * @return {@code [steps, features]}
     */
    public static Tensor sinusoidal(int steps, int features, float base, Device device) {
        float[] m = new float[steps * features];
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < features; i++) {
                double angle = t / Math.pow(base, 2.0 * (i / 2) / features);
                m[t * features + i] = (float) (i % 2 == 0 ? Math.sin(angle) : Math.cos(angle));
            }
        }
        return Tensor.of(Shape.of(steps, features), m, device);
    }

    // ---------------------------------------------------------------- helpers

    /** Multiplies one axis by a matrix {@code [in, out]}. */
    static Var alongAxis(Var x, int axis, Tensor matrix) {
        return moveAxisBack(moveAxisLast(x, axis).matmul(matrix), axis);
    }

    static Var moveAxisLast(Var x, int axis) {
        int rank = x.shape().rank();
        if (axis == rank - 1) {
            return x;
        }
        int[] perm = new int[rank];
        int k = 0;
        for (int i = 0; i < rank; i++) {
            if (i != axis) {
                perm[k++] = i;
            }
        }
        perm[rank - 1] = axis;
        return x.permute(perm);
    }

    static Var moveAxisBack(Var x, int axis) {
        int rank = x.shape().rank();
        if (axis == rank - 1) {
            return x;
        }
        int[] perm = new int[rank];
        for (int i = 0; i < rank; i++) {
            perm[i] = i < axis ? i : i == axis ? rank - 1 : i - 1;
        }
        return x.permute(perm);
    }

    private static int[] moveLast(int[] dims, int axis) {
        int[] out = new int[dims.length];
        int k = 0;
        for (int i = 0; i < dims.length; i++) {
            if (i != axis) {
                out[k++] = dims[i];
            }
        }
        out[dims.length - 1] = dims[axis];
        return out;
    }

    static int[] imageDims(int batch, int channels, int[] spatial) {
        int[] dims = new int[spatial.length + 2];
        dims[0] = batch;
        dims[1] = channels;
        System.arraycopy(spatial, 0, dims, 2, spatial.length);
        return dims;
    }

    static int[] channelShape(int channels, int spatialRank) {
        int[] dims = new int[spatialRank + 1];
        Arrays.fill(dims, 1);
        dims[0] = channels;
        return dims;
    }
}
