package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Integer-only inference for an {@link EdgeNet}: int8 weights (symmetric, per output channel),
 * int8 activations (affine, {@code r = S (q - Z)}), int32 accumulators and biases, and
 * requantisation by a fixed-point multiplier and a shift, with ReLU folded into the output clamp
 * and batch norm folded into the weights.
 *
 * <p>For a layer with input scale {@code S_in}, weight scale {@code S_w[c]} and output scale
 * {@code S_out}, the real multiplier {@code M = S_in S_w[c] / S_out} is written as
 * {@code M0 * 2^shift} with {@code M0} an int32 in {@code [2^30, 2^31)}, and the int32
 * accumulator is requantised with gemmlowp's saturating rounding doubling high multiply and
 * rounding right shift, so no floating point is used at run time.
 *
 * <p>Two engines run the same integer model: {@link #run} uses an im2col / GEMM kernel with the
 * zero-point offset hoisted out of the inner loop ({@code sum (q_x - Z) q_w = sum q_x q_w - Z sum
 * q_w}), and {@link #runReference} is a plain scalar loop over the definition; their outputs must
 * agree bit for bit.
 *
 * <p>Paper: Jacob et al., "Quantization and Training of Neural Networks for Efficient
 * Integer-Arithmetic-Only Inference", CVPR 2018,
 * <a href="https://arxiv.org/abs/1712.05877">arXiv:1712.05877</a>.
 */
public final class IntegerInference {

    /** An integer layer. */
    private abstract static sealed class QLayer permits QConv, QLinear, QPool {
        int zIn;
        int zOut;
        int lo;
        int hi;
        int inSize;
        int outSize;
    }

    private static final class QConv extends QLayer {
        int in;
        int out;
        int k;
        int stride;
        int pad;
        int groups;
        int inH;
        int inW;
        int outH;
        int outW;
        byte[] w;
        int[] bias;
        int[] mult;
        int[] shift;
        int[] wSum;
    }

    private static final class QLinear extends QLayer {
        int in;
        int out;
        byte[] w;          // [out][in], transposed for row dot products
        int[] bias;
        int[] mult;
        int[] shift;
        int[] wSum;
    }

    private static final class QPool extends QLayer {
        int channels;
        int area;
        int mult;
        int shift;
    }

    private final List<QLayer> layers;
    private final float inScale;
    private final int inZero;
    private final float outScale;
    private final int outZero;

    private IntegerInference(List<QLayer> layers, float inScale, int inZero, float outScale, int outZero) {
        this.layers = layers;
        this.inScale = inScale;
        this.inZero = inZero;
        this.outScale = outScale;
        this.outZero = outZero;
    }

    // ------------------------------------------------------------------ quantisation helpers

    /**
     * Affine int8 parameters for a real range (widened to contain zero, so that zero padding and
     * ReLU are exact).
     *
     * @param min the smallest value
     * @param max the largest value
     * @return {@code {scale, zeroPoint}} with {@code q = clamp(round(r / scale) + zeroPoint, -128, 127)}
     */
    public static float[] affineParams(float min, float max) {
        float lo = Math.min(0, min);
        float hi = Math.max(0, max);
        float scale = (hi - lo) / 255f;
        if (!(scale > 1e-12f)) {
            scale = 1e-6f;
        }
        int zero = Math.round(-128 - lo / scale);
        zero = Math.max(-128, Math.min(127, zero));
        return new float[] {scale, zero};
    }

    /**
     * Per-output-channel symmetric int8 scales of a weight.
     *
     * @param w the weight
     * @param convLayout {@code [out, ...]} when true, {@code [in, out]} when false
     * @return one scale per output channel
     */
    public static float[] weightScales(Tensor w, boolean convLayout) {
        float[] v = w.toArray();
        int out = convLayout ? w.shape().dim(0) : w.shape().dim(w.shape().rank() - 1);
        float[] max = new float[out];
        for (int i = 0; i < v.length; i++) {
            int c = channel(i, v.length, out, convLayout);
            max[c] = Math.max(max[c], Math.abs(v[i]));
        }
        float[] s = new float[out];
        for (int c = 0; c < out; c++) {
            s[c] = max[c] > 0 ? max[c] / 127f : 1f;
        }
        return s;
    }

    private static int channel(int i, int n, int out, boolean convLayout) {
        return convLayout ? i / (n / out) : i % out;
    }

    /**
     * Rounds weights to the per-channel int8 grid and back (the fake quantisation that
     * {@link EdgeNet}'s quantisation-aware training uses).
     *
     * @param w the weight
     * @param convLayout {@code [out, ...]} when true, {@code [in, out]} when false
     * @return the dequantised weights
     */
    public static Tensor fakeQuantWeights(Tensor w, boolean convLayout) {
        float[] v = w.toArray();
        float[] s = weightScales(w, convLayout);
        float[] q = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            float sc = s[channel(i, v.length, s.length, convLayout)];
            q[i] = Math.max(-127, Math.min(127, Math.round(v[i] / sc))) * sc;
        }
        return Tensor.of(w.shape(), q);
    }

    /**
     * Writes a positive real multiplier as an int32 fixed-point multiplier and a power-of-two
     * exponent: {@code m ~= mult * 2^-31 * 2^shift}.
     *
     * @param m the multiplier
     * @return {@code {mult, shift}} (shift positive means a left shift)
     */
    public static int[] quantizeMultiplier(double m) {
        if (m <= 0) {
            return new int[] {0, 0};
        }
        int e = Math.getExponent(m) + 1;
        double f = m / Math.pow(2, e);           // in [0.5, 1)
        long q = Math.round(f * (1L << 31));
        if (q == 1L << 31) {
            q /= 2;
            e++;
        }
        if (e < -31) {
            return new int[] {0, 0};
        }
        return new int[] {(int) q, e};
    }

    /**
     * gemmlowp's saturating rounding doubling high multiply: {@code round(a * b / 2^31)}.
     *
     * @param a a value
     * @param b a fixed-point multiplier
     * @return the product's high half
     */
    static int saturatingRoundingDoublingHighMul(int a, int b) {
        if (a == Integer.MIN_VALUE && b == Integer.MIN_VALUE) {
            return Integer.MAX_VALUE;
        }
        long ab = (long) a * b;
        long nudge = ab >= 0 ? 1L << 30 : 1 - (1L << 30);
        return (int) ((ab + nudge) / (1L << 31));
    }

    /**
     * gemmlowp's rounding right shift (round half away from zero).
     *
     * @param x the value
     * @param exponent the shift
     * @return {@code round(x / 2^exponent)}
     */
    static int roundingDivideByPot(int x, int exponent) {
        if (exponent == 0) {
            return x;
        }
        int mask = (1 << exponent) - 1;
        int remainder = x & mask;
        int threshold = (mask >> 1) + (x < 0 ? 1 : 0);
        return (x >> exponent) + (remainder > threshold ? 1 : 0);
    }

    /**
     * Multiplies an int32 accumulator by a quantised multiplier, integers only.
     *
     * @param acc the accumulator
     * @param mult the fixed-point multiplier
     * @param shift the exponent from {@link #quantizeMultiplier}
     * @return the scaled value
     */
    public static int requantize(int acc, int mult, int shift) {
        int left = Math.max(0, shift);
        int right = Math.max(0, -shift);
        return roundingDivideByPot(saturatingRoundingDoublingHighMul(acc * (1 << left), mult), right);
    }

    // ------------------------------------------------------------------ conversion

    /**
     * Converts a trained network. A network trained with quantisation-aware training uses the
     * activation ranges it tracked; otherwise (post-training quantisation) the ranges are the
     * minimum and maximum of each layer's output on calibration data. Batch norm is folded.
     *
     * @param net the network
     * @param calibration calibration inputs (ignored for ranges a QAT network tracked)
     * @return the integer model
     */
    public static IntegerInference convert(EdgeNet net, Tensor calibration) {
        int n = net.units().size();
        float[][] ranges = new float[n + 1][];
        boolean qat = net.isQuantAware() && net.inputRange().tracked;
        if (qat) {
            ranges[0] = new float[] {net.inputRange().emaMin, net.inputRange().emaMax};
            for (int i = 0; i < n; i++) {
                EdgeNet.Unit u = net.units().get(i);
                ranges[i + 1] = new float[] {u.emaMin, u.emaMax};
            }
        } else {
            Var h = Var.of(calibration);
            ranges[0] = minMax(calibration);
            for (int i = 0; i < n; i++) {
                Var in = h;
                int at = i;
                h = Var.noGrad(() -> net.forwardUnits(in, at, at + 1, false));
                ranges[i + 1] = minMax(h.value());
            }
        }
        float[][] sz = new float[n + 1][];
        for (int i = 0; i <= n; i++) {
            sz[i] = affineParams(ranges[i][0], ranges[i][1]);
        }
        List<int[]> shapes = net.shapes();
        int[] inShape = net.inputShape();
        List<QLayer> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float sIn = sz[i][0];
            int zIn = (int) sz[i][1];
            float sOut = sz[i + 1][0];
            int zOut = (int) sz[i + 1][1];
            int[] before = i == 0 ? inShape : shapes.get(i - 1);
            int[] after = shapes.get(i);
            QLayer q = switch (net.units().get(i)) {
                case EdgeNet.Conv c -> conv(c, before, after, sIn, sOut);
                case EdgeNet.Linear l -> linear(l, sIn, sOut);
                case EdgeNet.Pool p -> pool(before, sIn, sOut);
            };
            boolean relu = switch (net.units().get(i)) {
                case EdgeNet.Conv c -> c.relu();
                case EdgeNet.Linear l -> l.relu();
                case EdgeNet.Pool p -> false;
            };
            q.zIn = zIn;
            q.zOut = zOut;
            q.lo = relu ? Math.max(-128, zOut) : -128;
            q.hi = 127;
            q.inSize = (int) EdgeNet.size(before);
            q.outSize = (int) EdgeNet.size(after);
            out.add(q);
        }
        return new IntegerInference(out, sz[0][0], (int) sz[0][1], sz[n][0], (int) sz[n][1]);
    }

    private static float[] minMax(Tensor t) {
        float lo = 0;
        float hi = 0;
        for (float v : t.toArray()) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        return new float[] {lo, hi};
    }

    private static QConv conv(EdgeNet.Conv c, int[] before, int[] after, float sIn, float sOut) {
        Tensor[] f = c.folded();
        QConv q = new QConv();
        q.in = c.inChannels();
        q.out = c.outChannels();
        q.k = c.kernel();
        q.stride = c.stride();
        q.pad = c.pad();
        q.groups = c.groups();
        q.inH = before[1];
        q.inW = before[2];
        q.outH = after[1];
        q.outW = after[2];
        float[] s = weightScales(f[0], true);
        float[] w = f[0].toArray();
        float[] b = f[1].toArray();
        int per = w.length / q.out;
        q.w = new byte[w.length];
        q.wSum = new int[q.out];
        q.bias = new int[q.out];
        q.mult = new int[q.out];
        q.shift = new int[q.out];
        for (int oc = 0; oc < q.out; oc++) {
            for (int j = 0; j < per; j++) {
                int v = Math.max(-127, Math.min(127, Math.round(w[oc * per + j] / s[oc])));
                q.w[oc * per + j] = (byte) v;
                q.wSum[oc] += v;
            }
            q.bias[oc] = (int) Math.round(b[oc] / ((double) sIn * s[oc]));
            int[] m = quantizeMultiplier((double) sIn * s[oc] / sOut);
            q.mult[oc] = m[0];
            q.shift[oc] = m[1];
        }
        return q;
    }

    private static QLinear linear(EdgeNet.Linear l, float sIn, float sOut) {
        Tensor wt = l.weight().value();
        int in = wt.shape().dim(0);
        int outN = wt.shape().dim(1);
        float[] s = weightScales(wt, false);
        float[] w = wt.toArray();
        float[] b = l.bias().value().toArray();
        QLinear q = new QLinear();
        q.in = in;
        q.out = outN;
        q.w = new byte[w.length];
        q.wSum = new int[outN];
        q.bias = new int[outN];
        q.mult = new int[outN];
        q.shift = new int[outN];
        for (int o = 0; o < outN; o++) {
            for (int i = 0; i < in; i++) {
                int v = Math.max(-127, Math.min(127, Math.round(w[i * outN + o] / s[o])));
                q.w[o * in + i] = (byte) v;
                q.wSum[o] += v;
            }
            q.bias[o] = (int) Math.round(b[o] / ((double) sIn * s[o]));
            int[] m = quantizeMultiplier((double) sIn * s[o] / sOut);
            q.mult[o] = m[0];
            q.shift[o] = m[1];
        }
        return q;
    }

    private static QPool pool(int[] before, float sIn, float sOut) {
        QPool q = new QPool();
        q.channels = before[0];
        q.area = before[1] * before[2];
        int[] m = quantizeMultiplier((double) sIn / (sOut * q.area));
        q.mult = m[0];
        q.shift = m[1];
        return q;
    }

    // ------------------------------------------------------------------ execution

    /**
     * Quantises float inputs with the model's input parameters.
     *
     * @param x inputs {@code [n, ...]}
     * @return per sample, the int8 values
     */
    public byte[][] quantizeInput(Tensor x) {
        int n = x.shape().dim(0);
        float[] v = x.toArray();
        int per = v.length / n;
        byte[][] q = new byte[n][per];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < per; j++) {
                q[i][j] = (byte) Math.max(-128, Math.min(127, Math.round(v[i * per + j] / inScale) + inZero));
            }
        }
        return q;
    }

    /**
     * Runs the optimised integer kernels (im2col and an offset-corrected integer GEMM).
     *
     * @param input one sample's int8 input
     * @return the int8 outputs
     */
    public byte[] run(byte[] input) {
        byte[] h = input;
        for (QLayer l : layers) {
            h = switch (l) {
                case QConv c -> convFast(c, h);
                case QLinear d -> linearFast(d, h);
                case QPool p -> pool(p, h);
            };
        }
        return h;
    }

    /**
     * Runs the scalar reference kernels, written directly from the definition.
     *
     * @param input one sample's int8 input
     * @return the int8 outputs
     */
    public byte[] runReference(byte[] input) {
        byte[] h = input;
        for (QLayer l : layers) {
            h = switch (l) {
                case QConv c -> convReference(c, h);
                case QLinear d -> linearReference(d, h);
                case QPool p -> pool(p, h);
            };
        }
        return h;
    }

    private static byte clamp(int v, QLayer l) {
        return (byte) Math.max(l.lo, Math.min(l.hi, v));
    }

    private static byte[] convReference(QConv c, byte[] x) {
        byte[] y = new byte[c.out * c.outH * c.outW];
        int icPer = c.in / c.groups;
        int ocPer = c.out / c.groups;
        for (int oc = 0; oc < c.out; oc++) {
            int g = oc / ocPer;
            for (int oy = 0; oy < c.outH; oy++) {
                for (int ox = 0; ox < c.outW; ox++) {
                    int acc = c.bias[oc];
                    for (int ic = 0; ic < icPer; ic++) {
                        int ch = g * icPer + ic;
                        for (int ky = 0; ky < c.k; ky++) {
                            int iy = oy * c.stride - c.pad + ky;
                            if (iy < 0 || iy >= c.inH) {
                                continue;
                            }
                            for (int kx = 0; kx < c.k; kx++) {
                                int ix = ox * c.stride - c.pad + kx;
                                if (ix < 0 || ix >= c.inW) {
                                    continue;
                                }
                                int xv = x[(ch * c.inH + iy) * c.inW + ix] - c.zIn;
                                acc += xv * c.w[((oc * icPer + ic) * c.k + ky) * c.k + kx];
                            }
                        }
                    }
                    y[(oc * c.outH + oy) * c.outW + ox] = clamp(c.zOut + requantize(acc, c.mult[oc], c.shift[oc]), c);
                }
            }
        }
        return y;
    }

    private static byte[] convFast(QConv c, byte[] x) {
        int icPer = c.in / c.groups;
        int ocPer = c.out / c.groups;
        int patch = icPer * c.k * c.k;
        int pixels = c.outH * c.outW;
        byte[] cols = new byte[pixels * patch];
        byte[] y = new byte[c.out * pixels];
        for (int g = 0; g < c.groups; g++) {
            // im2col for this group, padding with the input zero point (a real zero)
            for (int oy = 0; oy < c.outH; oy++) {
                for (int ox = 0; ox < c.outW; ox++) {
                    int row = (oy * c.outW + ox) * patch;
                    int j = 0;
                    for (int ic = 0; ic < icPer; ic++) {
                        int base = (g * icPer + ic) * c.inH;
                        for (int ky = 0; ky < c.k; ky++) {
                            int iy = oy * c.stride - c.pad + ky;
                            for (int kx = 0; kx < c.k; kx++) {
                                int ix = ox * c.stride - c.pad + kx;
                                cols[row + j++] = iy < 0 || iy >= c.inH || ix < 0 || ix >= c.inW
                                        ? (byte) c.zIn : x[(base + iy) * c.inW + ix];
                            }
                        }
                    }
                }
            }
            for (int o = 0; o < ocPer; o++) {
                int oc = g * ocPer + o;
                int wOff = oc * patch;
                int offset = c.bias[oc] - c.zIn * c.wSum[oc];
                for (int p = 0; p < pixels; p++) {
                    int acc = 0;
                    int row = p * patch;
                    for (int j = 0; j < patch; j++) {
                        acc += cols[row + j] * c.w[wOff + j];
                    }
                    y[oc * pixels + p] = clamp(c.zOut + requantize(acc + offset, c.mult[oc], c.shift[oc]), c);
                }
            }
        }
        return y;
    }

    private static byte[] linearReference(QLinear d, byte[] x) {
        byte[] y = new byte[d.out];
        for (int o = 0; o < d.out; o++) {
            int acc = d.bias[o];
            for (int i = 0; i < d.in; i++) {
                acc += (x[i] - d.zIn) * d.w[o * d.in + i];
            }
            y[o] = clamp(d.zOut + requantize(acc, d.mult[o], d.shift[o]), d);
        }
        return y;
    }

    private static byte[] linearFast(QLinear d, byte[] x) {
        byte[] y = new byte[d.out];
        for (int o = 0; o < d.out; o++) {
            int acc = 0;
            int off = o * d.in;
            for (int i = 0; i < d.in; i++) {
                acc += x[i] * d.w[off + i];
            }
            acc += d.bias[o] - d.zIn * d.wSum[o];
            y[o] = clamp(d.zOut + requantize(acc, d.mult[o], d.shift[o]), d);
        }
        return y;
    }

    private static byte[] pool(QPool p, byte[] x) {
        byte[] y = new byte[p.channels];
        for (int c = 0; c < p.channels; c++) {
            int acc = 0;
            for (int i = 0; i < p.area; i++) {
                acc += x[c * p.area + i] - p.zIn;
            }
            y[c] = clamp(p.zOut + requantize(acc, p.mult, p.shift), p);
        }
        return y;
    }

    // ------------------------------------------------------------------ evaluation

    /**
     * Runs the integer model on float inputs and dequantises the outputs.
     *
     * @param x inputs {@code [n, ...]}
     * @return the outputs {@code [n, outputs]}
     */
    public Tensor predict(Tensor x) {
        byte[][] q = quantizeInput(x);
        int outs = layers.get(layers.size() - 1).outSize;
        float[] y = new float[q.length * outs];
        for (int i = 0; i < q.length; i++) {
            byte[] o = run(q[i]);
            for (int j = 0; j < outs; j++) {
                y[i * outs + j] = (o[j] - outZero) * outScale;
            }
        }
        return Tensor.of(Shape.of(q.length, outs), y);
    }

    /**
     * Classification accuracy of the integer model (the arg-max of the int8 outputs).
     *
     * @param x inputs
     * @param y class ids
     * @return the share classified correctly
     */
    public double accuracy(Tensor x, Tensor y) {
        byte[][] q = quantizeInput(x);
        float[] ids = y.toArray();
        int ok = 0;
        for (int i = 0; i < q.length; i++) {
            ok += argmax(run(q[i])) == (int) ids[i] ? 1 : 0;
        }
        return ok / (double) q.length;
    }

    private static int argmax(byte[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Whether the optimised and the reference kernels agree bit for bit.
     *
     * @param x inputs
     * @return the share of samples whose int8 outputs are identical
     */
    public double bitExact(Tensor x) {
        byte[][] q = quantizeInput(x);
        int same = 0;
        for (byte[] s : q) {
            same += java.util.Arrays.equals(run(s), runReference(s)) ? 1 : 0;
        }
        return same / (double) q.length;
    }

    /**
     * Compares the integer model with the network's own float forward pass (its fake-quantised
     * forward when quantisation-aware), both quantised to the output grid.
     *
     * @param net the source network
     * @param x inputs
     * @return {@code {share of equal int8 outputs, largest difference in output steps, arg-max agreement}}
     */
    public double[] agreement(EdgeNet net, Tensor x) {
        float[] f = Var.noGrad(() -> net.forward(Var.of(x), false)).value().toArray();
        byte[][] q = quantizeInput(x);
        int outs = layers.get(layers.size() - 1).outSize;
        int equal = 0;
        int maxDiff = 0;
        int argmaxSame = 0;
        for (int i = 0; i < q.length; i++) {
            byte[] o = run(q[i]);
            byte[] r = new byte[outs];
            for (int j = 0; j < outs; j++) {
                int fq = Math.max(-128, Math.min(127, Math.round(f[i * outs + j] / outScale) + outZero));
                r[j] = (byte) fq;
                equal += fq == o[j] ? 1 : 0;
                maxDiff = Math.max(maxDiff, Math.abs(fq - o[j]));
            }
            argmaxSame += argmax(o) == argmax(r) ? 1 : 0;
        }
        return new double[] {equal / (double) (q.length * outs), maxDiff, argmaxSame / (double) q.length};
    }

    /**
     * Bytes of the deployed model: int8 weights, int32 biases, and an int32 multiplier and an
     * int8 shift per output channel.
     *
     * @return the model size in bytes
     */
    public long modelBytes() {
        long b = 0;
        for (QLayer l : layers) {
            b += switch (l) {
                case QConv c -> c.w.length + 9L * c.out;
                case QLinear d -> d.w.length + 9L * d.out;
                case QPool p -> 5;
            };
        }
        return b;
    }

    /**
     * Integer multiply-accumulates per sample.
     *
     * @return the count
     */
    public long macs() {
        long m = 0;
        for (QLayer l : layers) {
            m += switch (l) {
                case QConv c -> (long) c.out * c.outH * c.outW * (c.in / c.groups) * c.k * c.k;
                case QLinear d -> (long) d.in * d.out;
                case QPool p -> (long) p.channels * p.area;
            };
        }
        return m;
    }

    /**
     * Peak activation bytes of layer-by-layer execution (one layer's int8 input and output).
     *
     * @return the peak
     */
    public long peakActivationBytes() {
        long peak = 0;
        for (QLayer l : layers) {
            peak = Math.max(peak, (long) l.inSize + l.outSize);
        }
        return peak;
    }

    /**
     * Mean wall time per sample of the optimised integer engine.
     *
     * @param x inputs
     * @param repeats passes over the inputs
     * @return nanoseconds per sample
     */
    public double nanosPerSample(Tensor x, int repeats) {
        byte[][] q = quantizeInput(x);
        long sink = 0;
        long t0 = System.nanoTime();
        for (int r = 0; r < repeats; r++) {
            for (byte[] s : q) {
                sink += run(s)[0];
            }
        }
        long t = System.nanoTime() - t0;
        return sink == Long.MIN_VALUE ? 0 : t / (double) (repeats * q.length);
    }
}
