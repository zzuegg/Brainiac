package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.util.Arrays;

/**
 * Runs tensor operations on the CPU in plain, single-threaded Java. Storage is a
 * {@code float[]}, so a tensor holds at most about 2 billion values.
 */
public final class CpuBackend implements Backend {

    /** Creates the backend; called by {@link java.util.ServiceLoader}. */
    public CpuBackend() {
    }

    @Override
    public String name() {
        return "cpu";
    }

    @Override
    public Device.Kind kind() {
        return Device.Kind.CPU;
    }

    @Override
    public Storage allocate(long size) {
        if (size > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("the CPU backend holds at most " + (Integer.MAX_VALUE - 8)
                    + " values per tensor, requested " + size);
        }
        return new CpuStorage(new float[(int) size]);
    }

    @Override
    public Storage upload(float[] data) {
        return new CpuStorage(data.clone());
    }

    @Override
    public float[] download(Storage src) {
        return data(src).clone();
    }

    @Override
    public void fill(Storage dst, float value) {
        Arrays.fill(data(dst), value);
    }

    @Override
    public void unary(UnaryOp op, Storage in, Storage out) {
        float[] x = data(in);
        float[] y = data(out);
        switch (op) {
            case NEG -> { for (int i = 0; i < x.length; i++) y[i] = -x[i]; }
            case ABS -> { for (int i = 0; i < x.length; i++) y[i] = Math.abs(x[i]); }
            case EXP -> { for (int i = 0; i < x.length; i++) y[i] = (float) Math.exp(x[i]); }
            case LOG -> { for (int i = 0; i < x.length; i++) y[i] = (float) Math.log(x[i]); }
            case SQRT -> { for (int i = 0; i < x.length; i++) y[i] = (float) Math.sqrt(x[i]); }
            case SQUARE -> { for (int i = 0; i < x.length; i++) y[i] = x[i] * x[i]; }
            case TANH -> { for (int i = 0; i < x.length; i++) y[i] = (float) Math.tanh(x[i]); }
            case SIGMOID -> { for (int i = 0; i < x.length; i++) y[i] = sigmoid(x[i]); }
            case RELU -> { for (int i = 0; i < x.length; i++) y[i] = x[i] > 0f ? x[i] : 0f; }
            case STEP -> { for (int i = 0; i < x.length; i++) y[i] = x[i] > 0f ? 1f : 0f; }
            case SIGN -> { for (int i = 0; i < x.length; i++) y[i] = Math.signum(x[i]); }
        }
    }

    private static float sigmoid(float x) {
        // Split by sign so exp never overflows.
        if (x >= 0f) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        }
        double e = Math.exp(x);
        return (float) (e / (1.0 + e));
    }

    @Override
    public void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape, Storage out) {
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        int rank = shape.length;
        long[] contiguous = contiguousStrides(shape);

        if (Arrays.equals(aStrides, contiguous) && Arrays.equals(bStrides, contiguous)) {
            for (int i = 0; i < z.length; i++) {
                z[i] = apply(op, x[i], y[i]);
            }
            return;
        }
        if (Arrays.equals(aStrides, contiguous) && isZero(bStrides)) {
            float s = y[0];
            for (int i = 0; i < z.length; i++) {
                z[i] = apply(op, x[i], s);
            }
            return;
        }

        // General case: walk the output in row-major order, tracking both input offsets.
        int[] index = new int[rank];
        long ia = 0;
        long ib = 0;
        for (int i = 0; i < z.length; i++) {
            z[i] = apply(op, x[(int) ia], y[(int) ib]);
            for (int d = rank - 1; d >= 0; d--) {
                index[d]++;
                ia += aStrides[d];
                ib += bStrides[d];
                if (index[d] < shape[d]) {
                    break;
                }
                ia -= aStrides[d] * shape[d];
                ib -= bStrides[d] * shape[d];
                index[d] = 0;
            }
        }
    }

    private static float apply(BinaryOp op, float a, float b) {
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case POW -> (float) Math.pow(a, b);
            case MAX -> Math.max(a, b);
            case MIN -> Math.min(a, b);
            case GREATER -> a > b ? 1f : 0f;
            case EQUAL -> a == b ? 1f : 0f;
        };
    }

    @Override
    public void reduce(ReduceOp op, Storage in, long outer, long length, long inner, Storage out) {
        float[] x = data(in);
        float[] y = data(out);
        int len = (int) length;
        int inn = (int) inner;
        for (int o = 0; o < outer; o++) {
            int base = o * len * inn;
            for (int j = 0; j < inn; j++) {
                int start = base + j;
                y[o * inn + j] = switch (op) {
                    case SUM -> (float) sum(x, start, len, inn);
                    case MEAN -> (float) (sum(x, start, len, inn) / len);
                    case MAX -> {
                        float m = Float.NEGATIVE_INFINITY;
                        for (int i = 0; i < len; i++) m = Math.max(m, x[start + i * inn]);
                        yield m;
                    }
                    case MIN -> {
                        float m = Float.POSITIVE_INFINITY;
                        for (int i = 0; i < len; i++) m = Math.min(m, x[start + i * inn]);
                        yield m;
                    }
                    case ARGMAX -> {
                        int best = 0;
                        for (int i = 1; i < len; i++) {
                            if (x[start + i * inn] > x[start + best * inn]) best = i;
                        }
                        yield best;
                    }
                };
            }
        }
    }

    private static double sum(float[] x, int start, int len, int stride) {
        // Accumulate in double so long sums stay accurate.
        double s = 0;
        for (int i = 0; i < len; i++) {
            s += x[start + i * stride];
        }
        return s;
    }

    @Override
    public void matmul(Storage a, Storage b, int m, int k, int n, Storage out) {
        float[] x = data(a);
        float[] y = data(b);
        float[] z = data(out);
        Arrays.fill(z, 0f);
        // i-k-j order reads both b and out row by row, which is cache friendly.
        for (int i = 0; i < m; i++) {
            int zRow = i * n;
            for (int p = 0; p < k; p++) {
                float av = x[i * k + p];
                if (av == 0f) {
                    continue;
                }
                int yRow = p * n;
                for (int j = 0; j < n; j++) {
                    z[zRow + j] += av * y[yRow + j];
                }
            }
        }
    }

    @Override
    public void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out, long outOffset,
            long[] outStrides) {
        float[] x = data(in);
        float[] z = data(out);
        int rank = shape.length;
        long count = 1;
        for (int d : shape) {
            count *= d;
        }
        if (count == 0) {
            return;
        }
        if (rank == 0) {
            z[(int) outOffset] = x[(int) inOffset];
            return;
        }
        // copy along the last axis in a tight loop, step the other axes like an odometer
        int last = rank - 1;
        int run = shape[last];
        long inStep = inStrides[last];
        long outStep = outStrides[last];
        int[] index = new int[rank];
        long src = inOffset;
        long dst = outOffset;
        for (long done = 0; done < count; done += run) {
            if (inStep == 1 && outStep == 1) {
                System.arraycopy(x, (int) src, z, (int) dst, run);
            } else {
                long s = src;
                long t = dst;
                for (int k = 0; k < run; k++, s += inStep, t += outStep) {
                    z[(int) t] = x[(int) s];
                }
            }
            for (int d = last - 1; d >= 0; d--) {
                index[d]++;
                src += inStrides[d];
                dst += outStrides[d];
                if (index[d] < shape[d]) {
                    break;
                }
                src -= inStrides[d] * shape[d];
                dst -= outStrides[d] * shape[d];
                index[d] = 0;
            }
        }
    }

    private static long[] contiguousStrides(int[] shape) {
        long[] s = new long[shape.length];
        long step = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = step;
            step *= shape[i];
        }
        return s;
    }

    private static boolean isZero(long[] strides) {
        for (long s : strides) {
            if (s != 0) {
                return false;
            }
        }
        return true;
    }

    private static float[] data(Storage s) {
        if (s instanceof CpuStorage c) {
            return c.data;
        }
        throw new IllegalArgumentException("storage " + s + " does not belong to the CPU backend");
    }
}
