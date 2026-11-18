package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

/**
 * Storage backed by a Java array: {@code float[]} for float32, {@code double[]} for
 * float64 and {@code short[]} bit patterns for float16 and bfloat16.
 */
final class CpuStorage implements Storage {

    final DType dtype;
    /** the values of a float32 storage, otherwise {@code null} */
    final float[] data;
    /** the values of a float64 storage, otherwise {@code null} */
    final double[] wide;
    /** the bits of a float16 or bfloat16 storage, otherwise {@code null} */
    final short[] half;
    /** Integer and boolean values: kept as longs, read through doubles (exact up to 2^53). */
    final long[] ints;

    CpuStorage(float[] data) {
        this.dtype = DType.FLOAT32;
        this.data = data;
        this.wide = null;
        this.half = null;
        this.ints = null;
    }

    private CpuStorage(DType dtype, int size) {
        this.dtype = dtype;
        this.data = dtype == DType.FLOAT32 ? new float[size] : null;
        this.wide = dtype == DType.FLOAT64 ? new double[size] : null;
        this.half = dtype == DType.FLOAT16 || dtype == DType.BFLOAT16 ? new short[size] : null;
        this.ints = dtype.floating() ? null : new long[size];
    }

    static CpuStorage allocate(int size, DType dtype) {
        return dtype == DType.FLOAT32 ? new CpuStorage(new float[size]) : new CpuStorage(dtype, size);
    }

    @Override
    public long size() {
        return switch (dtype) {
            case FLOAT32 -> data.length;
            case FLOAT64 -> wide.length;
            case FLOAT16, BFLOAT16 -> half.length;
            case INT32, INT64, UINT8, BOOL -> ints.length;
        };
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    /** @return the values as doubles; the backing array itself for float64, so do not write to it */
    double[] read() {
        if (wide != null) {
            return wide;
        }
        double[] d = new double[(int) size()];
        if (data != null) {
            for (int i = 0; i < d.length; i++) {
                d[i] = data[i];
            }
        } else if (ints != null) {
            for (int i = 0; i < d.length; i++) {
                d[i] = ints[i];
            }
        } else if (dtype == DType.FLOAT16) {
            for (int i = 0; i < d.length; i++) {
                d[i] = Float.float16ToFloat(half[i]);
            }
        } else {
            for (int i = 0; i < d.length; i++) {
                d[i] = DType.bfloat16ToFloat(half[i]);
            }
        }
        return d;
    }

    /** @return an array to compute results into; the backing array itself for float64 */
    double[] target() {
        return wide != null ? wide : new double[(int) size()];
    }

    /**
     * Stores computed values, rounding them to this storage's type.
     *
     * @param values the values, possibly the array {@link #target()} returned
     */
    void write(double[] values) {
        switch (dtype) {
            case FLOAT64 -> {
                if (values != wide) {
                    System.arraycopy(values, 0, wide, 0, wide.length);
                }
            }
            case FLOAT32 -> {
                for (int i = 0; i < data.length; i++) {
                    data[i] = (float) values[i];
                }
            }
            case FLOAT16 -> {
                for (int i = 0; i < half.length; i++) {
                    half[i] = Float.floatToFloat16((float) values[i]);
                }
            }
            case BFLOAT16 -> {
                for (int i = 0; i < half.length; i++) {
                    half[i] = DType.floatToBfloat16((float) values[i]);
                }
            }
            case INT32, INT64, UINT8, BOOL -> {
                for (int i = 0; i < ints.length; i++) {
                    ints[i] = (long) dtype.round(values[i]);
                }
            }
        }
    }

    /** @return the values as floats, a new array */
    float[] readFloat() {
        if (data != null) {
            return data.clone();
        }
        double[] d = read();
        float[] f = new float[d.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }
}
