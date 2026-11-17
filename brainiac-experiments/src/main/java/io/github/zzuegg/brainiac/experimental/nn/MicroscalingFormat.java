package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * 4-bit microscaling floats: every value is an FP4 E2M1 number (sign, two exponent bits, one
 * mantissa bit: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6), and each block of consecutive values
 * shares one 8-bit scale. Small blocks follow local ranges closely, so 4-bit floats come near
 * int4's accuracy with hardware-friendly decoding (Blackwell tensor cores run both natively).
 * <ul>
 * <li>{@link #MXFP4} (OCP Microscaling spec): blocks of 32 with a power-of-two E8M0 scale,
 *     {@code 2^(floor(log2(amax)) - 2)}, 4.25 bits per value;</li>
 * <li>{@link #NVFP4} (NVIDIA): blocks of 16 with an FP8 E4M3 scale, {@code amax / 6} rounded
 *     to E4M3 relative to one float32 scale per tensor, 4.5 bits per value.</li>
 * </ul>
 * Matrices {@code [rows, cols]} are blocked along the rows of each column, the input axis of
 * a {@code [in, out]} weight, as the core {@code Tensor.quantized(bits, group)} blocks int
 * levels; {@link #quantize} returns the dequantized values. It sits next to {@link Fp8Format}.
 *
 * <p>Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023),
 * <a href="https://arxiv.org/abs/2310.10537">arXiv:2310.10537</a>; NVIDIA, "Introducing NVFP4"
 * (2025).
 */
public enum MicroscalingFormat {

    /** OCP MXFP4: E2M1 values, 32 per power-of-two scale. */
    MXFP4(32),
    /** NVIDIA NVFP4: E2M1 values, 16 per E4M3 scale, plus a float32 scale per tensor. */
    NVFP4(16);

    private static final float[] GRID = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    private static final float MAX = 6;

    private final int block;

    MicroscalingFormat(int block) {
        this.block = block;
    }

    /** @return the values per shared scale */
    public int block() {
        return block;
    }

    /** @return the bits per value, the shared scale included */
    public double bitsPerValue() {
        return 4 + 8.0 / block;
    }

    /**
     * Rounds a magnitude-scaled value to the nearest E2M1 number (ties to the even grid point,
     * saturating at 6).
     *
     * @param v the value
     * @return the nearest FP4 E2M1 value
     */
    public static float roundE2m1(float v) {
        float a = Math.min(Math.abs(v), MAX);
        int best = 0;
        for (int i = 1; i < GRID.length; i++) {
            float d = Math.abs(GRID[i] - a);
            float b = Math.abs(GRID[best] - a);
            if (d < b || d == b && i % 2 == 0) {
                best = i;
            }
        }
        return Math.copySign(GRID[best], v);
    }

    /**
     * Quantizes a {@code [rows, cols]} matrix block by block along its rows and returns the
     * values it stands for.
     *
     * @param matrix the matrix, rows a multiple of {@link #block()}
     * @return the dequantized matrix
     */
    public Tensor quantize(Tensor matrix) {
        int rows = matrix.shape().dim(0);
        int cols = matrix.shape().dim(1);
        if (rows % block != 0) {
            throw new IllegalArgumentException(rows + " rows are not a multiple of " + block);
        }
        float[] w = matrix.toArray();
        float tensorScale = 1;
        if (this == NVFP4) {
            float amax = 0;
            for (float v : w) {
                amax = Math.max(amax, Math.abs(v));
            }
            tensorScale = amax == 0 ? 1 : amax / (Fp8Format.E4M3.max() * MAX);
        }
        float[] out = new float[w.length];
        for (int c = 0; c < cols; c++) {
            for (int r0 = 0; r0 < rows; r0 += block) {
                float amax = 0;
                for (int r = r0; r < r0 + block; r++) {
                    amax = Math.max(amax, Math.abs(w[r * cols + c]));
                }
                float scale = scale(amax, tensorScale);
                for (int r = r0; r < r0 + block; r++) {
                    out[r * cols + c] = scale == 0 ? 0 : roundE2m1(w[r * cols + c] / scale) * scale;
                }
            }
        }
        return Tensor.of(matrix.shape(), out, matrix.device());
    }

    private float scale(float amax, float tensorScale) {
        if (amax == 0) {
            return 0;
        }
        if (this == MXFP4) {
            return (float) Math.scalb(1.0, Math.getExponent(amax) - 2);   // E8M0: E2M1's largest exponent is 2
        }
        float local = Fp8Format.E4M3.round(amax / MAX / tensorScale);
        return local * tensorScale;
    }
}
