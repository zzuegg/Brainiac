package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * A dense layer run with low-bit weights <em>and</em> activations (W4A4), with the
 * activations first rotated by a random Hadamard matrix as QuaRot and SpinQuant do. A few
 * activation channels of large models are hundreds of times larger than the rest, so one
 * scale per token wastes almost every level on them. Multiplying by an orthogonal matrix
 * {@code H} leaves the product unchanged, {@code x W = (x H)(H^T W)}, but spreads each
 * outlier over every channel: the rotated activations and weights are close to Gaussian and
 * quantize well. A Hadamard matrix with random signs is orthogonal, needs no data, and
 * applies in {@code O(n log n)} with the fast Walsh-Hadamard transform.
 *
 * <p>Each forward pass rotates the input rows (unless the layer has no rotation), rounds each
 * row to symmetric {@code bits}-bit levels with its own scale, and multiplies by the weight
 * rotated once and quantized per output column, by round-to-nearest or by {@link Gptq} on
 * rotated calibration inputs, as QuaRot pairs them. {@link #plain} (no rotation) and
 * {@link #smoothed} ({@link SmoothQuant} scales instead of a rotation) are the baselines in
 * the same layer. Inference only; the input size must be a power of two for the rotation.
 *
 * <p>Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" (2024),
 * <a href="https://arxiv.org/abs/2404.00456">arXiv:2404.00456</a>; Liu et al., "SpinQuant:
 * LLM Quantization with Learned Rotations" (2024),
 * <a href="https://arxiv.org/abs/2405.16406">arXiv:2405.16406</a>.
 */
public final class RotatedQuant implements Layer {

    private final int in;
    private final int out;
    private final int bits;
    private final float[] signs;
    private final double[] smooth;
    private final Tensor weight;
    private final Tensor bias;
    private final Activation activation;

    private RotatedQuant(Dense dense, int bits, float[] signs, double[] smooth, Tensor calibration) {
        Tensor w = dense.weight().value();
        this.in = w.shape().dim(0);
        this.out = w.shape().dim(1);
        this.bits = bits;
        this.signs = signs;
        this.smooth = smooth;
        this.activation = dense.activation();
        this.bias = dense.bias() == null ? null : dense.bias().value();
        float[] m = w.toArray();
        if (smooth != null) {
            for (int r = 0; r < in; r++) {
                for (int c = 0; c < out; c++) {
                    m[r * out + c] *= (float) smooth[r];
                }
            }
        }
        if (signs != null) {   // H^T W = Had (D W): rotate the columns
            float[] col = new float[in];
            for (int c = 0; c < out; c++) {
                for (int r = 0; r < in; r++) {
                    col[r] = m[r * out + c] * signs[r];
                }
                walshHadamard(col);
                for (int r = 0; r < in; r++) {
                    m[r * out + c] = col[r];
                }
            }
        }
        Tensor transformed = Tensor.of(Shape.of(in, out), m, w.device());
        this.weight = calibration == null ? Gptq.roundToNearest(transformed, bits)
                : new Gptq(bits).quantize(transformed, transformInputs(calibration));
    }

    /**
     * W4A4 with a random Hadamard rotation and round-to-nearest weights.
     *
     * @param dense the layer
     * @param bits bits for weights and activations
     * @param seed seeds the random signs
     * @return the quantized layer
     */
    public static RotatedQuant hadamard(Dense dense, int bits, long seed) {
        return new RotatedQuant(dense, bits, signs(dense, seed), null, null);
    }

    /**
     * W4A4 with a random Hadamard rotation and GPTQ weights calibrated on rotated inputs.
     *
     * @param dense the layer
     * @param bits bits for weights and activations
     * @param calibration calibration inputs {@code [samples, in]}
     * @param seed seeds the random signs
     * @return the quantized layer
     */
    public static RotatedQuant hadamardGptq(Dense dense, int bits, Tensor calibration, long seed) {
        return new RotatedQuant(dense, bits, signs(dense, seed), null, calibration);
    }

    /**
     * The baseline without rotation: per-token activations and per-column weights.
     *
     * @param dense the layer
     * @param bits bits for weights and activations
     * @return the quantized layer
     */
    public static RotatedQuant plain(Dense dense, int bits) {
        return new RotatedQuant(dense, bits, null, null, null);
    }

    /**
     * The SmoothQuant baseline: activations divided and weights multiplied by
     * {@link SmoothQuant#scales} instead of a rotation.
     *
     * @param dense the layer
     * @param bits bits for weights and activations
     * @param calibration calibration inputs {@code [samples, in]}
     * @param alpha the migration strength, 0.5 in the paper
     * @return the quantized layer
     */
    public static RotatedQuant smoothed(Dense dense, int bits, Tensor calibration, float alpha) {
        return new RotatedQuant(dense, bits, null, SmoothQuant.scales(calibration, dense.weight().value(), alpha),
                null);
    }

    private static float[] signs(Dense dense, long seed) {
        int n = dense.weight().value().shape().dim(0);
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("a Hadamard rotation needs a power-of-two input size, got " + n);
        }
        SplittableRandom r = new SplittableRandom(seed);
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            s[i] = r.nextBoolean() ? 1 : -1;
        }
        return s;
    }

    /** The normalized fast Walsh-Hadamard transform, in place. */
    static void walshHadamard(float[] x) {
        int n = x.length;
        for (int h = 1; h < n; h *= 2) {
            for (int i = 0; i < n; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    float a = x[j];
                    float b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
            }
        }
        float norm = (float) (1 / Math.sqrt(n));
        for (int i = 0; i < n; i++) {
            x[i] *= norm;
        }
    }

    /** Rows of inputs as the quantizer sees them: smoothed or rotated, not yet rounded. */
    private Tensor transformInputs(Tensor x) {
        float[] a = x.toArray();
        int rows = a.length / in;
        float[] row = new float[in];
        for (int n = 0; n < rows; n++) {
            System.arraycopy(a, n * in, row, 0, in);
            transform(row);
            System.arraycopy(row, 0, a, n * in, in);
        }
        return Tensor.of(Shape.of(rows, in), a, x.device());
    }

    private void transform(float[] row) {
        if (smooth != null) {
            for (int i = 0; i < in; i++) {
                row[i] /= (float) smooth[i];
            }
        }
        if (signs != null) {
            for (int i = 0; i < in; i++) {
                row[i] *= signs[i];
            }
            walshHadamard(row);
        }
    }

    @Override
    public Var forward(Var input) {
        Shape shape = input.shape();
        float[] a = input.value().toArray();
        int rows = a.length / in;
        int levels = (1 << (bits - 1)) - 1;
        float[] row = new float[in];
        for (int n = 0; n < rows; n++) {
            System.arraycopy(a, n * in, row, 0, in);
            transform(row);
            float amax = 0;
            for (float v : row) {
                amax = Math.max(amax, Math.abs(v));
            }
            float scale = amax == 0 ? 1 : amax / levels;
            for (int i = 0; i < in; i++) {
                a[n * in + i] = Math.max(-levels, Math.min(levels, Math.round(row[i] / scale))) * scale;
            }
        }
        Tensor x = Tensor.of(Shape.of(rows, in), a, input.value().device());
        Tensor y = x.matmul(weight);
        if (bias != null) {
            y = y.add(bias);
        }
        int[] dims = shape.toArray();
        dims[dims.length - 1] = out;
        return activation.forward(Var.of(y)).reshape(dims);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "RotatedQuant[" + in + " -> " + out + ", W" + bits + "A" + bits
                + (signs != null ? ", Hadamard" : smooth != null ? ", smoothed" : "") + "]";
    }
}
