package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * A dense layer evaluated by <em>stochastic computing</em> (Gaines 1969; survey: Alaghi and
 * Hayes, "Survey of stochastic computing", ACM TECS 2013,
 * <a href="https://dl.acm.org/doi/10.1145/2465787.2465794">10.1145/2465787.2465794</a>): every
 * number is a random bitstream of length {@code L} whose share of ones encodes it, so
 * arithmetic needs only single gates.
 *
 * <ul>
 *   <li><b>Bipolar encoding</b>: {@code v in [-1, 1]} is a stream with {@code P(1) = (v + 1) / 2}.</li>
 *   <li><b>Multiplication</b> is one XNOR gate per bit: the XNOR of independent bipolar streams
 *       of {@code a} and {@code b} encodes {@code a * b}.</li>
 *   <li><b>Addition</b>, either scaled by a multiplexer ({@link Adder#MUX}: each clock a
 *       random select line passes one product bit, so the output encodes the <em>mean</em> of
 *       the products, and the {@code 1/n} scale costs precision), or by an accumulative parallel
 *       counter ({@link Adder#APC}: the ones of all product streams are counted every clock,
 *       which is exact up to the stream noise and costs a small binary adder tree).</li>
 * </ul>
 *
 * <p>The weights (and the bias, as an extra input fixed at 1) are scaled by their largest
 * magnitude into {@code [-1, 1]} and the inputs are clipped to {@code [-1, 1]}, so it is meant
 * for converting a trained layer with bounded (for example tanh) activations. The error falls
 * like {@code 1/sqrt(L)}: {@link #bitOperations} counts the gate evaluations, the "energy" that
 * stream length buys accuracy with. Inference only; the output carries no gradient. Streams are
 * generated with 64 bits per word, so the products are word-wide XNORs and popcounts.
 */
public final class StochasticBitstreamDense implements Layer {

    /** How product streams are summed. */
    public enum Adder {
        /** a multiplexer with a random select line: scaled addition, the classic SC adder */
        MUX,
        /** an accumulative parallel counter: counts ones across inputs every clock */
        APC
    }

    private final int inputs;
    private final int units;
    private final float[] weights;     // [inputs + 1, units] scaled into [-1, 1], last row the bias
    private final float scale;
    private final int length;
    private final Adder adder;
    private final SplittableRandom random;
    private long bitOps;

    private StochasticBitstreamDense(float[] w, float[] b, int inputs, int units, int length, Adder adder, long seed) {
        if (length < 1) {
            throw new IllegalArgumentException("stream length must be positive");
        }
        this.inputs = inputs;
        this.units = units;
        this.length = length;
        this.adder = adder;
        this.random = new SplittableRandom(seed);
        float max = 1e-8f;
        for (float v : w) {
            max = Math.max(max, Math.abs(v));
        }
        for (float v : b) {
            max = Math.max(max, Math.abs(v));
        }
        this.scale = max;
        this.weights = new float[(inputs + 1) * units];
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < units; j++) {
                weights[i * units + j] = w[i * units + j] / max;
            }
        }
        for (int j = 0; j < units; j++) {
            weights[inputs * units + j] = b[j] / max;
        }
    }

    /**
     * Converts trained dense weights.
     *
     * @param weight {@code [inputs, units]}
     * @param bias {@code [units]}, or {@code null}
     * @param length the stream length {@code L} in bits
     * @param adder the adder circuit
     * @param seed the seed of the stream generators
     * @return the layer
     */
    public static StochasticBitstreamDense of(Tensor weight, Tensor bias, int length, Adder adder, long seed) {
        int in = weight.shape().dim(0);
        int out = weight.shape().dim(1);
        float[] b = bias == null ? new float[out] : bias.toArray();
        return new StochasticBitstreamDense(weight.toArray(), b, in, out, length, adder, seed);
    }

    /** @return the stream length in bits */
    public int length() {
        return length;
    }

    /** @return gate evaluations (XNOR products plus adder bits) since creation */
    public long bitOperations() {
        return bitOps;
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 2 || s.dim(1) != inputs) {
            throw new IllegalArgumentException("StochasticBitstreamDense expects [batch, " + inputs + "], got " + s);
        }
        int batch = s.dim(0);
        float[] x = input.value().toArray();
        int words = (length + 63) >>> 6;
        int n = inputs + 1;
        // weight streams, generated once per forward pass (one SNG per weight)
        long[][] w = new long[n * units][];
        for (int k = 0; k < n * units; k++) {
            w[k] = stream(weights[k], words);
        }
        float[] out = new float[batch * units];
        long[][] xs = new long[n][];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < inputs; i++) {
                xs[i] = stream(Math.max(-1, Math.min(1, x[b * inputs + i])), words);
            }
            xs[inputs] = stream(1, words);
            for (int j = 0; j < units; j++) {
                double value;
                if (adder == Adder.APC) {
                    long ones = 0;
                    for (int i = 0; i < n; i++) {
                        long[] xi = xs[i];
                        long[] wij = w[i * units + j];
                        for (int t = 0; t < words; t++) {
                            ones += Long.bitCount(~(xi[t] ^ wij[t]) & mask(t, words));
                        }
                    }
                    value = 2.0 * ones / length - n;           // sum of the n bipolar products
                    bitOps += (long) n * length * 2;
                } else {
                    long ones = 0;
                    for (int t = 0; t < length; t++) {
                        int i = random.nextInt(n);
                        long bit = ~(xs[i][t >>> 6] ^ w[i * units + j][t >>> 6]) >>> (t & 63) & 1L;
                        ones += bit;
                    }
                    value = (2.0 * ones / length - 1) * n;     // the mux output is the mean, rescaled
                    bitOps += (long) length * 2;
                }
                out[b * units + j] = (float) (value * scale);
            }
        }
        return Var.of(Tensor.of(Shape.of(batch, units), out));
    }

    private long mask(int word, int words) {
        int rem = length & 63;
        return word == words - 1 && rem != 0 ? (1L << rem) - 1 : -1L;
    }

    /** A bipolar stream: each bit is one with probability {@code (v + 1) / 2}. */
    private long[] stream(float v, int words) {
        double p = (v + 1) / 2;
        long[] s = new long[words];
        for (int t = 0; t < length; t++) {
            if (random.nextDouble() < p) {
                s[t >>> 6] |= 1L << (t & 63);
            }
        }
        return s;
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }
}
