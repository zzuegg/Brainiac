package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A variational quantum circuit, simulated exactly on a state vector of at most ten qubits,
 * as a layer: the "quantum neural network" classifier of Schuld et al. ("Circuit-centric
 * quantum classifiers", <a href="https://arxiv.org/abs/1804.00633">arXiv:1804.00633</a>) and
 * Mitarai et al. ("Quantum circuit learning", <a href="https://arxiv.org/abs/1803.00745">arXiv:1803.00745</a>).
 *
 * <ul>
 *   <li><b>Encoding</b>: angle encoding, {@code RY(x_j)} on qubit {@code j} (input feature
 *       {@code j mod inputs}); with <em>data re-uploading</em> the encoding is repeated
 *       before every ansatz layer, which raises the Fourier degree the circuit can express.</li>
 *   <li><b>Ansatz</b>: the hardware-efficient ansatz, per layer {@code RY(theta) RZ(theta)} on
 *       every qubit followed by a ring of CNOTs.</li>
 *   <li><b>Read-out</b>: the expectation {@code <Z_i>} of every qubit, in {@code [-1, 1]};
 *       the layer maps {@code [batch, inputs]} to {@code [batch, qubits]}.</li>
 *   <li><b>Gradients</b>: the <em>parameter-shift rule</em>, the one a real quantum
 *       computer uses, {@code dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2}, for the
 *       circuit weights and for the inputs (so classical layers before it train too).</li>
 *   <li><b>Shot noise</b>: with {@code shots > 0} every expectation, in the forward pass and
 *       in each shifted evaluation, is a finite-sample estimate (Gaussian approximation of the
 *       binomial, variance {@code (1 - E^2) / shots}), as on hardware.</li>
 * </ul>
 *
 * <p>The cost of the simulation is {@code 2^qubits} amplitudes per circuit, and each backward
 * pass runs two circuits per rotation gate per sample, so it is meant for toy problems.
 */
public final class QuantumCircuitLayer implements Layer {
    /** The largest supported register. */
    public static final int MAX_QUBITS = 10;

    private final int inputs;
    private final int qubits;
    private final int layers;
    private final boolean reupload;
    private final Var theta;
    private int shots;
    private final SplittableRandom noise;
    private long circuitRuns;

    private QuantumCircuitLayer(int inputs, int qubits, int layers, boolean reupload, SplittableRandom random) {
        if (qubits < 1 || qubits > MAX_QUBITS) {
            throw new IllegalArgumentException("qubits must be in 1.." + MAX_QUBITS + ", got " + qubits);
        }
        if (inputs < 1 || layers < 1) {
            throw new IllegalArgumentException("need at least one input and one layer");
        }
        this.inputs = inputs;
        this.qubits = qubits;
        this.layers = layers;
        this.reupload = reupload;
        float[] t = new float[layers * qubits * 2];
        for (int i = 0; i < t.length; i++) {
            t[i] = (float) ((random.nextDouble() * 2 - 1) * Math.PI);
        }
        this.theta = Var.param(Tensor.of(Shape.of(t.length), t));
        this.noise = new SplittableRandom(random.nextLong());
    }

    /**
     * Creates a circuit.
     *
     * @param inputs the input features
     * @param qubits the register size, 1 to {@value #MAX_QUBITS}
     * @param layers the ansatz layers
     * @param reupload whether the inputs are encoded again before every layer
     * @param seed the seed of the initial angles (uniform in {@code [-pi, pi]}) and of the shot noise
     * @return the layer
     */
    public static QuantumCircuitLayer of(int inputs, int qubits, int layers, boolean reupload, long seed) {
        return new QuantumCircuitLayer(inputs, qubits, layers, reupload, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param qubits the register size
     * @param layers the ansatz layers
     * @param reupload data re-uploading
     * @return the spec
     */
    public static LayerSpec spec(int qubits, int layers, boolean reupload) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(qubits);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new QuantumCircuitLayer(input.dim(-1), qubits, layers, reupload, random);
            }
        };
    }

    /**
     * Sets the measurement budget.
     *
     * @param n shots per expectation, 0 for the exact expectation
     * @return this layer
     */
    public QuantumCircuitLayer shots(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("shots must be >= 0");
        }
        this.shots = n;
        return this;
    }

    /** @return the number of qubits */
    public int qubits() {
        return qubits;
    }

    /** @return the circuit angles */
    public Var weights() {
        return theta;
    }

    /** @return how many circuits were simulated so far (forward and shifted evaluations) */
    public long circuitRuns() {
        return circuitRuns;
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 2 || s.dim(1) != inputs) {
            throw new IllegalArgumentException("QuantumCircuitLayer expects [batch, " + inputs + "], got " + s);
        }
        return Var.apply(new CircuitOp(), input, theta);
    }

    @Override
    public List<Var> parameters() {
        return List.of(theta);
    }

    @Override
    public int outputSize(int inputSize) {
        return qubits;
    }

    /** Rotation gates in circuit order; each reads an input feature or a weight. */
    private int gates() {
        return layers * qubits * (reupload ? 3 : 2) + (reupload ? 0 : qubits);
    }

    /**
     * Runs the circuit for one sample.
     *
     * @param x the inputs
     * @param xOffset where this sample's inputs start in {@code x}
     * @param t the weights
     * @param shifted the rotation gate to shift, or -1
     * @param shift the shift
     * @param out receives {@code <Z_i>}
     */
    private void run(float[] x, int xOffset, float[] t, int shifted, double shift, double[] out) {
        circuitRuns++;
        int dim = 1 << qubits;
        double[] re = new double[dim];
        double[] im = new double[dim];
        re[0] = 1;
        int gate = 0;
        int w = 0;
        for (int l = 0; l < layers; l++) {
            if (l == 0 || reupload) {
                for (int q = 0; q < qubits; q++) {
                    double a = x[xOffset + q % inputs] + (gate == shifted ? shift : 0);
                    ry(re, im, q, a);
                    gate++;
                }
            }
            for (int q = 0; q < qubits; q++) {
                ry(re, im, q, t[w++] + (gate == shifted ? shift : 0));
                gate++;
                rz(re, im, q, t[w++] + (gate == shifted ? shift : 0));
                gate++;
            }
            if (qubits > 1) {
                for (int q = 0; q < qubits; q++) {
                    int target = (q + 1) % qubits;
                    if (qubits == 2 && q == 1) {
                        break;
                    }
                    cnot(re, im, q, target);
                }
            }
        }
        for (int q = 0; q < qubits; q++) {
            double z = 0;
            for (int i = 0; i < dim; i++) {
                double p = re[i] * re[i] + im[i] * im[i];
                z += ((i >> q) & 1) == 0 ? p : -p;
            }
            if (shots > 0) {
                double sd = Math.sqrt(Math.max(0, 1 - z * z) / shots);
                z = Math.max(-1, Math.min(1, z + noise.nextGaussian() * sd));
            }
            out[q] = z;
        }
    }

    /** Which input feature (>= 0) or weight (encoded as -1 - index) each rotation gate reads. */
    private int[] gateSources() {
        int[] src = new int[gates()];
        int gate = 0;
        int w = 0;
        for (int l = 0; l < layers; l++) {
            if (l == 0 || reupload) {
                for (int q = 0; q < qubits; q++) {
                    src[gate++] = q % inputs;
                }
            }
            for (int q = 0; q < qubits; q++) {
                src[gate++] = -1 - w++;
                src[gate++] = -1 - w++;
            }
        }
        return src;
    }

    private static void ry(double[] re, double[] im, int q, double angle) {
        double c = Math.cos(angle / 2);
        double s = Math.sin(angle / 2);
        int bit = 1 << q;
        for (int i = 0; i < re.length; i++) {
            if ((i & bit) == 0) {
                int j = i | bit;
                double r0 = re[i];
                double i0 = im[i];
                double r1 = re[j];
                double i1 = im[j];
                re[i] = c * r0 - s * r1;
                im[i] = c * i0 - s * i1;
                re[j] = s * r0 + c * r1;
                im[j] = s * i0 + c * i1;
            }
        }
    }

    private static void rz(double[] re, double[] im, int q, double angle) {
        double c = Math.cos(angle / 2);
        double s = Math.sin(angle / 2);
        int bit = 1 << q;
        for (int i = 0; i < re.length; i++) {
            double r = re[i];
            double m = im[i];
            if ((i & bit) == 0) {   // multiply by exp(-i a / 2)
                re[i] = c * r + s * m;
                im[i] = c * m - s * r;
            } else {                // multiply by exp(+i a / 2)
                re[i] = c * r - s * m;
                im[i] = c * m + s * r;
            }
        }
    }

    private static void cnot(double[] re, double[] im, int control, int target) {
        int cb = 1 << control;
        int tb = 1 << target;
        for (int i = 0; i < re.length; i++) {
            if ((i & cb) != 0 && (i & tb) == 0) {
                int j = i | tb;
                double r = re[i];
                double m = im[i];
                re[i] = re[j];
                im[i] = im[j];
                re[j] = r;
                im[j] = m;
            }
        }
    }

    /** The circuit as an autograd op with parameter-shift gradients. */
    private final class CircuitOp implements Op {
        @Override
        public Tensor forward(Tensor... in) {
            float[] x = in[0].toArray();
            float[] t = in[1].toArray();
            int batch = in[0].shape().dim(0);
            float[] out = new float[batch * qubits];
            double[] e = new double[qubits];
            for (int b = 0; b < batch; b++) {
                run(x, b * inputs, t, -1, 0, e);
                for (int q = 0; q < qubits; q++) {
                    out[b * qubits + q] = (float) e[q];
                }
            }
            return Tensor.of(Shape.of(batch, qubits), out);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
            float[] g = grad.toArray();
            float[] x = in[0].toArray();
            float[] t = in[1].toArray();
            int batch = in[0].shape().dim(0);
            float[] gx = new float[x.length];
            float[] gt = new float[t.length];
            int[] src = gateSources();
            double[] plus = new double[qubits];
            double[] minus = new double[qubits];
            for (int b = 0; b < batch; b++) {
                for (int gate = 0; gate < src.length; gate++) {
                    run(x, b * inputs, t, gate, Math.PI / 2, plus);
                    run(x, b * inputs, t, gate, -Math.PI / 2, minus);
                    double d = 0;
                    for (int q = 0; q < qubits; q++) {
                        d += g[b * qubits + q] * (plus[q] - minus[q]) / 2;
                    }
                    if (src[gate] >= 0) {
                        gx[b * inputs + src[gate]] += (float) d;
                    } else {
                        gt[-1 - src[gate]] += (float) d;
                    }
                }
            }
            return new Tensor[] {Tensor.of(in[0].shape(), gx), Tensor.of(in[1].shape(), gt)};
        }

        @Override
        public String name() {
            return "quantumCircuit";
        }
    }
}
