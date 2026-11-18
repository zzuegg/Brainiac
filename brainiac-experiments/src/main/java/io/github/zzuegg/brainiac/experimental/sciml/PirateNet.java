package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * PirateNet (Wang, Li, Chen and Perdikaris, 2024): a residual coordinate network that makes
 * deep physics-informed networks trainable. Coordinates pass through random Fourier features
 * {@code phi(x) = [cos(B x), sin(B x)]}; two gates {@code U = tanh(phi W_u)},
 * {@code V = tanh(phi W_v)} are mixed into each block,
 *
 * <pre>
 * f = tanh(x_l W1),  z1 = f * U + (1 - f) * V
 * g = tanh(z1 W2),   z2 = g * U + (1 - g) * V
 * h = tanh(z2 W3),   x_{l+1} = alpha_l h + (1 - alpha_l) x_l
 * </pre>
 *
 * <p>with a trainable {@code alpha_l} that starts at 0, so the network starts as a linear
 * model on the Fourier features and grows its depth during training. {@link #fitOutput}
 * initialises the last layer by least squares on the embedding, for example to fit the
 * initial condition. The class is a {@link Layer} from {@code [n, d]} coordinates to
 * {@code [n, outputs]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2402.00326">arXiv:2402.00326</a>.
 */
public final class PirateNet implements Layer {

    private final int inputs;
    private final int outputs;
    private final int width;
    private final Tensor frequencies;
    private final Var gateU;
    private final Var gateV;
    private final List<Var[]> blocks = new ArrayList<>();
    private final Var output;

    private PirateNet(int inputs, int outputs, int width, int depth, double scale, SplittableRandom random) {
        if (width % 2 != 0) {
            throw new IllegalArgumentException("width must be even");
        }
        this.inputs = inputs;
        this.outputs = outputs;
        this.width = width;
        this.frequencies = Tensor.normal(Shape.of(inputs, width / 2), 0, (float) scale, random);
        this.gateU = glorot(width, width, random);
        this.gateV = glorot(width, width, random);
        for (int i = 0; i < depth; i++) {
            blocks.add(new Var[] {glorot(width, width, random), glorot(width, width, random),
                glorot(width, width, random), Var.param(Tensor.zeros(Shape.of(1)))});
        }
        this.output = Var.param(Tensor.zeros(Shape.of(width, outputs)));
    }

    private static Var glorot(int in, int out, SplittableRandom random) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(2.0 / (in + out)), random));
    }

    /**
     * Creates the network.
     *
     * @param inputs the coordinates
     * @param outputs the field components
     * @param width the hidden width (even; also the Fourier feature count)
     * @param depth the residual blocks (three dense layers each)
     * @param scale the Fourier frequency scale, for example 1 to 2 on unit domains
     * @param seed the seed
     * @return the network
     */
    public static PirateNet of(int inputs, int outputs, int width, int depth, double scale, long seed) {
        return new PirateNet(inputs, outputs, width, depth, scale, new SplittableRandom(seed));
    }

    /**
     * Describes the network for model builders.
     *
     * @param outputs the field components
     * @param width the hidden width
     * @param depth the residual blocks
     * @param scale the Fourier frequency scale
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int width, int depth, double scale) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PirateNet(input.dim(-1), outputs, width, depth, scale, random);
            }
        };
    }

    /**
     * The Fourier embedding {@code [cos(x B), sin(x B)]}.
     *
     * @param x coordinates {@code [n, d]}
     * @return {@code [n, width]}
     */
    public Var embed(Var x) {
        Var p = x.matmul(Var.of(frequencies));
        return Var.concat(1, p.cos(), p.sin());
    }

    @Override
    public Var forward(Var x) {
        Var phi = embed(x);
        Var u = phi.matmul(gateU).tanh();
        Var v = phi.matmul(gateV).tanh();
        Var h = phi;
        for (Var[] b : blocks) {
            Var f = h.matmul(b[0]).tanh();
            Var z1 = f.mul(u).add(f.neg().add(1f).mul(v));
            Var g = z1.matmul(b[1]).tanh();
            Var z2 = g.mul(u).add(g.neg().add(1f).mul(v));
            Var r = z2.matmul(b[2]).tanh();
            h = r.mul(b[3]).add(h.mul(b[3].neg().add(1f)));
        }
        return h.matmul(output);
    }

    /**
     * Initialises the output layer by least squares, valid while every gate {@code alpha} is
     * still 0 (the network output is then linear in the embedding).
     *
     * @param points coordinates {@code [m][d]}, such as the initial-condition points
     * @param values the target of the first output at them
     */
    public void fitOutput(double[][] points, double[] values) {
        Tensor phi = Var.noGrad(() -> embed(Var.of(CollocationSampler.tensor(points)))).value();
        // in double precision with a small ridge: the embedding columns are often nearly collinear
        Tensor phi64 = Tensor.of(phi.shape(), phi.toDoubleArray(), DType.FLOAT64, Device.cpu());
        Tensor y64 = Tensor.of(Shape.of(values.length, 1), values.clone(), DType.FLOAT64, Device.cpu());
        double[] col = Linalg.lstsq(phi64, y64, 1e-4).toDoubleArray();
        float[] full = new float[width * outputs];
        for (int i = 0; i < width; i++) {
            full[i * outputs] = (float) col[i];
        }
        output.assign(Tensor.of(Shape.of(width, outputs), full));
    }

    /** @return the current gate value {@code alpha_l} of every block */
    public double[] alphas() {
        double[] a = new double[blocks.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = blocks.get(i)[3].value().item();
        }
        return a;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(gateU, gateV));
        blocks.forEach(b -> p.addAll(List.of(b)));
        p.add(output);
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "PirateNet(" + inputs + " -> " + outputs + ", width=" + width + ", blocks=" + blocks.size() + ")";
    }
}
