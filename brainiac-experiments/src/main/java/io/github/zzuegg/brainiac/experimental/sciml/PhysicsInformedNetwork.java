package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A physics-informed neural network (Raissi, Perdikaris and Karniadakis, 2019): a coordinate
 * network {@code u_theta(x, t)} trained so that a PDE residual, computed by automatic
 * differentiation at collocation points, vanishes, together with initial- and
 * boundary-condition losses. No solution data is needed.
 *
 * <p>The field network is any {@link Layer} from coordinates {@code [n, d]} to values
 * {@code [n, outputs]} whose rows do not interact (an MLP, {@link PirateNet},
 * {@link PhysicsInformedKan}); this class is itself a {@code Layer} that forwards to it, so it
 * drops into code that expects one. {@link #derivatives} differentiates the field with the
 * core higher-order autograd ({@code Var.gradients(..., createGraph = true)}), so a residual
 * such as Burgers' {@code u_t + u u_x - nu u_xx} is written as a lambda:
 *
 * <pre>{@code
 * PhysicsInformedNetwork pinn = PhysicsInformedNetwork.of(PhysicsInformedNetwork.tanhMlp(2, 32, 3, 1, 7));
 * Residual burgers = d -> d.first(1).add(d.u().mul(d.first(0))).sub(d.second(0).mul(nu));
 * Var loss = pinn.residual(points, burgers).square().mean();
 * }</pre>
 *
 * <p>Training recipes (Adam then L-BFGS, and the fixes of {@link PinnTraining}) live in
 * {@link PinnTraining}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1711.10561">arXiv:1711.10561</a>.
 */
public final class PhysicsInformedNetwork implements Layer {

    /** A PDE residual written from the field's derivatives. */
    @FunctionalInterface
    public interface Residual {

        /**
         * @param d the field and its derivatives at the points
         * @return the residual per point, {@code [n, 1]} (or {@code [n, k]} for systems)
         */
        Var apply(Derivatives d);
    }

    /**
     * The field and its first and second partial derivatives at a set of points, computed
     * lazily and cached; every value stays differentiable with respect to the parameters.
     */
    public static final class Derivatives {

        private final Var x;
        private final Var u;
        private final int inputs;
        private final Map<Integer, Var> gradients = new HashMap<>();
        private final Map<Long, Var> seconds = new HashMap<>();

        Derivatives(Var x, Var u) {
            this.x = x;
            this.u = u;
            this.inputs = x.shape().dim(1);
        }

        /** @return the points as a differentiable input, {@code [n, d]} */
        public Var x() {
            return x;
        }

        /**
         * @param axis a coordinate
         * @return that coordinate of every point, {@code [n, 1]}
         */
        public Var coordinate(int axis) {
            return x.slice(1, axis, axis + 1);
        }

        /** @return the first output, {@code [n, 1]} */
        public Var u() {
            return u(0);
        }

        /**
         * @param output an output channel
         * @return that output, {@code [n, 1]}
         */
        public Var u(int output) {
            return u.slice(1, output, output + 1);
        }

        /** @return all outputs, {@code [n, outputs]} */
        public Var all() {
            return u;
        }

        private Var gradient(int output) {
            return gradients.computeIfAbsent(output,
                    o -> Var.gradients(u(o).sum(), List.of(x), true).get(0));
        }

        /**
         * @param axis a coordinate
         * @return {@code du/dx_axis} of the first output, {@code [n, 1]}
         */
        public Var first(int axis) {
            return first(axis, 0);
        }

        /**
         * @param axis a coordinate
         * @param output an output channel
         * @return {@code du_output/dx_axis}, {@code [n, 1]}
         */
        public Var first(int axis, int output) {
            return gradient(output).slice(1, axis, axis + 1);
        }

        /**
         * @param axis a coordinate
         * @return {@code d2u/dx_axis^2} of the first output
         */
        public Var second(int axis) {
            return second(axis, axis, 0);
        }

        /**
         * The mixed second derivative {@code d2 u_output / dx_a dx_b}.
         *
         * @param a the first coordinate
         * @param b the second coordinate
         * @param output an output channel
         * @return {@code [n, 1]}
         */
        public Var second(int a, int b, int output) {
            Var row = seconds.computeIfAbsent((long) output * inputs + a,
                    k -> Var.gradients(first(a, output).sum(), List.of(x), true).get(0));
            return row.slice(1, b, b + 1);
        }

        /**
         * A third derivative along one axis, for dispersive equations such as KdV.
         *
         * @param axis the coordinate
         * @return {@code d3u/dx_axis^3} of the first output
         */
        public Var third(int axis) {
            return Var.gradients(second(axis).sum(), List.of(x), true).get(0).slice(1, axis, axis + 1);
        }
    }

    private final Layer field;

    private PhysicsInformedNetwork(Layer field) {
        this.field = field;
    }

    /**
     * Wraps a field network.
     *
     * @param field coordinates {@code [n, d]} to values {@code [n, outputs]}, row by row
     * @return the PINN
     */
    public static PhysicsInformedNetwork of(Layer field) {
        return new PhysicsInformedNetwork(field);
    }

    /**
     * A tanh MLP field, the standard PINN network (Glorot-normal initialisation).
     *
     * @param inputs the coordinates
     * @param width the hidden width
     * @param depth the hidden layers
     * @param outputs the field components
     * @param seed the seed
     * @return the network, as a layer
     */
    public static Layer tanhMlp(int inputs, int width, int depth, int outputs, long seed) {
        int[] sizes = new int[depth + 2];
        sizes[0] = inputs;
        for (int i = 1; i <= depth; i++) {
            sizes[i] = width;
        }
        sizes[depth + 1] = outputs;
        SimpleMlp mlp = SimpleMlp.tanh(seed, sizes);
        return new Layer() {
            @Override
            public Var forward(Var input) {
                return mlp.forward(input);
            }

            @Override
            public List<Var> parameters() {
                return mlp.parameters();
            }

            @Override
            public int outputSize(int inputSize) {
                return outputs;
            }

            @Override
            public String toString() {
                return "tanhMlp" + java.util.Arrays.toString(sizes);
            }
        };
    }

    /** @return the field network */
    public Layer field() {
        return field;
    }

    @Override
    public Var forward(Var input) {
        return field.forward(input);
    }

    @Override
    public List<Var> parameters() {
        return field.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return field.outputSize(inputSize);
    }

    /**
     * The field at points as a differentiable function of them.
     *
     * @param points {@code [n, d]}
     * @return the derivatives, computed on demand
     */
    public Derivatives derivatives(Tensor points) {
        Var x = Var.param(points);
        return new Derivatives(x, field.forward(x));
    }

    /**
     * The residual at collocation points.
     *
     * @param points {@code [n, d]}
     * @param residual the PDE residual
     * @return the residual, {@code [n, k]}
     */
    public Var residual(Tensor points, Residual residual) {
        return residual.apply(derivatives(points));
    }

    /**
     * Predictions without a graph.
     *
     * @param points {@code [n, d]}
     * @return the field's first output at each point
     */
    public double[] predict(double[][] points) {
        int n = points.length;
        int d = points[0].length;
        float[] flat = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                flat[i * d + j] = (float) points[i][j];
            }
        }
        Tensor out = Var.noGrad(() -> field.forward(Var.of(Tensor.of(Shape.of(n, d), flat)))).value();
        int k = out.shape().dim(1);
        float[] v = out.toArray();
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            r[i] = v[i * k];
        }
        return r;
    }

    @Override
    public String toString() {
        return "PhysicsInformedNetwork(" + field + ")";
    }
}
