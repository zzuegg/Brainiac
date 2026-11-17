package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A neural controlled differential equation: a continuous-time RNN driven by the input path.
 * The observations are joined into a {@link ControlPath} {@code X(s)} and the hidden state
 * follows
 *
 * <pre>
 * z(0) = X(0) W_init,     dz/ds = f(z) dX/ds,     f(z) = tanh(z W1 + b1) W2   (a hidden x channels matrix)
 * </pre>
 *
 * <p>so new information enters exactly as fast as the path changes. Put the observation
 * time into one channel and irregularly sampled, per-sample timed series need no
 * imputation or binning: the CDE is the continuous analogue of an RNN and handles messy
 * time series natively. Solved with {@link OdeSolver} steps inside every interval between
 * knots (the derivative of a linear path jumps at knots), with direct backpropagation.
 *
 * <p>Input {@code [batch, points, channels]}, output the final state {@code [batch, hidden]}.
 *
 * <p>Paper: Kidger et al., "Neural Controlled Differential Equations for Irregular Time
 * Series", <a href="https://arxiv.org/abs/2005.08926">arXiv:2005.08926</a>.
 */
public final class NeuralCde implements Layer {

    private final int channels;
    private final int hidden;
    private final ControlPath.Interpolation interpolation;
    private final int stepsPerInterval;
    private final Var init;
    private final Var w1;
    private final Var b1;
    private final Var w2;

    /**
     * Creates the layer.
     *
     * @param channels the input channels (include time as one of them)
     * @param hidden the hidden state width
     * @param width the hidden width of the vector field network
     * @param interpolation how observations are joined
     * @param stepsPerInterval RK4 steps between consecutive knots
     * @param random the random source
     */
    public NeuralCde(int channels, int hidden, int width, ControlPath.Interpolation interpolation,
            int stepsPerInterval, RandomGenerator random) {
        this.channels = channels;
        this.hidden = hidden;
        this.interpolation = interpolation;
        this.stepsPerInterval = stepsPerInterval;
        this.init = MixerOps.weight(random, channels, hidden);
        this.w1 = MixerOps.weight(random, hidden, width);
        this.b1 = MixerOps.filled(0f, width);
        this.w2 = MixerOps.uniform(random, (float) (0.5 / Math.sqrt(width)), width, hidden * channels);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param hidden the state width
     * @param width the vector field width
     * @param interpolation how observations are joined
     * @param stepsPerInterval solver steps per interval
     * @return the spec
     */
    public static LayerSpec of(int hidden, int width, ControlPath.Interpolation interpolation, int stepsPerInterval) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(hidden);
            }

            @Override
            public Layer create(Shape input, java.util.SplittableRandom random) {
                return new NeuralCde(input.dim(-1), hidden, width, interpolation, stepsPerInterval, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != channels) {
            throw new IllegalArgumentException("NeuralCde expects [batch, points, " + channels + "], got " + s);
        }
        int b = s.dim(0);
        ControlPath path = new ControlPath(input.value(), interpolation);
        Var z = Var.of(path.value(0)).matmul(init);
        OdeSolver solver = OdeSolver.rk4(stepsPerInterval);
        OdeSolver.Dynamics field = (t, state) -> {
            Var f = state.matmul(w1).add(b1).tanh().matmul(w2).reshape(b, hidden, channels);
            Var dx = Var.of(path.derivative(t)).reshape(b, channels, 1);
            return f.matmul(dx).reshape(b, hidden);
        };
        for (int i = 0; i + 1 < path.points(); i++) {
            // integrate inside the interval so the solver never straddles a knot
            float lo = i + 1e-4f;
            float hi = i + 1 - 1e-4f;
            z = solver.integrate(field, z, lo, hi);
        }
        return z;
    }

    @Override
    public List<Var> parameters() {
        return List.of(init, w1, b1, w2);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(hidden);
    }

    @Override
    public String toString() {
        return "NeuralCde(" + channels + " -> " + hidden + ", " + interpolation + ")";
    }
}
