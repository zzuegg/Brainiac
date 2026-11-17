package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A neural ODE layer: depth as continuous time. The output is the solution at time
 * {@code T} of {@code dz/dt = f(t, z; theta)} from {@code z(0) = input}, where {@code f} is a
 * small MLP {@code tanh([z, t] W1 + b1) W2 + b2}. Any {@link OdeSolver} can integrate it.
 *
 * <p>Two ways to get gradients:
 * <ul>
 *   <li>{@link Gradient#DIRECT}: backpropagate through the solver's operations; exact for the
 *       discretized solution, memory grows with the number of steps;</li>
 *   <li>{@link Gradient#ADJOINT}: the adjoint method. The forward solve keeps no graph; the
 *       backward pass integrates the augmented system
 *       {@code dz/dt = f, da/dt = -a df/dz, da_theta/dt = -a df/dtheta} from {@code T} back to 0
 *       with the same solver, with the vector-Jacobian products computed by a nested autograd
 *       pass on fresh copies of the state and parameters. Memory is constant in the number of
 *       steps; the gradient matches the direct one up to the solver error.</li>
 * </ul>
 *
 * <p>Input and output are {@code [batch, features]}.
 *
 * <p>Paper: Chen et al., "Neural Ordinary Differential Equations",
 * <a href="https://arxiv.org/abs/1806.07366">arXiv:1806.07366</a>.
 */
public final class NeuralOde implements Layer {

    /** How gradients are computed. */
    public enum Gradient {
        /** Backpropagation through the solver. */
        DIRECT,
        /** The continuous adjoint method. */
        ADJOINT
    }

    private final int features;
    private final OdeSolver solver;
    private final float duration;
    private final Gradient gradient;
    private final List<Var> params;

    /**
     * Creates the layer.
     *
     * @param features the state width
     * @param hidden the hidden width of the dynamics MLP
     * @param solver the solver
     * @param duration the integration time {@code T}
     * @param gradient direct or adjoint gradients
     * @param random the random source
     */
    public NeuralOde(int features, int hidden, OdeSolver solver, float duration, Gradient gradient,
            RandomGenerator random) {
        this.features = features;
        this.solver = solver;
        this.duration = duration;
        this.gradient = gradient;
        this.params = List.of(MixerOps.weight(random, features + 1, hidden), MixerOps.filled(0f, hidden),
                MixerOps.weight(random, hidden, features), MixerOps.filled(0f, features));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param hidden the hidden width of the dynamics
     * @param solver the solver
     * @param duration the integration time
     * @param gradient direct or adjoint gradients
     * @return the spec
     */
    public static LayerSpec of(int hidden, OdeSolver solver, float duration, Gradient gradient) {
        return MixerOps.sameShape((in, r) -> new NeuralOde(in.dim(-1), hidden, solver, duration, gradient, r));
    }

    /** @return the solver, for its step and evaluation counts */
    public OdeSolver solver() {
        return solver;
    }

    /** The dynamics with explicit parameters. */
    private static Var dynamics(float t, Var z, List<Var> p) {
        int b = z.shape().dim(0);
        Var zt = Var.concat(1, z, Var.of(Tensor.full(Shape.of(b, 1), t)));
        return zt.matmul(p.get(0)).add(p.get(1)).tanh().matmul(p.get(2)).add(p.get(3));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != features) {
            throw new IllegalArgumentException("NeuralOde expects [batch, " + features + "], got " + input.shape());
        }
        if (gradient == Gradient.DIRECT) {
            return solver.integrate((t, z) -> dynamics(t, z, params), input, 0, duration);
        }
        Var[] inputs = new Var[1 + params.size()];
        inputs[0] = input;
        for (int i = 0; i < params.size(); i++) {
            inputs[i + 1] = params.get(i);
        }
        return Var.apply(new Adjoint(), inputs);
    }

    /** The adjoint method as a custom op over (z0, params...). */
    private final class Adjoint implements Op {

        @Override
        public Tensor forward(Tensor... in) {
            List<Var> fixed = new ArrayList<>();
            for (int i = 1; i < in.length; i++) {
                fixed.add(Var.of(in[i]));
            }
            return solver.integrate((t, z) -> dynamics(t, z, fixed), Var.of(in[0]), 0, duration).value();
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
            int n = in.length - 1;
            int b = in[0].shape().dim(0);
            // augmented state [z | a | a_theta (flattened)], integrated backwards
            int[] sizes = new int[n];
            int total = 0;
            for (int i = 0; i < n; i++) {
                sizes[i] = Math.toIntExact(in[i + 1].size());
                total += sizes[i];
            }
            int thetaSize = total;
            int zSize = b * features;
            float[] start = new float[2 * zSize + thetaSize];
            System.arraycopy(output.toArray(), 0, start, 0, zSize);
            System.arraycopy(grad.toArray(), 0, start, zSize, zSize);
            Var augmented = Var.of(Tensor.of(Shape.of(start.length), start));
            OdeSolver.Dynamics aug = (t, s) -> {
                float[] v = s.value().toArray();
                Var z = Var.param(Tensor.of(Shape.of(b, features), java.util.Arrays.copyOfRange(v, 0, zSize)));
                Tensor a = Tensor.of(Shape.of(b, features), java.util.Arrays.copyOfRange(v, zSize, 2 * zSize));
                List<Var> p = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    p.add(Var.param(in[i + 1]));
                }
                Var f = dynamics(t, z, p);
                f.backward(a);
                float[] out = new float[v.length];
                System.arraycopy(f.value().toArray(), 0, out, 0, zSize);
                float[] dz = z.grad().toArray();
                for (int i = 0; i < zSize; i++) {
                    out[zSize + i] = -dz[i];
                }
                int offset = 2 * zSize;
                for (int i = 0; i < n; i++) {
                    Tensor g = p.get(i).grad();
                    float[] gv = g == null ? new float[sizes[i]] : g.toArray();
                    for (int j = 0; j < sizes[i]; j++) {
                        out[offset + j] = -gv[j];
                    }
                    offset += sizes[i];
                }
                return Var.of(Tensor.of(Shape.of(out.length), out));
            };
            float[] end = solver.integrate(aug, augmented, duration, 0).value().toArray();
            Tensor[] grads = new Tensor[in.length];
            grads[0] = Tensor.of(in[0].shape(), java.util.Arrays.copyOfRange(end, zSize, 2 * zSize));
            int offset = 2 * zSize;
            for (int i = 0; i < n; i++) {
                grads[i + 1] = Tensor.of(in[i + 1].shape(), java.util.Arrays.copyOfRange(end, offset, offset + sizes[i]));
                offset += sizes[i];
            }
            return grads;
        }

        @Override
        public String name() {
            return "neuralOdeAdjoint";
        }
    }

    @Override
    public List<Var> parameters() {
        return params;
    }

    @Override
    public String toString() {
        return "NeuralOde(" + features + ", " + solver.method() + ", T=" + duration + ", " + gradient + ")";
    }
}
