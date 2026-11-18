package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * A universal differential equation (Rackauckas et al., 2020): the right-hand side is the
 * known part of the physics plus a neural network for the unknown part,
 * {@code x' = f_known(x; p) + NN_theta(x)}, and both the physical parameters {@code p} and
 * the network are trained through the ODE solver (the experimental {@link OdeSolver}, the same
 * one {@code NeuralOde} uses). Because the network only has to learn what the physics misses,
 * far less data is needed than for a pure neural ODE; afterwards the learned term can be passed
 * to {@link Sindy} to recover a closed form ({@link #networkTerm}).
 *
 * <p>With no known part it is a plain neural ODE; with no network it fits the physics alone.
 * As a {@link Layer}, {@code forward} advances states {@code [b, d]} by the configured step.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2001.04385">arXiv:2001.04385</a>.
 */
public final class UniversalDifferentialEquation implements Layer {

    /** The known physics, differentiable in the state and its parameters. */
    @FunctionalInterface
    public interface KnownPhysics {

        /**
         * @param x states {@code [b, d]}
         * @param parameters the physical parameters, each {@code [1]}
         * @return the known part of {@code x'}, {@code [b, d]}
         */
        Var apply(Var x, List<Var> parameters);
    }

    private final int dimension;
    private final KnownPhysics known;
    private final List<Var> physical = new ArrayList<>();
    private final SimpleMlp network;
    private final OdeSolver solver;
    private final float step;
    private final float scale;

    private UniversalDifferentialEquation(int dimension, KnownPhysics known, float[] parameters, int width,
            OdeSolver solver, float step, float scale, long seed) {
        this.dimension = dimension;
        this.known = known;
        for (float p : parameters) {
            physical.add(Var.param(Tensor.of(Shape.of(1), new float[] {p})));
        }
        this.network = width > 0 ? SimpleMlp.tanh(seed, dimension, width, width, dimension) : null;
        if (network != null) {
            Var last = network.weight(network.depth() - 1);
            last.assign(last.value().mul(0.1f));
        }
        this.solver = solver;
        this.step = step;
        this.scale = scale;
    }

    /**
     * Creates a UDE.
     *
     * @param dimension the state size
     * @param known the known physics, or {@code null} for a pure neural ODE
     * @param parameters initial physical parameters
     * @param width the network width, 0 for physics only
     * @param solver the ODE solver used in training and rollouts
     * @param step the time step {@link #forward} advances
     * @param scale inputs are divided by it before the network (a fixed normalisation)
     * @param seed the seed
     * @return the model
     */
    public static UniversalDifferentialEquation of(int dimension, KnownPhysics known, float[] parameters, int width,
            OdeSolver solver, float step, float scale, long seed) {
        return new UniversalDifferentialEquation(dimension, known, parameters, width, solver, step, scale, seed);
    }

    /**
     * The right-hand side.
     *
     * @param x states {@code [b, d]}
     * @return {@code x'}
     */
    public Var rhs(Var x) {
        Var out = null;
        if (known != null) {
            out = known.apply(x, physical);
        }
        if (network != null) {
            Var n = network.forward(x.div(scale));
            out = out == null ? n : out.add(n);
        }
        return out;
    }

    @Override
    public Var forward(Var x) {
        return solver.integrate((t, z) -> rhs(z), x, 0, step);
    }

    /**
     * The one-step shooting loss on consecutive samples of a trajectory.
     *
     * @param states {@code [n][d]}, spaced by the model's step
     * @return the mean squared one-step error
     */
    public Var loss(double[][] states) {
        double[][] a = java.util.Arrays.copyOfRange(states, 0, states.length - 1);
        double[][] b = java.util.Arrays.copyOfRange(states, 1, states.length);
        return forward(Var.of(HamiltonianNetwork.tensor(a))).sub(HamiltonianNetwork.tensor(b)).square().mean();
    }

    /**
     * A rollout.
     *
     * @param x0 the initial state
     * @param steps the steps
     * @return states {@code [steps + 1][d]}
     */
    public double[][] rollout(double[] x0, int steps) {
        return Var.noGrad(() -> {
            double[][] out = new double[steps + 1][];
            out[0] = x0.clone();
            Var x = Var.of(HamiltonianNetwork.tensor(new double[][] {x0}));
            for (int t = 1; t <= steps; t++) {
                x = forward(x);
                float[] v = x.value().toArray();
                out[t] = new double[dimension];
                for (int i = 0; i < dimension; i++) {
                    out[t][i] = v[i];
                }
            }
            return out;
        });
    }

    /**
     * The network's output at states, for symbolic recovery with {@link Sindy}.
     *
     * @param states {@code [n][d]}
     * @return {@code NN(x)} {@code [n][d]}
     */
    public double[][] networkTerm(double[][] states) {
        float[] v = Var.noGrad(() -> network.forward(Var.of(HamiltonianNetwork.tensor(states)).div(scale)))
                .value().toArray();
        double[][] out = new double[states.length][dimension];
        for (int i = 0; i < states.length; i++) {
            for (int j = 0; j < dimension; j++) {
                out[i][j] = v[i * dimension + j];
            }
        }
        return out;
    }

    /** @return the current physical parameters */
    public double[] physicalParameters() {
        double[] p = new double[physical.size()];
        for (int i = 0; i < p.length; i++) {
            p[i] = physical.get(i).value().item();
        }
        return p;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(physical);
        if (network != null) {
            p.addAll(network.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "UniversalDifferentialEquation(d=" + dimension + ", physics=" + (known != null) + ", network="
                + (network != null) + ")";
    }
}
