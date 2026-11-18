package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A Hamiltonian neural network (Greydanus, Dzamba and Yosinski, 2019): a network learns a
 * scalar energy {@code H_theta(q, p)} and the dynamics follow from Hamilton's equations,
 * {@code q' = dH/dp}, {@code p' = -dH/dq}, so the learned flow conserves the learned energy
 * exactly. Rolled out with a {@link SymplecticIntegrator}, long trajectories do not drift.
 *
 * <p>As a {@link Layer}, {@code forward} maps states {@code [b, 2d]} (positions then momenta)
 * to time derivatives {@code [b, 2d]}, differentiable for training on observed derivatives;
 * the symplectic gradient comes from {@code Var.gradients(..., createGraph = true)}.
 * {@link #gradient()} freezes the network into a fast double-precision gradient for
 * integrators.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1906.01563">arXiv:1906.01563</a>.
 */
public final class HamiltonianNetwork implements Layer {

    private final int dimension;
    private final SimpleMlp energy;

    private HamiltonianNetwork(int dimension, int width, int depth, long seed) {
        this.dimension = dimension;
        int[] sizes = new int[depth + 2];
        sizes[0] = 2 * dimension;
        for (int i = 1; i <= depth; i++) {
            sizes[i] = width;
        }
        sizes[depth + 1] = 1;
        this.energy = SimpleMlp.tanh(seed, sizes);
    }

    /**
     * Creates the network.
     *
     * @param dimension degrees of freedom {@code d}
     * @param width hidden width
     * @param depth hidden layers
     * @param seed the seed
     * @return the network
     */
    public static HamiltonianNetwork of(int dimension, int width, int depth, long seed) {
        return new HamiltonianNetwork(dimension, width, depth, seed);
    }

    /**
     * The learned energy.
     *
     * @param x states {@code [b, 2d]}
     * @return {@code [b, 1]}
     */
    public Var energy(Var x) {
        return energy.forward(x);
    }

    @Override
    public Var forward(Var input) {
        Var x = input.requiresGrad() ? input : Var.param(input.value());
        Var g = Var.gradients(energy.forward(x).sum(), List.of(x), true).get(0);
        Var dq = g.slice(1, 0, dimension);
        Var dp = g.slice(1, dimension, 2 * dimension);
        return Var.concat(1, dp, dq.neg());
    }

    /**
     * The mean squared error of the predicted time derivatives.
     *
     * @param states {@code [b, 2d]}
     * @param derivatives observed time derivatives {@code [b, 2d]}
     * @return the loss
     */
    public Var loss(Tensor states, Tensor derivatives) {
        return forward(Var.param(states)).sub(derivatives).square().mean();
    }

    /** @return a frozen double-precision gradient of the current energy, for integrators */
    public SymplecticIntegrator.Gradient gradient() {
        MlpSnapshot s = new MlpSnapshot(energy);
        return (q, p) -> {
            double[] x = new double[2 * dimension];
            System.arraycopy(q, 0, x, 0, dimension);
            System.arraycopy(p, 0, x, dimension, dimension);
            double[] g = s.inputGradient(x, 0);
            return new double[][] {java.util.Arrays.copyOfRange(g, 0, dimension),
                java.util.Arrays.copyOfRange(g, dimension, 2 * dimension)};
        };
    }

    /** @return the frozen learned energy as a function of the state */
    public java.util.function.ToDoubleFunction<double[]> energyFunction() {
        MlpSnapshot s = new MlpSnapshot(energy);
        return x -> s.forward(x)[0];
    }

    /**
     * Batches of states as a tensor.
     *
     * @param x states {@code [n][k]}
     * @return {@code [n, k]}
     */
    public static Tensor tensor(double[][] x) {
        int n = x.length;
        int k = x[0].length;
        float[] f = new float[n * k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                f[i * k + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(n, k), f);
    }

    @Override
    public List<Var> parameters() {
        return energy.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return 2 * dimension;
    }

    @Override
    public String toString() {
        return "HamiltonianNetwork(d=" + dimension + ")";
    }
}
