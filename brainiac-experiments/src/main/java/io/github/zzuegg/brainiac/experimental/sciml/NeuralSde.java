package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A neural SDE {@code dX = f_theta(X) dt + g_theta(X) dW} with learned drift and diagonal
 * diffusion networks ({@code g = softplus(.) + floor > 0}). Fitted to observed transitions by
 * the Euler-Maruyama likelihood: over a step {@code dt},
 * {@code X' | X ~ N(X + f dt, diag(g^2 dt))}, so {@link #nll} is a closed-form Gaussian
 * negative log-likelihood and training needs no adjoint. Unlike a neural ODE it models the
 * noise, so it reproduces noise-driven behaviour such as switching between the wells of a
 * double-well potential. {@link #simulate} samples paths with {@link SdeSolver}.
 *
 * <p>As a {@link Layer}, {@code forward} returns the drift.
 *
 * <p>Papers: Li et al., <a href="https://arxiv.org/abs/2001.01328">arXiv:2001.01328</a>;
 * Kidger et al., <a href="https://arxiv.org/abs/2102.03657">arXiv:2102.03657</a>.
 */
public final class NeuralSde implements Layer {

    private static final float FLOOR = 1e-3f;
    private final int dimension;
    private final SimpleMlp drift;
    private final SimpleMlp diffusion;

    private NeuralSde(int dimension, int width, long seed) {
        this.dimension = dimension;
        this.drift = SimpleMlp.tanh(seed, dimension, width, width, dimension);
        this.diffusion = SimpleMlp.tanh(seed + 1, dimension, width, dimension);
    }

    /**
     * Creates the model.
     *
     * @param dimension the state size
     * @param width hidden width of both networks
     * @param seed the seed
     * @return the model
     */
    public static NeuralSde of(int dimension, int width, long seed) {
        return new NeuralSde(dimension, width, seed);
    }

    @Override
    public Var forward(Var x) {
        return drift.forward(x);
    }

    /**
     * @param x states {@code [b, d]}
     * @return the diffusion {@code [b, d]}, positive
     */
    public Var diffusion(Var x) {
        return diffusion.forward(x).softplus().add(FLOOR);
    }

    /**
     * The Euler-Maruyama transition negative log-likelihood per scalar observation.
     *
     * @param x states {@code [b, d]}
     * @param next the states one step later
     * @param dt the step
     * @return the mean NLL
     */
    public Var nll(Tensor x, Tensor next, float dt) {
        Var xv = Var.of(x);
        Var mean = xv.add(drift.forward(xv).mul(dt));
        Var var = diffusion(xv).square().mul(dt);
        Var r = Var.of(next).sub(mean);
        return r.square().div(var).add(var.log()).add((float) Math.log(2 * Math.PI)).mul(0.5f).mean();
    }

    /**
     * Samples a path.
     *
     * @param x0 the initial state
     * @param dt the step
     * @param steps the steps
     * @param random the noise source
     * @return states {@code [steps + 1][d]}
     */
    public double[][] simulate(double[] x0, double dt, int steps, SplittableRandom random) {
        MlpSnapshot f = new MlpSnapshot(drift);
        MlpSnapshot g = new MlpSnapshot(diffusion);
        return SdeSolver.simulate(f::forward, x -> {
            double[] v = g.forward(x);
            for (int i = 0; i < v.length; i++) {
                v[i] = Math.log1p(Math.exp(-Math.abs(v[i]))) + Math.max(v[i], 0) + FLOOR;
            }
            return v;
        }, x0, dt, steps, false, random);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(drift.parameters());
        p.addAll(diffusion.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return dimension;
    }

    @Override
    public String toString() {
        return "NeuralSde(d=" + dimension + ")";
    }
}
