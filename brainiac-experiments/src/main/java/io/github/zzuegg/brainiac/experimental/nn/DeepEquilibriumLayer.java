package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.AndersonSolver;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A deep equilibrium layer (Bai, Kolter and Koltun 2019,
 * <a href="https://arxiv.org/abs/1909.01377">arXiv:1909.01377</a>): an "infinitely deep"
 * weight-tied layer whose output is the fixed point {@code z* = tanh(z* W + x U + b)}, found by
 * {@link AndersonSolver} without recording a graph. The backward pass uses the implicit function
 * theorem: it solves {@code u = dL/dz* + (df/dz)^T u} (itself a fixed point, solved with the same
 * accelerated solver on vector-Jacobian products) and then back-propagates {@code u} through one
 * application of {@code f}. Training memory is that of a single layer, whatever the effective
 * depth.
 *
 * <p>With a positive {@code jacobianWeight}, {@link #auxiliaryLoss()} returns the Hutchinson
 * estimate of {@code |df/dz|_F^2} at the fixed point (Bai et al. 2021,
 * <a href="https://arxiv.org/abs/2106.14342">arXiv:2106.14342</a>), which keeps the fixed point
 * stable and cheap to find; add it to the loss.
 */
public final class DeepEquilibriumLayer implements Layer {

    private final Var u;
    private final Var bias;
    private final Var w;
    private final AndersonSolver solver;
    private final float jacobianWeight;
    private final SplittableRandom random;
    private Var auxiliary;
    private int forwardIterations;
    private int backwardIterations;

    /**
     * @param inputs the input width
     * @param hidden the state width
     * @param solver the forward and backward solver
     * @param jacobianWeight the weight of the Jacobian regularizer, 0 for none
     * @param random the source of the initial weights and the regularizer's probes
     */
    public DeepEquilibriumLayer(int inputs, int hidden, AndersonSolver solver, float jacobianWeight,
            SplittableRandom random) {
        this.u = Var.param(Tensor.normal(Shape.of(inputs, hidden), 0, (float) Math.sqrt(1.0 / inputs), random));
        this.bias = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        this.w = Var.param(Tensor.normal(Shape.of(hidden, hidden), 0, (float) (0.5 / Math.sqrt(hidden)), random));
        this.solver = solver;
        this.jacobianWeight = jacobianWeight;
        this.random = random;
    }

    /**
     * A spec for model builders.
     *
     * @param hidden the state width
     * @param solver the solver
     * @param jacobianWeight the Jacobian regularizer's weight
     * @return the spec
     */
    public static LayerSpec spec(int hidden, AndersonSolver solver, float jacobianWeight) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = hidden;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DeepEquilibriumLayer(input.dim(-1), hidden, solver, jacobianWeight, random);
            }
        };
    }

    private static Var cell(Var z, Var inject, Var w) {
        return z.matmul(w).add(inject).tanh();
    }

    @Override
    public Var forward(Var input) {
        Var inject = input.matmul(u).add(bias);
        Tensor injT = inject.value();
        Tensor wT = w.value();
        AndersonSolver.Result fwd = Var.noGrad(() -> solver.solve(
                z -> z.matmul(wT).add(injT).tanh(), Tensor.zeros(injT.shape())));
        forwardIterations = fwd.iterations();
        Tensor zStar = fwd.z();
        Var out = Var.custom(in -> zStar, (in, o, grad) -> implicitGradient(in[0], in[1], zStar, grad), inject, w);
        if (jacobianWeight > 0 && Var.gradEnabled()) {
            Var zv = Var.param(zStar);
            Var f = cell(zv, inject, w);
            Tensor probe = Tensor.normal(f.shape(), 0, 1, random);
            Var vjp = Var.gradients(f, Var.of(probe), List.of(zv), true).get(0);
            auxiliary = vjp.square().sum().div(injT.shape().dim(0)).mul(jacobianWeight);
        } else {
            auxiliary = null;
        }
        return out;
    }

    private Tensor[] implicitGradient(Tensor inject, Tensor wT, Tensor zStar, Tensor grad) {
        Var zv = Var.param(zStar);
        Var iv = Var.param(inject);
        Var wv = Var.param(wT);
        Var f = cell(zv, iv, wv);
        AndersonSolver.Result back = solver.solve(
                g -> Var.gradients(f, Var.of(g), List.of(zv), false).get(0).value().add(grad), grad);
        backwardIterations = back.iterations();
        List<Var> grads = Var.gradients(f, Var.of(back.z()), List.of(iv, wv), false);
        return new Tensor[] {grads.get(0).value(), grads.get(1).value()};
    }

    /** @return solver iterations of the last forward pass */
    public int forwardIterations() {
        return forwardIterations;
    }

    /** @return solver iterations of the last backward pass */
    public int backwardIterations() {
        return backwardIterations;
    }

    @Override
    public Var auxiliaryLoss() {
        return auxiliary;
    }

    @Override
    public List<Var> parameters() {
        return List.of(u, bias, w);
    }

    @Override
    public int outputSize(int inputSize) {
        return w.shape().dim(0);
    }

    @Override
    public String toString() {
        return "DeepEquilibriumLayer(" + u.shape().dim(0) + " -> " + w.shape().dim(0) + ", " + solver + ")";
    }
}
