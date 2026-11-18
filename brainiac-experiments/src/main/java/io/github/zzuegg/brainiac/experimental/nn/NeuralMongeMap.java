package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * A neural optimal transport (Monge) map learned from samples with two input-convex networks
 * (Makkuva et al. 2020, <a href="https://arxiv.org/abs/1908.10962">arXiv:1908.10962</a>).
 *
 * <p>By Brenier's theorem the quadratic-cost OT map from {@code P} to {@code Q} is the gradient
 * of a convex function. The semi-dual of OT becomes the min-max problem
 * {@code min_f max_g E_P[<x, grad g(x)> - f(grad g(x))] + E_Q[f(y)]} over convex {@code f, g}
 * ({@link InputConvexNetwork}s); at the optimum {@code f} is the Kantorovich potential and
 * {@code T = grad g} the transport map. Training alternates a few ascent steps on {@code g} with
 * one descent step on {@code f}, differentiating through {@code grad g} (second-order autograd).
 * Unlike a discrete coupling or a barycentric projection, the result is a continuous map that
 * transports new points.
 */
public final class NeuralMongeMap {

    private final InputConvexNetwork f;
    private final InputConvexNetwork g;
    private final Optimizer optF;
    private final Optimizer optG;

    /**
     * Creates the two potentials.
     *
     * @param dims the dimension
     * @param width the hidden width
     * @param layers the hidden layers
     * @param learningRate the Adam learning rate of both
     * @param seed the initialisation seed
     */
    public NeuralMongeMap(int dims, int width, int layers, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        f = new InputConvexNetwork(dims, width, layers, 0.1f, r);
        g = new InputConvexNetwork(dims, width, layers, 0.1f, r);
        optF = Optimizer.adam(learningRate);
        optG = Optimizer.adam(learningRate);
    }

    /**
     * One outer iteration: {@code innerSteps} updates of {@code g}, then one of {@code f}.
     *
     * @param source a source batch {@code [B, d]}
     * @param target a target batch {@code [B, d]}
     * @param innerSteps updates of {@code g} per update of {@code f}, for example 5
     * @return the current value of the min-max objective (up to the constant)
     */
    public double step(Tensor source, Tensor target, int innerSteps) {
        for (int k = 0; k < innerSteps; k++) {
            Var x = Var.of(source);
            Var tx = g.gradient(x, true);
            Var v = tx.mul(x).sum(1, true).sub(f.forward(tx)).mean();
            v.neg().backward();
            optG.step(g.parameters());
            g.project();
            for (Var p : f.parameters()) {
                p.zeroGrad();
            }
        }
        Var tx = Var.of(g.gradient(Var.of(source), false).value());
        Var value = f.forward(Var.of(target)).mean().sub(f.forward(tx).mean());
        value.backward();
        optF.step(f.parameters());
        f.project();
        for (Var p : g.parameters()) {
            p.zeroGrad();
        }
        return value.value().item();
    }

    /**
     * Transports points with {@code grad g}.
     *
     * @param x points {@code [B, d]}
     * @return {@code T(x)}
     */
    public Tensor transport(Tensor x) {
        return g.gradient(Var.of(x), false).value();
    }
}
