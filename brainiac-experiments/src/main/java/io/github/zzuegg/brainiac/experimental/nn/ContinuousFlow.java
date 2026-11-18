package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A continuous normalizing flow, FFJORD (Grathwohl et al.,
 * <a href="https://arxiv.org/abs/1810.01367">arXiv:1810.01367</a>): data moves along a
 * learned velocity field {@code dz/dt = v(z, t)} from {@code t = 0} (data) to {@code t = 1}
 * (a standard normal), and the log-density changes by the instantaneous change of variables
 * {@code d log p / dt = -tr(dv/dz)}, so {@code log p(x) = log N(z_1) + int_0^1 tr(dv/dz) dt}
 * is exact. The ODE is solved with the {@link OdeSolver} of core-style {@link NeuralOde}, on
 * the state augmented with the running log-determinant.
 *
 * <p>The trace is exact (one vector-Jacobian product per dimension) or the unbiased Hutchinson
 * estimate {@code eps^T (dv/dz) eps} with one product, which is what makes FFJORD scale to
 * high dimensions. The field is any core {@link Denoiser} and follows core
 * {@code FlowMatching}'s time convention (data at 0, noise at 1), so the same class also gives
 * the exact likelihood of a flow-matching or probability-flow model.
 */
public final class ContinuousFlow implements DensityFlow {

    /** How the divergence {@code tr(dv/dz)} is computed. */
    public enum Trace {
        /** One vector-Jacobian product per dimension: exact. */
        EXACT,
        /** Hutchinson's estimator with one Gaussian probe held fixed through each solve. */
        HUTCHINSON
    }

    private final Denoiser field;
    private final List<Var> parameters;
    private final int dim;
    private final OdeSolver solver;
    private final SplittableRandom random;
    private Trace trace = Trace.EXACT;
    private boolean differentiable;
    private int evaluations;
    private Var kinetic;

    private ContinuousFlow(Denoiser field, List<Var> parameters, int dim, OdeSolver solver, long seed) {
        this.field = field;
        this.parameters = List.copyOf(parameters);
        this.dim = dim;
        this.solver = solver;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Wraps a velocity field.
     *
     * @param field {@code v(z, t)}, called with per-row times in [0, 1]
     * @param parameters its trainable parameters
     * @param dim the data dimension
     * @param solver the ODE solver
     * @param seed the seed of the Hutchinson probes
     * @return the flow
     */
    public static ContinuousFlow of(Denoiser field, List<Var> parameters, int dim, OdeSolver solver, long seed) {
        return new ContinuousFlow(field, parameters, dim, solver, seed);
    }

    /**
     * Chooses the trace computation.
     *
     * @param trace exact or Hutchinson
     * @return this flow
     */
    public ContinuousFlow trace(Trace trace) {
        this.trace = trace;
        return this;
    }

    /**
     * Records the divergence computation so the log-density can be backpropagated into the
     * field's parameters (needed for maximum-likelihood training, not for evaluation).
     *
     * @param on whether {@link #forward} builds a differentiable graph
     * @return this flow
     */
    public ContinuousFlow training(boolean on) {
        this.differentiable = on;
        return this;
    }

    /**
     * The kinetic energy {@code int_0^1 |v|^2 dt} of each row's path in the last
     * {@link #forward}. Adding a small multiple to the loss (RNODE, Finlay et al.,
     * <a href="https://arxiv.org/abs/2002.02798">arXiv:2002.02798</a>) keeps the paths straight,
     * so a fixed-step solver stays accurate and cannot be exploited by the training.
     *
     * @return {@code [batch]}, differentiable when {@link #training} is on
     */
    public Var kineticEnergy() {
        return kinetic;
    }

    /** @return the field evaluations of the last solve */
    public int evaluations() {
        return evaluations;
    }

    @Override
    public Flows.Result forward(Var x) {
        int b = x.shape().dim(0);
        Tensor probe = trace == Trace.HUTCHINSON ? Tensor.normal(Shape.of(b, dim), 0, 1, random) : null;
        Var start = Var.concat(1, x, Var.of(Tensor.zeros(Shape.of(b, 2))));
        Var end = solver.integrate((t, state) -> augmented(t, state, probe), start, 0f, 1f);
        evaluations = solver.evaluations();
        kinetic = end.slice(1, dim + 1, dim + 2).reshape(b);
        return new Flows.Result(end.slice(1, 0, dim), end.slice(1, dim, dim + 1).reshape(b));
    }

    /**
     * One evaluation of the augmented dynamics. The field and its divergence are computed on a
     * fresh leaf copy of the state, so each vector-Jacobian product walks only this
     * evaluation's graph (not the whole trajectory, which would make a solve quadratic); a
     * custom op then splices the result back onto the trajectory with the exact
     * vector-Jacobian product as its state gradient.
     */
    private Var augmented(float t, Var state, Tensor probe) {
        Var zIn = state.slice(1, 0, dim);
        Var local = Var.param(zIn.value());
        Var out = localDynamics(t, local, probe);
        if (!differentiable) {
            return Var.of(out.value());
        }
        return Var.apply(new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[1];
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
                Tensor dz = Var.gradients(out, Var.of(grad), List.of(local), false).get(0).value();
                return new Tensor[] {dz, grad};
            }

            @Override
            public String name() {
                return "ContinuousFlow.splice";
            }
        }, zIn, out);
    }

    private Var localDynamics(float t, Var z, Tensor probe) {
        int b = z.shape().dim(0);
        float[] time = new float[b];
        java.util.Arrays.fill(time, t);
        Var v = field.apply(z, Tensor.of(Shape.of(b), time));
        Var div;
        if (probe != null) {
            Var g = Var.gradients(v, Var.of(probe), List.of(z), differentiable).get(0);
            div = g.mul(probe).sum(1, true);
        } else {
            List<Var> cols = new ArrayList<>();
            for (int k = 0; k < dim; k++) {
                float[] e = new float[b * dim];
                for (int r = 0; r < b; r++) {
                    e[r * dim + k] = 1;
                }
                Var g = Var.gradients(v, Var.of(Tensor.of(Shape.of(b, dim), e)), List.of(z), differentiable).get(0);
                cols.add(g.slice(1, k, k + 1));
            }
            div = cols.get(0);
            for (int k = 1; k < dim; k++) {
                div = div.add(cols.get(k));
            }
        }
        return Var.concat(1, v, div, v.square().sum(1, true));
    }

    @Override
    public Tensor inverse(Tensor z) {
        Var out = solver.integrate((t, s) -> {
            int b = s.shape().dim(0);
            float[] time = new float[b];
            java.util.Arrays.fill(time, t);
            return Var.of(field.apply(s, Tensor.of(Shape.of(b), time)).value());
        }, Var.of(z), 1f, 0f);
        evaluations = solver.evaluations();
        return out.value();
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public List<Var> parameters() {
        return parameters;
    }
}
