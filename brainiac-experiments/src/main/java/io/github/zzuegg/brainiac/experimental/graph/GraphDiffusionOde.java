package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * GRAND, graph neural diffusion (Chamberlain et al., 2021, <a href="https://arxiv.org/abs/2106.10934">arXiv:2106.10934</a>):
 * message passing as a diffusion PDE on the graph,
 * <pre>
 * dX/dt = (A(X) - I) X,   A(X)_ij = softmax_{j in N(i) + i} ((X W_q)_i . (X W_k)_j / sqrt(a))
 * </pre>
 * integrated from {@code t = 0} to {@code T} by an {@link OdeSolver} (Euler, RK4 or
 * adaptive Dormand-Prince). Depth is integration time: the solver, not a layer count,
 * decides how many propagation steps run, and since the right-hand side is a row-stochastic
 * attention minus the identity, every step is a convex averaging of neighbours whose
 * speed the model learns. With {@code recompute = false} the attention is computed once
 * from {@code X(0)} (GRAND-l, linear diffusion); with {@code true} it is recomputed at
 * every evaluation (GRAND-nl).
 *
 * <p>Inputs: node features {@code [n, width]} and the adjacency {@code [n, n]}; output
 * {@code X(T)}, the same shape.
 */
public final class GraphDiffusionOde implements MultiInputLayer {

    private final int width;
    private final int attentionWidth;
    private final OdeSolver solver;
    private final float time;
    private final boolean recompute;
    private final Var wq;
    private final Var wk;

    /**
     * Creates the block.
     *
     * @param width node features
     * @param attentionWidth the query/key width {@code a}
     * @param solver the integrator
     * @param time the integration time {@code T}
     * @param recompute whether the attention follows {@code X(t)}
     * @param random the initialization source
     */
    public GraphDiffusionOde(int width, int attentionWidth, OdeSolver solver, float time, boolean recompute,
            SplittableRandom random) {
        this.width = width;
        this.attentionWidth = attentionWidth;
        this.solver = Objects.requireNonNull(solver);
        this.time = time;
        this.recompute = recompute;
        this.wq = GraphMath.weight(width, attentionWidth, random);
        this.wk = GraphMath.weight(width, attentionWidth, random);
    }

    /** @return the solver, which reports its function evaluations */
    public OdeSolver solver() {
        return solver;
    }

    private Var attention(Var x, Tensor bias) {
        Var scores = x.matmul(wq).matmul(x.matmul(wk).transpose()).mul((float) (1 / Math.sqrt(attentionWidth)));
        return GraphOps.softmax(scores.add(bias));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x0 = inputs.get(0);
        Tensor bias = GraphOps.attentionBias(inputs.get(1).value());
        Var fixed = recompute ? null : attention(x0, bias);
        return solver.integrate((t, z) -> (fixed == null ? attention(z, bias) : fixed).matmul(z).sub(z), x0, 0,
                time);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk);
    }

    @Override
    public int outputSize(int inputSize) {
        return width;
    }

    @Override
    public String toString() {
        return "GraphDiffusionOde(T=" + time + ", " + solver.method() + (recompute ? ", nonlinear" : "") + ")";
    }
}
