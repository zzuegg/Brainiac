package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An input-convex neural network (Amos et al. 2017, <a href="https://arxiv.org/abs/1609.07152">
 * arXiv:1609.07152</a>): a scalar network {@code phi(x)} that is convex in its input.
 *
 * <p>Each hidden layer is {@code z_(k+1) = softplus(W_k z_k + A_k x + b_k)} with
 * {@code W_k >= 0}; a non-negative combination of convex functions passed through a convex,
 * non-decreasing activation stays convex, and the direct "passthrough" terms {@code A_k x} keep
 * it expressive. The output adds {@code alpha/2 |x|^2} for strong convexity. The non-negativity
 * is kept by {@link #project} after every optimiser step (clipping, as in Makkuva et al.).
 * {@link #gradient} returns {@code grad phi(x)}, by Brenier's theorem the form of every
 * quadratic-cost optimal transport map ({@link NeuralMongeMap}).
 */
public final class InputConvexNetwork implements Layer {

    private final List<Var> passthrough = new ArrayList<>();
    private final List<Var> hidden = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final Var outW;
    private final Var outA;
    private final float alpha;

    /**
     * Creates an ICNN.
     *
     * @param inputs the input dimension
     * @param width the hidden width
     * @param layers the number of hidden layers, at least 1
     * @param alpha the strong-convexity weight of {@code |x|^2 / 2}, for example 0.1
     * @param random the initialisation source
     */
    public InputConvexNetwork(int inputs, int width, int layers, float alpha, SplittableRandom random) {
        this.alpha = alpha;
        for (int l = 0; l < layers; l++) {
            passthrough.add(Var.param(Tensor.normal(Shape.of(inputs, width), 0, (float) Math.sqrt(1.0 / inputs),
                    random)));
            biases.add(Var.param(Tensor.zeros(Shape.of(1, width))));
            if (l > 0) {
                hidden.add(Var.param(Tensor.uniform(Shape.of(width, width), 0, (float) (1.0 / width), random)));
            }
        }
        outW = Var.param(Tensor.uniform(Shape.of(width, 1), 0, (float) (1.0 / width), random));
        outA = Var.param(Tensor.zeros(Shape.of(inputs, 1)));
    }

    @Override
    public Var forward(Var x) {
        Var z = x.matmul(passthrough.getFirst()).add(biases.getFirst()).softplus();
        for (int l = 1; l < passthrough.size(); l++) {
            z = z.matmul(hidden.get(l - 1)).add(x.matmul(passthrough.get(l))).add(biases.get(l)).softplus();
        }
        return z.matmul(outW).add(x.matmul(outA)).add(x.square().sum(1, true).mul(0.5f * alpha));
    }

    /**
     * The input gradient {@code grad phi(x)}, recorded so it can be differentiated again.
     *
     * @param x the inputs {@code [B, d]} (a fresh leaf is made from their value)
     * @param createGraph keep the graph for training through the map
     * @return {@code [B, d]}
     */
    public Var gradient(Var x, boolean createGraph) {
        Var leaf = Var.param(x.value());
        Var phi = forward(leaf).sum();
        return Var.gradients(phi, List.of(leaf), createGraph).getFirst();
    }

    /** Clips the weights that must be non-negative back to {@code >= 0}. */
    public void project() {
        for (Var w : hidden) {
            w.assign(w.value().relu());
        }
        outW.assign(outW.value().relu());
    }

    @Override
    public List<Var> parameters() {
        List<Var> ps = new ArrayList<>(passthrough);
        ps.addAll(hidden);
        ps.addAll(biases);
        ps.add(outW);
        ps.add(outA);
        return ps;
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }
}
