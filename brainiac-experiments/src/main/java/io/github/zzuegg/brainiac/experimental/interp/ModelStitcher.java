package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Model stitching (Lenc and Vedaldi, 2015; Bansal et al., NeurIPS 2021): the functional test
 * of whether two networks represent the same thing at a layer. The bottom of network A (up
 * to a step of its {@link Model#graph()}) feeds a trainable affine "stitching" layer whose
 * output replaces the same step of network B, whose top then finishes the computation. Both
 * networks stay frozen; only the stitch is trained, on B's task loss. A small accuracy
 * penalty against B alone means A's layer carries what B's top needs, up to a linear map.
 *
 * <p>Both models are driven through {@link Model#forward(Var, boolean, Model.Hook)}, so any
 * core model can be stitched without changes.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2106.07682">arXiv:2106.07682</a>.
 */
public final class ModelStitcher {

    private final Model bottom;
    private final Model top;
    private final int step;
    private final Var weight;
    private final Var bias;

    /**
     * Creates a stitch, initialised to a small random map.
     *
     * @param bottom network A, which supplies the activation of {@code step}
     * @param top network B, whose {@code step} output is replaced
     * @param step the graph step (layer index of a sequential model)
     * @param widthBottom A's width at that step
     * @param widthTop B's width at that step
     * @param seed the seed
     */
    public ModelStitcher(Model bottom, Model top, int step, int widthBottom, int widthTop, long seed) {
        this.bottom = bottom;
        this.top = top;
        this.step = step;
        SplittableRandom r = new SplittableRandom(seed);
        this.weight = Var.param(Tensor.normal(Shape.of(widthBottom, widthTop), 0,
                (float) Math.sqrt(1.0 / widthBottom), r));
        this.bias = Var.param(Tensor.zeros(Shape.of(widthTop)));
    }

    /**
     * The activation of a step of a model.
     *
     * @param model the model
     * @param step the step
     * @param x inputs
     * @return its value {@code [batch, width]}
     */
    public static Tensor activation(Model model, int step, Tensor x) {
        Tensor[] seen = new Tensor[1];
        Var.noGrad(() -> model.forward(Var.of(x), false, (s, out) -> {
            if (s.index() == step) {
                seen[0] = out.value();
            }
            return out;
        }));
        return seen[0];
    }

    /**
     * The stitched network's logits.
     *
     * @param x inputs
     * @return B's output with its step replaced by the stitched activation of A
     */
    public Var forward(Tensor x) {
        Var stitched = Var.of(activation(bottom, step, x)).matmul(weight).add(bias);
        return top.forward(Var.of(x), false, (s, out) -> s.index() == step ? stitched : out);
    }

    /**
     * Trains the stitch on B's task (cross-entropy), leaving both networks frozen.
     *
     * @param x inputs {@code [n, ...]}
     * @param y class ids {@code [n]}
     * @param steps Adam steps on minibatches
     * @param batch the minibatch size
     * @param seed the minibatch seed
     */
    public void train(Tensor x, Tensor y, int steps, int batch, long seed) {
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.shape().dim(0);
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(Math.max(1, n - batch + 1));
            int to = Math.min(n, from + batch);
            Var loss = Functional.crossEntropy(forward(x.slice(0, from, to)), y.slice(0, from, to));
            loss.backward();
            opt.step(parameters());
            for (Var p : top.parameters()) {
                p.zeroGrad();
            }
        }
    }

    /**
     * Accuracy of the stitched network.
     *
     * @param x inputs
     * @param y class ids
     * @return the share classified correctly
     */
    public double accuracy(Tensor x, Tensor y) {
        float[] pred = Var.noGrad(() -> forward(x)).value().argmax(1).toArray();
        float[] t = y.toArray();
        int hit = 0;
        for (int i = 0; i < t.length; i++) {
            hit += pred[i] == t[i] ? 1 : 0;
        }
        return (double) hit / t.length;
    }

    /** @return the stitch's weight and bias */
    public List<Var> parameters() {
        return List.of(weight, bias);
    }
}
