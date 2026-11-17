package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Mono-Forward (Gong et al. 2025, <a href="https://arxiv.org/abs/2501.09238">arXiv:2501.09238</a>):
 * every hidden layer owns a small projection matrix {@code M_l} that maps its activity to
 * class scores, and the layer and its projection are trained only by the cross-entropy of
 * those local scores. The layer's input is detached, so no gradient crosses a layer
 * boundary; memory stays at one layer's activations, and all layers could train in
 * parallel. Prediction uses the last layer's projection (or the sum of all, see
 * {@link #predictWithAllLayers}).
 */
public final class MonoForward implements LocalLearner {

    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final List<Var> projections = new ArrayList<>();
    private final Optimizer optimizer;
    private boolean allLayers;

    /**
     * @param inputs features per sample
     * @param classes the number of classes
     * @param optimizer the optimizer for all layers and projections
     * @param seed the random seed
     * @param hidden the hidden layer sizes, at least one
     */
    public MonoForward(int inputs, int classes, Optimizer optimizer, long seed, int... hidden) {
        if (hidden.length == 0) {
            throw new IllegalArgumentException("need at least one hidden layer");
        }
        this.optimizer = optimizer;
        SplittableRandom r = new SplittableRandom(seed);
        int in = inputs;
        for (int size : hidden) {
            weights.add(Var.param(Tensor.normal(Shape.of(in, size), 0, (float) Math.sqrt(2.0 / in), r)));
            biases.add(Var.param(Tensor.zeros(Shape.of(1, size))));
            projections.add(Var.param(Tensor.normal(Shape.of(size, classes), 0, (float) Math.sqrt(1.0 / size), r)));
            in = size;
        }
    }

    /**
     * @param on whether {@link #predict} sums the scores of every layer's projection
     * @return this learner
     */
    public MonoForward predictWithAllLayers(boolean on) {
        this.allLayers = on;
        return this;
    }

    /** @return every weight, bias and projection */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        p.addAll(biases);
        p.addAll(projections);
        return p;
    }

    /**
     * Local class scores of every layer.
     *
     * @param inputs the inputs
     * @return one {@code [batch, classes]} tensor per hidden layer
     */
    public List<Tensor> layerScores(Tensor inputs) {
        List<Tensor> out = new ArrayList<>();
        Tensor h = inputs;
        for (int l = 0; l < weights.size(); l++) {
            Var a = Var.of(h).matmul(weights.get(l)).add(biases.get(l)).relu();
            out.add(a.matmul(projections.get(l)).value());
            h = a.value();
        }
        return out;
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        Tensor h = inputs;
        Var total = null;
        for (int l = 0; l < weights.size(); l++) {
            Var a = Var.of(h).matmul(weights.get(l)).add(biases.get(l)).relu();   // input detached
            Var local = Functional.crossEntropy(a.matmul(projections.get(l)), labels);
            total = total == null ? local : total.add(local);
            h = a.value();
        }
        total.backward();
        float value = total.value().item() / weights.size();
        optimizer.step(parameters());
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        List<Tensor> scores = layerScores(inputs);
        if (!allLayers) {
            return scores.getLast();
        }
        Tensor sum = scores.get(0);
        for (int i = 1; i < scores.size(); i++) {
            sum = sum.add(scores.get(i));
        }
        return sum;
    }
}
