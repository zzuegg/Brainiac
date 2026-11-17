package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Helpers for μP, the maximal update parametrization (Yang et al. 2022, Tensor Programs V,
 * <a href="https://arxiv.org/abs/2203.03466">arXiv:2203.03466</a>): initialisation and
 * per-layer Adam learning rates scaled with the width so that every layer's activations
 * change by the same order of magnitude whatever the width. Hyperparameters tuned on a
 * narrow model (the <em>base width</em>) then transfer unchanged to a wide one: tune on the
 * CPU, train big on the GPU.
 *
 * <p>For Adam, with width multiplier {@code r = width / baseWidth} (Table 3 of the paper):
 * <ul>
 *   <li>input weights: variance {@code 1/fanIn}, learning rate {@code lr};</li>
 *   <li>hidden weights: variance {@code 1/fanIn}, learning rate {@code lr / r};</li>
 *   <li>output weights: variance {@code 1/fanIn^2} (scaled down by the width), learning
 *       rate {@code lr / r};</li>
 *   <li>biases: learning rate {@code lr}.</li>
 * </ul>
 */
public final class MaximalUpdate {

    /** Where a weight sits in the network. */
    public enum Role {
        /** maps the input (fixed size) to the hidden width */
        INPUT,
        /** maps hidden width to hidden width */
        HIDDEN,
        /** maps hidden width to the output (fixed size) */
        OUTPUT
    }

    private final int baseWidth;
    private final int width;

    /**
     * @param baseWidth the width hyperparameters were tuned at
     * @param width the width of this model
     */
    public MaximalUpdate(int baseWidth, int width) {
        if (baseWidth < 1 || width < 1) {
            throw new IllegalArgumentException("widths must be positive");
        }
        this.baseWidth = baseWidth;
        this.width = width;
    }

    /** @return {@code width / baseWidth} */
    public float widthMultiplier() {
        return (float) width / baseWidth;
    }

    /**
     * @param role the weight's role
     * @return a core {@link Init} with the μP variance for that role
     */
    public Init init(Role role) {
        return (shape, fanIn, fanOut, random) -> {
            double std = role == Role.OUTPUT ? 1.0 / fanIn : Math.sqrt(1.0 / fanIn);
            return Tensor.normal(shape, 0, (float) std, random);
        };
    }

    /**
     * @param role the weight's role
     * @param baseLearningRate the learning rate tuned at the base width
     * @return the Adam learning rate for that weight at this width
     */
    public float adamLearningRate(Role role, float baseLearningRate) {
        return role == Role.INPUT ? baseLearningRate : baseLearningRate / widthMultiplier();
    }

    /**
     * An Adam optimizer with μP learning rates.
     *
     * @param baseLearningRate the learning rate tuned at the base width
     * @param hiddenAndOutputWeights the weights whose rate scales with {@code 1/r}
     * @return {@code Optimizer.groups} with Adam at {@code lr} for the rest
     */
    public Optimizer adam(float baseLearningRate, List<Var> hiddenAndOutputWeights) {
        return Optimizer.groups(Optimizer.adam(baseLearningRate), Optimizer.Group.of(hiddenAndOutputWeights,
                Optimizer.adam(adamLearningRate(Role.HIDDEN, baseLearningRate))));
    }

    /**
     * Re-initialises a model's {@link Dense} layers in μP (the first is the input layer, the
     * last the output layer, the rest hidden) and returns the matching Adam optimizer.
     *
     * @param model a model whose weights are in {@code Dense} layers
     * @param baseLearningRate the learning rate tuned at the base width
     * @param seed the seed of the new weights
     * @return the optimizer for {@code model.parameters()}
     */
    public Optimizer apply(Model model, float baseLearningRate, long seed) {
        List<Dense> dense = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof Dense d) {
                dense.add(d);
            }
        }
        if (dense.size() < 2) {
            throw new IllegalArgumentException("need at least an input and an output Dense layer");
        }
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> scaled = new ArrayList<>();
        for (int i = 0; i < dense.size(); i++) {
            Role role = i == 0 ? Role.INPUT : i == dense.size() - 1 ? Role.OUTPUT : Role.HIDDEN;
            Var w = dense.get(i).weight();
            int fanIn = w.shape().dim(0);
            w.assign(init(role).create(w.shape(), fanIn, w.shape().dim(1), r));
            if (role != Role.INPUT) {
                scaled.add(w);
            }
        }
        return adam(baseLearningRate, scaled);
    }
}
