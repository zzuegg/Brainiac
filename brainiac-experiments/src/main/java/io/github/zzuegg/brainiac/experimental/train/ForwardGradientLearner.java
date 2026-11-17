package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.DualTensor;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

/**
 * Trains a tanh {@link SimpleMlp} with {@link ForwardGradient}s only: every batch runs
 * forward-mode passes with dual numbers and no backward pass at all, so no activations
 * need to be stored. The forward pass here assumes tanh hidden layers (as
 * {@link SimpleMlp#tanh} builds).
 */
public final class ForwardGradientLearner implements LocalLearner {

    private final SimpleMlp net;
    private final Optimizer optimizer;
    private final ForwardGradient estimator;

    /**
     * @param net a network with tanh hidden layers
     * @param optimizer the optimizer that applies the estimates
     * @param directions random directions averaged per batch
     * @param seed the seed
     */
    public ForwardGradientLearner(SimpleMlp net, Optimizer optimizer, int directions, long seed) {
        this.net = net;
        this.optimizer = optimizer;
        this.estimator = new ForwardGradient(seed, directions);
    }

    private DualTensor loss(List<DualTensor> p, Tensor x, Tensor labels) {
        DualTensor h = DualTensor.constant(x);
        int depth = net.depth();
        for (int l = 0; l < depth; l++) {
            h = h.matmul(p.get(2 * l)).add(p.get(2 * l + 1));
            if (l < depth - 1) {
                h = h.tanh();
            }
        }
        return h.crossEntropy(labels);
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        List<Var> params = net.parameters();
        float value = estimator.estimate(params, d -> loss(d, inputs, labels));
        optimizer.step(params);
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return net.predict(inputs);
    }
}
