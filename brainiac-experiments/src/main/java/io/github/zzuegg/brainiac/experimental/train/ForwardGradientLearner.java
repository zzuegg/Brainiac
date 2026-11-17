package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

/**
 * Trains a {@link SimpleMlp} with {@link ForwardGradient}s only: every batch the weights
 * move along random directions scaled by the loss' directional derivative, never along the
 * exact gradient.
 */
public final class ForwardGradientLearner implements LocalLearner {

    private final SimpleMlp net;
    private final Optimizer optimizer;
    private final ForwardGradient estimator;

    /**
     * @param net the network
     * @param optimizer the optimizer that applies the estimates
     * @param directions random directions averaged per batch
     * @param seed the seed
     */
    public ForwardGradientLearner(SimpleMlp net, Optimizer optimizer, int directions, long seed) {
        this.net = net;
        this.optimizer = optimizer;
        this.estimator = new ForwardGradient(seed, directions);
    }

    private Var loss(Tensor x, Tensor labels) {
        return Functional.crossEntropy(net.forward(Var.of(x)), labels);
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        List<Var> params = net.parameters();
        float value = estimator.estimate(params, p -> loss(inputs, labels));
        optimizer.step(params);
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return net.predict(inputs);
    }
}
