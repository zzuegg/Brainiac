package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

/**
 * Ordinary backpropagation with softmax cross-entropy on a {@link SimpleMlp}: the baseline
 * every other {@link LocalLearner} is measured against.
 */
public final class BackpropLearner implements LocalLearner {

    private final SimpleMlp net;
    private final Optimizer optimizer;

    /**
     * @param net the network
     * @param optimizer the optimizer for all its parameters
     */
    public BackpropLearner(SimpleMlp net, Optimizer optimizer) {
        this.net = net;
        this.optimizer = optimizer;
    }

    /** @return the network */
    public SimpleMlp network() {
        return net;
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        Var loss = Functional.crossEntropy(net.forward(Var.of(inputs)), labels);
        loss.backward();
        float value = loss.value().item();
        optimizer.step(net.parameters());
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return net.predict(inputs);
    }
}
