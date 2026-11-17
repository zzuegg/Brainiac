package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Synthetic gradients / Decoupled Neural Interfaces (Jaderberg et al. 2016,
 * <a href="https://arxiv.org/abs/1608.05343">arXiv:1608.05343</a>): every hidden layer has
 * a small model that <em>predicts</em> the loss gradient at its output, so the layer can
 * update as soon as its forward pass is done, without waiting for the layers above. The
 * synthesizers learn by regression: the one at the top hidden layer on the true gradient,
 * the others on the gradient that their upper neighbour's synthetic gradient implies after
 * one layer of backpropagation (bootstrapping, like TD learning).
 *
 * <p>Synthesizers are linear and start at zero, as the paper recommends; with
 * {@code conditioned} they also see the label (cDNI), which makes them far more accurate.
 */
public final class SyntheticGradients implements LocalLearner {

    private final SimpleMlp net;
    private final Optimizer optimizer;
    private final boolean conditioned;
    private final int classes;
    private final Var[] synthW;
    private final Var[] synthB;
    private float lastSynthError;

    /**
     * @param net the network, at least one hidden layer
     * @param optimizer the optimizer for the network and the synthesizers
     * @param conditioned whether synthesizers also see the one-hot label (cDNI)
     */
    public SyntheticGradients(SimpleMlp net, Optimizer optimizer, boolean conditioned) {
        this.net = net;
        this.optimizer = optimizer;
        this.conditioned = conditioned;
        int[] s = net.sizes();
        this.classes = s[s.length - 1];
        synthW = new Var[net.depth() - 1];
        synthB = new Var[net.depth() - 1];
        for (int l = 0; l < net.depth() - 1; l++) {   // predicts dLoss/dh_{l+1}
            int in = s[l + 1] + (conditioned ? classes : 0);
            synthW[l] = Var.param(Tensor.zeros(Shape.of(in, s[l + 1])));
            synthB[l] = Var.param(Tensor.zeros(Shape.of(1, s[l + 1])));
        }
    }

    /** @return the network */
    public SimpleMlp network() {
        return net;
    }

    /** @return the network's and the synthesizers' parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(net.parameters());
        for (int l = 0; l < synthW.length; l++) {
            p.add(synthW[l]);
            p.add(synthB[l]);
        }
        return p;
    }

    /** @return the synthesizers' mean squared error on the last batch */
    public float synthesizerError() {
        return lastSynthError;
    }

    private Var synth(int l, Tensor h, Tensor y) {
        Tensor in = conditioned ? Tensor.concat(1, h, y) : h;
        return Var.of(in).matmul(synthW[l]).add(synthB[l]);
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int depth = net.depth();
        int batch = inputs.shape().dim(0);
        Tensor y = LearningMath.oneHot(labels, classes);
        // forward, each layer updating at once from its synthetic gradient (per-sample scale)
        Tensor h = inputs;
        Tensor[] synthetic = new Tensor[depth - 1];
        Tensor[] implied = new Tensor[depth];   // gradient each layer's input receives
        Tensor[] outs = new Tensor[depth];
        for (int l = 0; l < depth - 1; l++) {
            Var in = Var.param(h);
            Var out = net.layer(l, in);
            outs[l] = out.value();
            synthetic[l] = synth(l, out.value(), y).value();
            out.backward(synthetic[l].div(batch));
            implied[l] = in.grad().mul(batch);
            h = out.value();
        }
        Var in = Var.param(h);
        Var loss = Functional.crossEntropy(net.layer(depth - 1, in), labels);
        loss.backward();
        implied[depth - 1] = in.grad().mul(batch);
        // synthesizer regression: true gradient on top, bootstrapped below
        Var synthLoss = null;
        for (int l = 0; l < depth - 1; l++) {
            Var term = synth(l, outs[l], y).sub(implied[l + 1]).square().mean();
            synthLoss = synthLoss == null ? term : synthLoss.add(term);
        }
        synthLoss.backward();
        lastSynthError = synthLoss.value().item() / (depth - 1);
        float value = loss.value().item();
        optimizer.step(parameters());
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return net.predict(inputs);
    }
}
