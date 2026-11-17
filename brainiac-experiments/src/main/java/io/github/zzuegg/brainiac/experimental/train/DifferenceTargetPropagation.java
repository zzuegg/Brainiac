package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Difference Target Propagation (Lee, Zhang, Fischer and Bengio 2015,
 * <a href="https://arxiv.org/abs/1412.7525">arXiv:1412.7525</a>): instead of gradients,
 * every hidden layer receives a <em>target</em> activity and moves its output towards it
 * with a purely local squared error.
 *
 * <ul>
 *   <li>The last hidden layer's target is a small gradient step on the loss:
 *       {@code t_{L-1} = h_{L-1} - eta * dLoss/dh_{L-1}}.</li>
 *   <li>Lower targets go through learned approximate inverses {@code g_l} of the layers,
 *       with the difference correction that cancels the inverse's error:
 *       {@code t_{l-1} = h_{l-1} + g_l(t_l) - g_l(h_l)}.</li>
 *   <li>Each inverse {@code g_l(h) = tanh(h V_l + c_l)} is trained as a denoising
 *       auto-encoder of its layer: {@code |g_l(f_l(h + noise)) - (h + noise)|^2}.</li>
 * </ul>
 *
 * <p>Because nothing is differentiated through the hidden layers, they may be
 * non-differentiable or quantized. Hidden layers should be tanh (the inverses are tanh).
 */
public final class DifferenceTargetPropagation implements LocalLearner {

    private final SimpleMlp net;
    private final Optimizer optimizer;
    private final Var[] inverseW;
    private final Var[] inverseB;
    private final float targetStep;
    private final float noise;
    private final SplittableRandom random;

    /**
     * @param net a network with tanh hidden layers and at least two hidden layers
     * @param optimizer the optimizer for the forward layers and the inverses
     * @param targetStep {@code eta}, the gradient step that sets the top target, for example 0.5
     * @param noise the standard deviation of the inverse's denoising noise, for example 0.1
     * @param seed the random seed
     */
    public DifferenceTargetPropagation(SimpleMlp net, Optimizer optimizer, float targetStep, float noise, long seed) {
        if (net.depth() < 3) {
            throw new IllegalArgumentException("target propagation needs at least two hidden layers");
        }
        this.net = net;
        this.optimizer = optimizer;
        this.targetStep = targetStep;
        this.noise = noise;
        this.random = new SplittableRandom(seed);
        int[] s = net.sizes();
        inverseW = new Var[net.depth()];
        inverseB = new Var[net.depth()];
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int l = 1; l < net.depth() - 1; l++) {   // inverse of weight layer l: h_{l+1} -> h_l
            inverseW[l] = Var.param(Tensor.normal(Shape.of(s[l + 1], s[l]), 0, (float) Math.sqrt(1.0 / s[l + 1]), r));
            inverseB[l] = Var.param(Tensor.zeros(Shape.of(1, s[l])));
        }
    }

    /** @return the network */
    public SimpleMlp network() {
        return net;
    }

    /** @return the network's parameters and the inverses' */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(net.parameters());
        for (int l = 1; l < net.depth() - 1; l++) {
            p.add(inverseW[l]);
            p.add(inverseB[l]);
        }
        return p;
    }

    private Var inverse(int l, Var h) {
        return h.matmul(inverseW[l]).add(inverseB[l]).tanh();
    }

    /**
     * How well inverse {@code l} reconstructs its layer's input on a batch.
     *
     * @param inputs network inputs
     * @param l a weight layer with an inverse, {@code 1 <= l < depth - 1}
     * @return the mean squared reconstruction error
     */
    public float reconstructionError(Tensor inputs, int l) {
        Tensor[] h = net.activations(inputs);
        return inverse(l, Var.of(h[l + 1])).value().sub(h[l]).square().mean().item();
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int depth = net.depth();
        int batch = inputs.shape().dim(0);
        Tensor[] h = net.activations(inputs);
        // the top layer learns from its true gradient, and gives the top hidden target
        Var top = Var.param(h[depth - 1]);
        Var loss = Functional.crossEntropy(net.layer(depth - 1, top), labels);
        loss.backward();
        float value = loss.value().item();
        Tensor[] target = new Tensor[depth];
        target[depth - 1] = h[depth - 1].sub(top.grad().mul(targetStep * batch));
        for (int k = depth - 1; k >= 2; k--) {
            Tensor gTarget = inverse(k - 1, Var.of(target[k])).value();
            Tensor gH = inverse(k - 1, Var.of(h[k])).value();
            target[k - 1] = h[k - 1].add(gTarget).sub(gH);
        }
        Var local = null;
        for (int l = 0; l < depth - 1; l++) {
            Var out = net.layer(l, Var.of(h[l]));   // input detached: local loss only
            Var layerLoss = out.sub(target[l + 1]).square().sum(1, false).mean().mul(0.5f);
            local = local == null ? layerLoss : local.add(layerLoss);
        }
        for (int l = 1; l < depth - 1; l++) {
            Tensor corrupted = h[l].add(Tensor.normal(h[l].shape(), 0, noise, random));
            Tensor forward = net.layer(l, Var.of(corrupted)).value();
            Var rec = inverse(l, Var.of(forward)).sub(corrupted).square().sum(1, false).mean().mul(0.5f);
            local = local.add(rec);
        }
        local.backward();
        optimizer.step(parameters());
        return value;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return net.predict(inputs);
    }
}
