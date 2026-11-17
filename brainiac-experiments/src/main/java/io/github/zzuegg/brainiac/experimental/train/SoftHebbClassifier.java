package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SoftHebbLayer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The SoftHebb recipe as a {@link LocalLearner}: a {@link SoftHebbLayer} learns features
 * with no feedback at all (each batch is one Hebbian update), and only a linear readout on
 * top is trained by gradient descent on the cross-entropy, a single-layer delta rule.
 * As in the paper, inputs are scaled to unit length before the competition and the
 * readout sees the <em>triangle</em> code {@code relu(u - mean(u))} of the activations
 * {@code u = x W} rather than the near one-hot soft-WTA output, which keeps more of the
 * information about how close a sample is to every prototype.
 */
public final class SoftHebbClassifier implements LocalLearner {

    private final SoftHebbLayer features;
    private final Var readout;
    private final Var bias;
    private final Optimizer optimizer;

    /**
     * @param inputs features per sample
     * @param units soft winner-take-all units
     * @param classes classes
     * @param temperature the competition temperature
     * @param hebbianRate the Hebbian learning rate
     * @param optimizer the readout's optimizer
     * @param seed the seed
     */
    public SoftHebbClassifier(int inputs, int units, int classes, float temperature, float hebbianRate,
            Optimizer optimizer, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.features = new SoftHebbLayer(inputs, units, temperature, hebbianRate, r);
        this.readout = Var.param(Tensor.zeros(Shape.of(units, classes)));
        this.bias = Var.param(Tensor.zeros(Shape.of(1, classes)));
        this.optimizer = optimizer;
    }

    /** @return the Hebbian feature layer */
    public SoftHebbLayer features() {
        return features;
    }

    private static Tensor unit(Tensor x) {
        return x.div(x.square().sum(1, true).sqrt().add(1e-6f));
    }

    private Tensor code(Tensor x) {
        Tensor u = x.matmul(features.weight());
        return u.sub(u.mean(1, true)).relu();
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        Tensor x = unit(inputs);
        features.learn(x);
        Var loss = Functional.crossEntropy(Var.of(code(x)).matmul(readout).add(bias), labels);
        loss.backward();
        optimizer.step(List.of(readout, bias));
        return loss.value().item();
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return code(unit(inputs)).matmul(readout.value()).add(bias.value());
    }
}
