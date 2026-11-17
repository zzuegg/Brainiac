package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A layer that learns by itself, without any feedback: SoftHebb (Moraitis et al. 2022,
 * <a href="https://arxiv.org/abs/2107.05747">arXiv:2107.05747</a>; Journé et al. 2023,
 * <a href="https://arxiv.org/abs/2209.11883">arXiv:2209.11883</a>). Its units compete
 * through a soft winner-take-all, {@code y = softmax(x W / temperature)}, and each unit
 * moves towards the inputs it wins with Oja-like normalisation:
 * {@code dW_k = rate * y_k (x - u_k W_k)}, with {@code u = x W}. The weights converge to
 * unit-length prototypes of the input clusters, which is approximately Bayesian inference
 * in a mixture model.
 *
 * <p>Use it like any layer, for example before a {@code Dense} readout trained by a
 * {@code Trainer}: in training mode every forward pass also applies the Hebbian update, its
 * weight is a buffer (not a parameter) so no optimizer touches it, and gradients still
 * flow through it to earlier layers. The output is the soft-WTA code {@code y}.
 */
public final class SoftHebbLayer implements Layer {

    private final int units;
    private final float temperature;
    private final float rate;
    private final Var weight;

    /**
     * @param inputs features per sample
     * @param units competing units
     * @param temperature the softmax temperature; smaller is harder competition
     * @param rate the Hebbian learning rate
     * @param random the source of the initial weights
     */
    public SoftHebbLayer(int inputs, int units, float temperature, float rate, SplittableRandom random) {
        this.units = units;
        this.temperature = temperature;
        this.rate = rate;
        Tensor w = Tensor.normal(Shape.of(inputs, units), 0, 1, random);
        this.weight = Var.of(w.div(w.square().sum(0, true).sqrt()));
    }

    /**
     * A spec for model builders.
     *
     * @param units competing units
     * @param temperature the softmax temperature
     * @param rate the Hebbian learning rate
     * @return the spec
     */
    public static LayerSpec spec(int units, float temperature, float rate) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SoftHebbLayer(input.dim(0), units, temperature, rate, random);
            }
        };
    }

    /** @return the {@code [inputs, units]} weights, one prototype per column */
    public Tensor weight() {
        return weight.value();
    }

    /**
     * Applies one Hebbian update for a batch.
     *
     * @param x the inputs, {@code [batch, inputs]}
     */
    public void learn(Tensor x) {
        Tensor u = x.matmul(weight.value());
        Tensor y = softmax(u);
        Tensor dw = x.transpose().matmul(y).sub(weight.value().mul(y.mul(u).sum(0, true)));
        weight.assign(weight.value().add(dw.mul(rate / x.shape().dim(0))));
        Tensor.evaluate(List.of(weight.value()));
    }

    private Tensor softmax(Tensor u) {
        Tensor s = u.div(temperature);
        Tensor e = s.sub(s.max(1, true)).exp();
        return e.div(e.sum(1, true));
    }

    @Override
    public Var forward(Var input) {
        return Activation.softmax(-1).forward(input.matmul(weight.value()).div(temperature));
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (training) {
            learn(input.value());
        }
        return forward(input);
    }

    @Override
    public List<Var> buffers() {
        return List.of(weight);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public String toString() {
        return "SoftHebb(" + units + ", T=" + temperature + ")";
    }
}
