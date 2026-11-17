package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.SurrogateSpike;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A layer of leaky integrate-and-fire (LIF) spiking neurons over time. Input spikes (or
 * currents) {@code [batch, steps, inputs]} drive membrane potentials through a dense
 * synapse matrix; each step the potential leaks, integrates the input current and, when
 * it crosses the threshold, emits a spike and drops by the threshold (soft reset):
 *
 * <pre>
 * u_t = beta * u_(t-1) + W x_t + b - theta * s_(t-1)
 * s_t = [u_t &gt; theta]
 * </pre>
 *
 * <p>The output is the spike train {@code [batch, steps, units]}. Training unrolls the
 * steps (backpropagation through time) and uses a {@link SurrogateSpike} gradient for the
 * step function, the standard recipe of SNN training (e.g. snnTorch; SpikeGPT,
 * <a href="https://arxiv.org/abs/2302.13939">2302.13939</a>).
 */
public final class LifLayer implements Layer {
    private final int inputs;
    private final int units;
    private final float beta;
    private final float threshold;
    private final float slope;
    private final Var weight;
    private final Var bias;
    private Tensor lastPotential;

    private LifLayer(int inputs, int units, float beta, float threshold, float slope, SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        this.beta = beta;
        this.threshold = threshold;
        this.slope = slope;
        this.weight = Var.param(Init.XAVIER.create(Shape.of(inputs, units), inputs, units, random).mul(2f));
        this.bias = Var.param(Tensor.zeros(Shape.of(units)));
    }

    /**
     * Creates a layer.
     *
     * @param inputs the input width
     * @param units the number of neurons
     * @param beta the leak per step, in (0, 1]
     * @param threshold the firing threshold
     * @param seed the seed
     * @return the layer
     */
    public static LifLayer of(int inputs, int units, float beta, float threshold, long seed) {
        if (!(beta > 0 && beta <= 1) || !(threshold > 0)) {
            throw new IllegalArgumentException("need 0 < beta <= 1 and threshold > 0");
        }
        return new LifLayer(inputs, units, beta, threshold, 5f, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [steps, inputs]}.
     *
     * @param units the number of neurons
     * @param beta the leak
     * @param threshold the threshold
     * @return the spec
     */
    public static LayerSpec spec(int units, float beta, float threshold) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new LifLayer(input.dim(-1), units, beta, threshold, 5f, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != inputs) {
            throw new IllegalArgumentException("LifLayer expects [batch, steps, " + inputs + "], got " + s);
        }
        int b = s.dim(0);
        int steps = s.dim(1);
        Var current = input.matmul(weight).add(bias);
        Var u = Var.of(Tensor.zeros(Shape.of(b, units)).to(input.value().device()));
        Var spikes = null;
        List<Var> out = new ArrayList<>();
        for (int t = 0; t < steps; t++) {
            u = u.mul(beta).add(current.slice(1, t, t + 1).reshape(b, units));
            if (spikes != null) {
                u = u.sub(spikes.mul(threshold));   // soft reset
            }
            spikes = SurrogateSpike.spike(u.sub(threshold), slope);
            out.add(spikes.reshape(b, 1, units));
        }
        lastPotential = u.value();
        return Var.concat(1, out.toArray(Var[]::new));
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the membrane potentials after the last forward pass, {@code [batch, units]} */
    public Tensor lastPotential() {
        return lastPotential;
    }
}
