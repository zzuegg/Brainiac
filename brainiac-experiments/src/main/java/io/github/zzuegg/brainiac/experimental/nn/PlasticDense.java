package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A dense projection with Hebbian fast weights (differentiable plasticity, Miconi et al.
 * 2018, <a href="https://arxiv.org/abs/1804.02464">arXiv:1804.02464</a>), run over the steps
 * of a sequence: {@code y_t = act(x_t (W + alpha . Hebb_t) + b)} and
 * {@code Hebb_t+1 = Hebb_t + eta (x_t^T y_t - Hebb_t)}. A second, plastic matrix changes
 * during inference, so the projection binds inputs to outputs it saw earlier in the same
 * sequence: short-term memory without a key/value cache.
 *
 * <p>An optional connection mask restricts both the fixed and the plastic weights, so a
 * sparse {@link Brain} projection can be made plastic with {@link #fromBrain}. Input
 * {@code [batch, steps, inputs]}, output {@code [batch, steps, units]}; the trace starts at
 * zero for every sequence.
 */
public final class PlasticDense implements Layer {

    private final int inputs;
    private final int units;
    private final Activation activation;
    private final Var w;
    private final Var alpha;
    private final Var etaRaw;
    private final Var bias;
    private final Tensor mask;
    private boolean plastic = true;

    /**
     * Creates the layer with fresh weights and every connection present.
     *
     * @param inputs the input features
     * @param units the output features
     * @param activation the activation
     * @param random the random source for the weights
     */
    public PlasticDense(int inputs, int units, Activation activation, SplittableRandom random) {
        this(inputs, units, activation, Init.XAVIER.create(Shape.of(inputs, units), inputs, units, random), null,
                random);
    }

    private PlasticDense(int inputs, int units, Activation activation, Tensor weights, Tensor mask,
            SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        this.activation = activation;
        this.mask = mask;
        this.w = Var.param(weights);
        this.alpha = Var.param(Tensor.normal(Shape.of(inputs, units), 0, 0.01f, random));
        this.etaRaw = Var.param(Tensor.full(Shape.of(1), -2f));
        this.bias = Var.param(Tensor.zeros(Shape.of(units)));
    }

    /**
     * Makes one projection of a {@link Brain} plastic: copies its weights and connection mask
     * (Hebbian traces form only where connections exist) and the target region's biases.
     *
     * @param brain the brain
     * @param projection the projection's index, as in {@link Brain#projections()}
     * @param activation the target region's activation
     * @param random the random source for the plasticity coefficients
     * @return the plastic projection
     */
    public static PlasticDense fromBrain(Brain brain, int projection, Activation activation, SplittableRandom random) {
        Brain.ProjectionStats p = brain.projections().get(projection);
        int to = -1;
        for (Brain.RegionStats r : brain.regions()) {
            if (r.name().equals(p.to())) {
                to = r.capacity();
            }
        }
        boolean[] m = brain.connectionMask(projection);
        if (to <= 0 || m.length % to != 0) {
            throw new IllegalArgumentException("cannot work out the shape of projection " + projection);
        }
        int from = m.length / to;
        float[] mv = new float[m.length];
        for (int i = 0; i < m.length; i++) {
            mv[i] = m[i] ? 1 : 0;
        }
        PlasticDense layer = new PlasticDense(from, to, activation,
                Tensor.of(Shape.of(from, to), brain.weightValues(projection)), Tensor.of(Shape.of(from, to), mv), random);
        layer.bias.assign(Tensor.of(Shape.of(to), brain.biasValues(p.to())));
        return layer;
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param units the output features
     * @param activation the activation
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PlasticDense(input.dim(-1), units, activation, random);
            }
        };
    }

    /**
     * @param on whether the plastic part is used; off, this is a plain dense layer
     * @return this layer
     */
    public PlasticDense plastic(boolean on) {
        this.plastic = on;
        return this;
    }

    /** @return the fixed weights */
    public Var weight() {
        return w;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int steps = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + inputs + "], got " + input.shape());
        }
        Var fixed = mask == null ? w : w.mul(mask);
        Var a = mask == null ? alpha : alpha.mul(mask);
        if (!plastic) {
            return activation.forward(input.matmul(fixed).add(bias));
        }
        Var eta = etaRaw.sigmoid();
        Var hebb = Var.of(Tensor.zeros(Shape.of(n, inputs, units)));
        Var[] outs = new Var[steps];
        for (int t = 0; t < steps; t++) {
            Var x = input.slice(1, t, t + 1);
            Var y = activation.forward(x.matmul(a.mul(hebb).add(fixed)).add(bias));
            hebb = hebb.add(x.transpose().matmul(y).sub(hebb).mul(eta));
            outs[t] = y;
        }
        return Var.concat(1, outs);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w, alpha, etaRaw, bias);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), units);
    }

    @Override
    public String toString() {
        return "PlasticDense(" + inputs + " -> " + units + (mask != null ? ", masked" : "") + ")";
    }
}
