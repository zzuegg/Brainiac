package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A recurrent layer with differentiable plasticity (Miconi et al. 2018,
 * <a href="https://arxiv.org/abs/1804.02464">arXiv:1804.02464</a>): every recurrent
 * connection has a fixed weight {@code W} and a plastic part {@code alpha * Hebb}, where the
 * Hebbian trace changes while the layer runs:
 * <pre>
 * h_t      = tanh(x_t W_in + h_(t-1) (W + alpha . Hebb_t) + b)
 * Hebb_t+1 = Hebb_t + eta (h_(t-1)^T h_t - Hebb_t)                    (HEBB)
 * Hebb_t+1 = Hebb_t + eta (h_(t-1)^T h_t - Hebb_t . h_t^2)               (OJA)
 * </pre>
 * {@code W}, {@code alpha} (per connection) and {@code eta} are trained by gradient descent
 * through the whole sequence; {@code Hebb} starts at zero for every sequence and is a
 * short-term memory without any key/value cache: the layer can memorize patterns seen once
 * within an episode and complete them later, which a plain recurrent net of the same size
 * cannot. Input {@code [batch, steps, inputs]}, output {@code [batch, steps, units]}.
 */
public final class PlasticRecurrent implements Layer {

    /** How the Hebbian trace is updated. */
    public enum Rule {
        /** Decaying Hebb: {@code Hebb += eta (pre^T post - Hebb)}. */
        HEBB,
        /** Oja's rule, which keeps the trace bounded: {@code Hebb_ij += eta post_j (pre_i - post_j Hebb_ij)}. */
        OJA
    }

    private final int inputs;
    private final int units;
    private final Var wIn;
    private final Var w;
    private final Var alpha;
    private final Var etaRaw;
    private final Var bias;
    private final Rule rule;
    private boolean plastic = true;
    private Tensor lastTrace;

    /**
     * Creates the layer.
     *
     * @param inputs the input features
     * @param units the recurrent units
     * @param rule the plasticity rule
     * @param random the random source for the weights
     */
    public PlasticRecurrent(int inputs, int units, Rule rule, SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        this.rule = rule;
        wIn = Var.param(Init.XAVIER.create(Shape.of(inputs, units), inputs, units, random));
        w = Var.param(Tensor.normal(Shape.of(units, units), 0, 0.01f, random));
        alpha = Var.param(Tensor.normal(Shape.of(units, units), 0, 0.01f, random));
        etaRaw = Var.param(Tensor.full(Shape.of(1), -2f));
        bias = Var.param(Tensor.zeros(Shape.of(units)));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param units the recurrent units
     * @param rule the plasticity rule
     * @return the spec
     */
    public static LayerSpec spec(int units, Rule rule) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PlasticRecurrent(input.dim(-1), units, rule, random);
            }
        };
    }

    /**
     * @param on whether the plastic part is used; off, this is a plain recurrent layer (a baseline)
     * @return this layer
     */
    public PlasticRecurrent plastic(boolean on) {
        this.plastic = on;
        return this;
    }

    /** @return the learned plasticity rate {@code eta} */
    public float eta() {
        return etaRaw.value().sigmoid().item();
    }

    /** @return the Hebbian trace at the end of the last forward pass, {@code [batch, units, units]}, or null */
    public Tensor lastTrace() {
        return lastTrace;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int steps = input.shape().dim(1);
        if (input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects " + inputs + " input features, got " + input.shape());
        }
        Var drive = input.matmul(wIn).add(bias);
        Var eta = etaRaw.sigmoid();
        Var h = Var.of(Tensor.zeros(Shape.of(n, 1, units)));
        Var hebb = Var.of(Tensor.zeros(Shape.of(n, units, units)));
        Var[] outs = new Var[steps];
        for (int t = 0; t < steps; t++) {
            Var weights = plastic ? alpha.mul(hebb).add(w) : w;
            Var next = Activation.TANH.forward(drive.slice(1, t, t + 1).add(h.matmul(weights)));
            if (plastic) {
                Var delta = switch (rule) {
                    case HEBB -> h.transpose().matmul(next).sub(hebb);
                    case OJA -> h.transpose().matmul(next).sub(hebb.mul(next.square()));
                };
                hebb = hebb.add(delta.mul(eta));
            }
            h = next;
            outs[t] = next;
        }
        lastTrace = hebb.value();
        return Var.concat(1, outs);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wIn, w, alpha, etaRaw, bias);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), units);
    }

    @Override
    public String toString() {
        return "PlasticRecurrent(" + units + ", " + rule + (plastic ? "" : ", frozen") + ")";
    }
}
