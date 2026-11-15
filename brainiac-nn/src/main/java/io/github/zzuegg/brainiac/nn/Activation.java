package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;

/**
 * A non-linearity, usually element-wise (softmax works per row). Activations are layers,
 * so they can also be added to a model on their own.
 *
 * <p>Only the forward function is needed: autograd works out the gradient.
 *
 * <pre>{@code
 * Activation swish = Activation.named("swish", x -> x.mul(x.sigmoid()));
 * }</pre>
 *
 * <p>An activation can also have <em>learnable parameters</em>, such as the negative slope
 * of {@link #prelu() PReLU}. Such an activation is a template: each layer that uses it
 * calls {@link #bind(int)} with its width and gets its own copy with its own parameters,
 * which are then trained along with the layer's weights.
 *
 * <pre>{@code
 * // a per-unit learnable scale: f(x) = a * tanh(x)
 * Activation scaledTanh = Activation.withParameters("scaled-tanh",
 *         width -> List.of(Tensor.ones(Shape.of(width))),
 *         (x, p) -> x.tanh().mul(p.get(0)));
 * }</pre>
 *
 * <p>To hand-write the gradient instead, for speed or for a function autograd cannot
 * express, wrap an {@link io.github.zzuegg.brainiac.autograd.Op}:
 * {@code Activation.named("cube", x -> Var.apply(CUBE_OP, x))}.
 *
 * <p>To load saved models that use a custom activation, {@linkplain #register(Activation)
 * register} it first.
 */
public interface Activation extends Layer {

    /** Passes values through unchanged. */
    Activation IDENTITY = named("identity", x -> x);

    /** {@code max(0, x)}: the usual choice for hidden layers. */
    Activation RELU = named("relu", Init.HE, Var::relu);

    /** {@code 1 / (1 + e^-x)}: squashes to (0, 1); for yes/no outputs. */
    Activation SIGMOID = named("sigmoid", Var::sigmoid);

    /** Hyperbolic tangent: squashes to (-1, 1). */
    Activation TANH = named("tanh", Var::tanh);

    /**
     * Turns each row into probabilities that sum to 1; for picking one of several
     * classes. Computed stably by subtracting the row maximum first.
     */
    Activation SOFTMAX = softmax(-1);

    /**
     * The logarithm of {@link #SOFTMAX}, computed stably; pair it with a negative
     * log-likelihood loss.
     */
    Activation LOG_SOFTMAX = logSoftmax(-1);

    /** SiLU, {@code x * sigmoid(x)}: Swish with a fixed {@code beta = 1}. */
    Activation SILU = named("silu", Init.HE, x -> x.mul(x.sigmoid()));

    /** GELU, {@code x * Phi(x)} with the exact normal CDF {@code Phi}; common in transformers. */
    Activation GELU = named("gelu", Init.HE,
            x -> x.mul(x.mul((float) (1 / Math.sqrt(2))).erf().add(1)).mul(0.5f));

    /** GELU with the faster tanh approximation of the normal CDF. */
    Activation GELU_TANH = named("gelu_tanh", Init.HE, x -> {
        Var inner = x.add(x.pow(3).mul(0.044715f)).mul((float) Math.sqrt(2 / Math.PI));
        return x.mul(inner.tanh().add(1)).mul(0.5f);
    });

    /** Softplus, {@code log(1 + e^x)}: a smooth ReLU. */
    Activation SOFTPLUS = named("softplus", Init.HE, Var::softplus);

    /** Mish, {@code x * tanh(softplus(x))}. */
    Activation MISH = named("mish", Init.HE, x -> x.mul(x.softplus().tanh()));

    /**
     * SELU: a scaled {@link #elu(float) ELU} that keeps activations normalized; use it with
     * {@link Init#LECUN}, its preferred init.
     */
    Activation SELU = named("selu", Config.EMPTY, Init.LECUN,
            x -> eluOf(x, 1.6732632423543772f).mul(1.0507009873554805f));

    /** {@code min(max(0, x), 6)}. */
    Activation RELU6 = named("relu6", Init.HE, x -> x.clamp(0, 6));

    /** {@code clamp(x, -1, 1)}: a cheap tanh. */
    Activation HARD_TANH = named("hard_tanh", x -> x.clamp(-1, 1));

    /** {@code clamp(x / 6 + 1/2, 0, 1)}: a cheap sigmoid. */
    Activation HARD_SIGMOID = named("hard_sigmoid", x -> x.div(6).add(0.5f).clamp(0, 1));

    /** {@code x * hardSigmoid(x)}: a cheap Swish. */
    Activation HARD_SWISH = named("hard_swish", Init.HE, x -> x.mul(x.div(6).add(0.5f).clamp(0, 1)));

    /** Softsign, {@code x / (1 + |x|)}. */
    Activation SOFTSIGN = named("softsign", x -> x.div(x.abs().add(1)));

    /**
     * Returns the activation to use in a layer with {@code width} units. Activations
     * without parameters return themselves. Templates for learnable activations return
     * a new copy with its own parameters; a copy that is already bound returns itself
     * for the same width, so passing one copy to several layers shares its parameters.
     *
     * @param width the number of values per sample the activation will see
     * @return the activation to use
     */
    default Activation bind(int width) {
        return this;
    }

    /**
     * The weight initialization that suits this activation, used when a dense layer is
     * created without an explicit {@link Init}.
     *
     * @return the preferred initialization; {@link Init#XAVIER} unless overridden
     */
    default Init preferredInit() {
        return Init.XAVIER;
    }

    /**
     * The name model files store and the {@linkplain #register(String, Function) registry}
     * looks up.
     *
     * @return the name; {@code toString()} unless overridden
     */
    default String name() {
        return toString();
    }

    /**
     * The width this copy is bound to, so a loaded model can bind it again before its
     * parameters are restored.
     *
     * @return the bound width, or -1 if the activation is not bound to one
     */
    default int boundWidth() {
        return -1;
    }

    /**
     * The settings needed to recreate this activation when a model is loaded, passed to
     * the factory {@linkplain #register(String, Function) registered} under its
     * {@link #name()}. Learnable parameters are not part of it; they are saved with the
     * model's other parameters.
     *
     * @return the settings; empty unless overridden
     */
    default Config config() {
        return Config.EMPTY;
    }

    /**
     * Makes an activation loadable from model files under its {@link #name()}.
     * Built-in activations are registered already. A learnable template is rebuilt and
     * bound to the saved width, then its parameters are restored.
     *
     * @param activation the activation or learnable template
     */
    static void register(Activation activation) {
        Objects.requireNonNull(activation, "activation");
        register(activation.name(), config -> activation);
    }

    /**
     * Makes activations with settings loadable from model files: the file stores the name
     * and {@link #config()}, and loading calls {@code factory} with that config.
     *
     * @param name the activation's {@link #name()}
     * @param factory recreates the activation (or learnable template) from its config
     */
    static void register(String name, Function<Config, Activation> factory) {
        Registry.activation(name, factory);
    }

    /**
     * Creates an activation from a function.
     *
     * @param name the name shown in model summaries
     * @param f the function
     * @return the activation
     */
    static Activation named(String name, UnaryOperator<Var> f) {
        return named(name, Init.XAVIER, f);
    }

    /**
     * Creates an activation from a function, with the weight initialization it prefers.
     *
     * @param name the name shown in model summaries
     * @param init the initialization dense layers use with it by default
     * @param f the function
     * @return the activation
     */
    static Activation named(String name, Init init, UnaryOperator<Var> f) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(f, "f");
        return new Activation() {
            @Override
            public Var forward(Var input) {
                return f.apply(input);
            }

            @Override
            public Init preferredInit() {
                return init;
            }

            @Override
            public String toString() {
                return name;
            }
        };
    }

    /**
     * Creates an activation with settings, so that saving a model keeps them. Register a
     * factory under the same name with {@link #register(String, Function)} to load it.
     *
     * @param name the name shown in model summaries and stored in files
     * @param config the settings, returned by {@link #config()}
     * @param init the initialization dense layers use with it by default
     * @param f the function
     * @return the activation
     */
    static Activation named(String name, Config config, Init init, UnaryOperator<Var> f) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(init, "init");
        Objects.requireNonNull(f, "f");
        return new Activation() {
            @Override
            public Var forward(Var input) {
                return f.apply(input);
            }

            @Override
            public Init preferredInit() {
                return init;
            }

            @Override
            public Config config() {
                return config;
            }

            @Override
            public String toString() {
                return name;
            }
        };
    }

    /**
     * Softmax over any axis.
     *
     * @param axis the axis whose values sum to 1; negative counts from the end
     * @return the activation
     */
    static Activation softmax(int axis) {
        return named("softmax", axis == -1 ? Config.EMPTY : Config.builder().put("axis", axis).build(), Init.XAVIER,
                x -> {
                    Var e = x.sub(x.max(axis, true).detach()).exp();
                    return e.div(e.sum(axis, true));
                });
    }

    /**
     * Log-softmax over any axis.
     *
     * @param axis the axis to normalize; negative counts from the end
     * @return the activation
     */
    static Activation logSoftmax(int axis) {
        return named("log_softmax", axis == -1 ? Config.EMPTY : Config.builder().put("axis", axis).build(),
                Init.XAVIER, x -> {
                    Var shifted = x.sub(x.max(axis, true).detach());
                    return shifted.sub(shifted.exp().sum(axis, true).log());
                });
    }

    /**
     * Leaky ReLU with the usual slope of 0.01 for negative inputs.
     *
     * @return the activation
     */
    static Activation leakyRelu() {
        return leakyRelu(0.01f);
    }

    /**
     * Leaky ReLU: {@code x} for positive inputs, {@code slope * x} otherwise. For a
     * learnable slope use {@link #prelu()}.
     *
     * @param slope the slope for negative inputs
     * @return the activation
     */
    static Activation leakyRelu(float slope) {
        return named("leaky_relu", Config.builder().put("slope", slope).build(), Init.HE,
                x -> x.relu().sub(x.neg().relu().mul(slope)));
    }

    /**
     * ELU with {@code alpha = 1}.
     *
     * @return the activation
     */
    static Activation elu() {
        return elu(1);
    }

    /**
     * ELU: {@code x} for positive inputs, {@code alpha * (e^x - 1)} otherwise.
     *
     * @param alpha the value negative inputs approach
     * @return the activation
     */
    static Activation elu(float alpha) {
        return named("elu", Config.builder().put("alpha", alpha).build(), Init.HE, x -> eluOf(x, alpha));
    }

    private static Var eluOf(Var x, float alpha) {
        // exp of min(x, 0), so large positive inputs cannot overflow into the unused branch
        return Var.where(x.greater(0), x, x.minimum(0).exp().sub(1).mul(alpha));
    }

    /**
     * Gives different groups of units different activations: the units are split into
     * equal contiguous groups (the first groups get one more unit when the width does not
     * divide evenly), one per activation.
     *
     * <pre>{@code
     * // 32 units: 16 ReLU, 16 tanh
     * model.dense(32, Activation.split(Activation.RELU, Activation.TANH));
     * }</pre>
     *
     * @param activations one activation per group; learnable ones get their own
     *     parameters for their group
     * @return the activation
     */
    static Activation split(Activation... activations) {
        return SplitActivation.of(null, activations);
    }

    /**
     * Gives different groups of units different activations, with group sizes of your
     * choice. Use sizes of 1 to choose an activation for every single unit.
     *
     * @param sizes the number of units in each group, adding up to the layer width
     * @param activations one activation per group
     * @return the activation
     */
    static Activation split(int[] sizes, Activation... activations) {
        return SplitActivation.of(Objects.requireNonNull(sizes, "sizes"), activations);
    }

    /**
     * A learned blend of several activations: every unit computes
     * {@code sum_k w_k * f_k(x)} with its own weights {@code w = softmax(logits)}, which
     * start equal and are trained with the rest of the model, so the network picks the
     * activation each unit works best with.
     *
     * @param candidates the activations to blend, at least two
     * @return the activation template
     */
    static Activation mix(Activation... candidates) {
        return MixedActivation.of(candidates);
    }

    /**
     * Creates a learnable activation. Each layer using it gets its own parameters,
     * created by {@code init} from the layer's width.
     *
     * @param name the name shown in model summaries
     * @param init creates the initial parameter values for a given width; shapes must
     *     broadcast against {@code [batch, width]}, e.g. {@code [width]} for one value
     *     per unit or {@code [1]} for one shared value
     * @param f the function of the input and the parameters, in the order {@code init}
     *     returned them
     * @return the activation template
     */
    static Activation withParameters(String name, IntFunction<List<Tensor>> init,
            BiFunction<Var, List<Var>, Var> f) {
        return withParameters(name, Init.XAVIER, init, f);
    }

    /**
     * Creates a learnable activation with the weight initialization it prefers.
     *
     * @param name the name shown in model summaries
     * @param weightInit the initialization dense layers use with it by default
     * @param init creates the initial parameter values for a given width
     * @param f the function of the input and the parameters
     * @return the activation template
     * @see #withParameters(String, IntFunction, BiFunction)
     */
    static Activation withParameters(String name, Init weightInit, IntFunction<List<Tensor>> init,
            BiFunction<Var, List<Var>, Var> f) {
        return new Parametric(name, weightInit, init, f, -1, List.of());
    }

    /**
     * Parametric ReLU: {@code x} for positive inputs, {@code a * x} otherwise, with a
     * learnable slope {@code a} per unit that starts at 0.25.
     *
     * @return the activation template
     */
    static Activation prelu() {
        return prelu(0.25f);
    }

    /**
     * Parametric ReLU with a learnable slope per unit.
     *
     * @param initialSlope the starting slope for negative inputs
     * @return the activation template
     */
    static Activation prelu(float initialSlope) {
        return withParameters("prelu", Init.HE,
                width -> List.of(Tensor.full(Shape.of(width), initialSlope)),
                (x, p) -> x.relu().sub(x.neg().relu().mul(p.get(0))));
    }

    /**
     * Swish with a learnable sharpness {@code beta} per unit: {@code x * sigmoid(beta * x)}.
     * {@code beta} starts at 1, where it equals SiLU.
     *
     * @return the activation template
     */
    static Activation swish() {
        return swish(1);
    }

    /**
     * Swish with a learnable sharpness per unit.
     *
     * @param initialBeta the starting value of {@code beta}
     * @return the activation template
     */
    static Activation swish(float initialBeta) {
        return withParameters("swish", Init.HE,
                width -> List.of(Tensor.full(Shape.of(width), initialBeta)),
                (x, p) -> x.mul(x.mul(p.get(0)).sigmoid()));
    }

    /** A learnable activation: a template when {@code width < 0}, otherwise a bound copy. */
    final class Parametric implements Activation {

        private final String name;
        private final Init weightInit;
        private final IntFunction<List<Tensor>> init;
        private final BiFunction<Var, List<Var>, Var> f;
        private final int width;
        private final List<Var> parameters;

        private Parametric(String name, Init weightInit, IntFunction<List<Tensor>> init,
                BiFunction<Var, List<Var>, Var> f, int width, List<Var> parameters) {
            this.name = Objects.requireNonNull(name, "name");
            this.weightInit = Objects.requireNonNull(weightInit, "weightInit");
            this.init = Objects.requireNonNull(init, "init");
            this.f = Objects.requireNonNull(f, "f");
            this.width = width;
            this.parameters = parameters;
        }

        @Override
        public Activation bind(int width) {
            if (width < 1) {
                throw new IllegalArgumentException(name + " needs a width of at least 1, got " + width);
            }
            if (this.width == width) {
                return this;
            }
            if (this.width >= 0) {
                throw new IllegalArgumentException(
                        name + " is bound to width " + this.width + " and cannot be used with width " + width);
            }
            List<Var> params = new ArrayList<>();
            for (Tensor t : init.apply(width)) {
                params.add(Var.param(t));
            }
            return new Parametric(name, weightInit, init, f, width, List.copyOf(params));
        }

        @Override
        public Var forward(Var input) {
            if (width < 0) {
                throw new IllegalStateException(name + " has learnable parameters and must be bound to a width first;"
                        + " use it in a dense(...) or activation(...) step, or call bind(width)");
            }
            Shape s = input.shape();
            if (s.dim(s.rank() - 1) != width) {
                throw new IllegalArgumentException(name + " is bound to width " + width + ", got input " + s);
            }
            return f.apply(input, parameters);
        }

        @Override
        public List<Var> parameters() {
            return parameters;
        }

        @Override
        public Init preferredInit() {
            return weightInit;
        }

        @Override
        public int boundWidth() {
            return width;
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
