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
    Activation SOFTMAX = named("softmax", x -> {
        Var shifted = x.sub(x.max(-1, true).detach());
        Var e = shifted.exp();
        return e.div(e.sum(-1, true));
    });

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
     * The settings needed to recreate this activation when a model is loaded, passed to
     * the factory {@linkplain #register(String, Function) registered} under its name
     * ({@link #toString()}). Learnable parameters are not part of it; they are saved
     * with the model's other parameters.
     *
     * @return the settings; empty unless overridden
     */
    default Config config() {
        return Config.EMPTY;
    }

    /**
     * Makes an activation loadable from model files under its name ({@link #toString()}).
     * Built-in activations are registered already. A learnable template is rebuilt and
     * bound to the saved width, then its parameters are restored.
     *
     * @param activation the activation or learnable template
     */
    static void register(Activation activation) {
        Objects.requireNonNull(activation, "activation");
        register(activation.toString(), config -> activation);
    }

    /**
     * Makes activations with settings loadable from model files: the file stores the name
     * and {@link #config()}, and loading calls {@code factory} with that config.
     *
     * @param name the activation's name, as its {@link #toString()} returns it
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

        /** @return the width this copy is bound to, or -1 for a template */
        public int width() {
            return width;
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
