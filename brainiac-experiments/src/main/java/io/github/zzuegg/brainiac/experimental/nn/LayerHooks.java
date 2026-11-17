package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Activation access built on core forward hooks ({@link Model#forward(Var, boolean, Model.Hook)}
 * over the steps of {@link Model#graph()}): the primitive behind steering vectors, sparse
 * autoencoders, activation patching, logit lens, probes and fact editing.
 *
 * <ul>
 *   <li>{@link #trace}: every step's output for an input;</li>
 *   <li>{@link #at}: a hook that passes one step's output through a function (to read,
 *       steer, patch or ablate it);</li>
 *   <li>{@link #resume}: run the rest of a sequential model from a given layer's (possibly
 *       edited) output, without recomputing the layers before it.</li>
 * </ul>
 *
 * <p>For a sequential model, step {@code i} is layer {@code i}.
 */
public final class LayerHooks {

    private LayerHooks() {
    }

    /**
     * Every step's output, in order.
     *
     * @param model a model
     * @param input a batch
     * @return {@code outputs.get(i)} is step {@code i}'s output; the last is the model's
     */
    public static List<Var> trace(Model model, Var input) {
        List<Var> out = new ArrayList<>();
        model.forward(input, false, (step, value) -> {
            out.add(value);
            return value;
        });
        return out;
    }

    /**
     * Every step's output values, in order.
     *
     * @param model a model
     * @param input a batch
     * @return the outputs
     */
    public static List<Tensor> activations(Model model, Tensor input) {
        return trace(model, Var.of(input)).stream().map(Var::value).toList();
    }

    /**
     * A hook that passes one step's output through a function and leaves the others alone.
     *
     * @param step the step, an index into {@link Model#graph()}
     * @param function applied to that step's output
     * @return the hook, for {@link Model#forward(Var, boolean, Model.Hook)}
     */
    public static Model.Hook at(int step, UnaryOperator<Var> function) {
        return (s, value) -> s.index() == step ? function.apply(value) : value;
    }

    /**
     * Runs the layers after {@code layer} on a value standing in for its output.
     *
     * @param model a sequential model
     * @param layer the layer whose output {@code activation} replaces, -1 for the input
     * @param activation the value
     * @return the model's output
     */
    public static Var resume(Model model, int layer, Var activation) {
        List<Layer> layers = model.layers();
        Var x = activation;
        for (int i = layer + 1; i < layers.size(); i++) {
            x = layers.get(i).forward(x, false);
        }
        return x;
    }
}
