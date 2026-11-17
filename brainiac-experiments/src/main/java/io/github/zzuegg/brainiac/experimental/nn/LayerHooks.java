package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Forward hooks and activation access for sequential models (chains of layers built with
 * {@code Model.sequential()}): the primitive behind steering vectors, sparse autoencoders,
 * activation patching, logit lens, probes and fact editing.
 *
 * <ul>
 *   <li>{@link #trace}: every layer's output for an input;</li>
 *   <li>{@link #resume}: run the rest of the model from a given layer's (possibly edited)
 *       output;</li>
 *   <li>{@link #withHook}: a new model sharing all weights with a {@link HookedLayer} in
 *       place of one layer.</li>
 * </ul>
 */
public final class LayerHooks {

    private LayerHooks() {
    }

    /**
     * Every layer's output, in order.
     *
     * @param model a sequential model
     * @param input a batch
     * @return {@code outputs.get(i)} is layer {@code i}'s output; the last is the model's
     */
    public static List<Var> trace(Model model, Var input) {
        List<Var> out = new ArrayList<>();
        Var x = input;
        for (Layer l : model.layers()) {
            x = l.forward(x, false);
            out.add(x);
        }
        return out;
    }

    /**
     * Every layer's output values, in order.
     *
     * @param model a sequential model
     * @param input a batch
     * @return the outputs
     */
    public static List<Tensor> activations(Model model, Tensor input) {
        return trace(model, Var.of(input)).stream().map(Var::value).toList();
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

    /**
     * A model that shares every layer (and so every weight) with {@code model}, except that
     * layer {@code layer}'s output passes through {@code hook}.
     *
     * @param model a sequential model
     * @param layer the layer to hook
     * @param hook applied to that layer's output
     * @return the hooked model
     */
    public static Model withHook(Model model, int layer, UnaryOperator<Var> hook) {
        Model.Builder b = Model.sequential().input(model.inputShape().toArray());
        List<Layer> layers = model.layers();
        for (int i = 0; i < layers.size(); i++) {
            b.layer(i == layer ? HookedLayer.of(layers.get(i), hook) : layers.get(i));
        }
        return b.build();
    }
}
