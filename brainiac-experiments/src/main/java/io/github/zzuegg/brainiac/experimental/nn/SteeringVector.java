package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Activation addition: steers a model by adding a direction to one layer's output, with no
 * training. The direction is the difference between the mean activations of two contrasting
 * sets of inputs (say, inputs of the wanted and the unwanted behaviour); adding a multiple
 * of it pushes any input toward the wanted side, and subtracting it pushes away.
 *
 * <p>Turner et al. 2023, "Steering Language Models With Activation Engineering",
 * <a href="https://arxiv.org/abs/2308.10248">arXiv:2308.10248</a>; Rimsky et al. 2024,
 * "Steering Llama 2 via Contrastive Activation Addition",
 * <a href="https://arxiv.org/abs/2312.06681">arXiv:2312.06681</a>.
 *
 * @param layer the layer whose output is steered
 * @param direction the steering direction, one value per feature of that output
 */
public record SteeringVector(int layer, float[] direction) {

    /** Copies the direction. */
    public SteeringVector {
        direction = direction.clone();
    }

    /**
     * The mean difference of a layer's outputs between two input sets. For sequence
     * outputs {@code [batch, time, features]} the mean runs over batch and time.
     *
     * @param model a sequential model
     * @param layer the layer
     * @param positive inputs showing the wanted behaviour
     * @param negative inputs showing the unwanted behaviour
     * @return the steering vector
     */
    public static SteeringVector fromContrast(Model model, int layer, Tensor positive, Tensor negative) {
        float[] p = meanActivation(model, layer, positive);
        float[] n = meanActivation(model, layer, negative);
        float[] d = new float[p.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = p[i] - n[i];
        }
        return new SteeringVector(layer, d);
    }

    private static float[] meanActivation(Model model, int layer, Tensor inputs) {
        Tensor a = LayerHooks.trace(model, Var.of(inputs)).get(layer).value();
        int features = a.shape().dim(-1);
        int rows = Math.toIntExact(a.size() / features);
        return a.reshape(rows, features).mean(0).to(DType.FLOAT32).toArray();
    }

    /**
     * A forward hook that adds {@code scale * direction} to the layer's output; run the
     * model with it through {@link Model#forward(Var, boolean, Model.Hook)}.
     *
     * @param scale the strength; negative steers away
     * @return the hook
     */
    public Model.Hook hook(float scale) {
        Tensor v = Tensor.of(Shape.of(direction.length), direction).mul(scale);
        return LayerHooks.at(layer, out -> out.add(v.to(out.value().device())));
    }

    /**
     * Runs a model with the steering applied.
     *
     * @param model the model the vector was taken from
     * @param input a batch
     * @param scale the strength; negative steers away
     * @return the model's output
     */
    public Tensor apply(Model model, Tensor input, float scale) {
        return model.forward(Var.of(input), false, hook(scale)).value();
    }

    /** @return the length of the direction */
    public double norm() {
        double s = 0;
        for (float x : direction) {
            s += x * x;
        }
        return Math.sqrt(s);
    }
}
