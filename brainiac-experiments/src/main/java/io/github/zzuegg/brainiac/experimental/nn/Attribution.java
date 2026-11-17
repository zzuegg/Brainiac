package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Input attribution: how much each input feature contributed to one output of a model.
 *
 * <ul>
 *   <li>{@link #saliency}: the gradient of the output with respect to the input
 *       (Simonyan et al. 2013, <a href="https://arxiv.org/abs/1312.6034">arXiv:1312.6034</a>);</li>
 *   <li>{@link #gradientTimesInput};</li>
 *   <li>{@link #integratedGradients}: the input-minus-baseline times the gradient averaged
 *       along the straight path from a baseline; it satisfies <em>completeness</em> (the
 *       attributions add up to {@code f(x) - f(baseline)}) (Sundararajan et al. 2017,
 *       <a href="https://arxiv.org/abs/1703.01365">arXiv:1703.01365</a>);</li>
 *   <li>{@link #smoothGrad}: saliency averaged over noisy copies of the input, which removes
 *       much of its noise (Smilkov et al. 2017, <a href="https://arxiv.org/abs/1706.03825">arXiv:1706.03825</a>).</li>
 * </ul>
 */
public final class Attribution {

    private Attribution() {
    }

    /**
     * The gradient of one output with respect to a batch of inputs.
     *
     * @param model the model
     * @param inputs inputs {@code [batch, features]}
     * @param output the output index
     * @return gradients, same layout as the inputs
     */
    public static float[] gradient(Model model, Tensor inputs, int output) {
        Var x = Var.param(inputs);
        model.forward(x).slice(1, output, output + 1).sum().backward();
        model.parameters().forEach(Var::zeroGrad);
        return x.grad().to(DType.FLOAT32).toArray();
    }

    /**
     * Saliency: the gradient of an output at an input.
     *
     * @param model the model
     * @param input one input
     * @param output the output index
     * @return one value per feature
     */
    public static float[] saliency(Model model, float[] input, int output) {
        return gradient(model, Tensor.of(Shape.of(1, input.length), input), output);
    }

    /**
     * Gradient times input.
     *
     * @param model the model
     * @param input one input
     * @param output the output index
     * @return one value per feature
     */
    public static float[] gradientTimesInput(Model model, float[] input, int output) {
        float[] g = saliency(model, input, output);
        for (int i = 0; i < g.length; i++) {
            g[i] *= input[i];
        }
        return g;
    }

    /**
     * Integrated gradients with a midpoint Riemann sum, all path points in one batch.
     *
     * @param model the model
     * @param input one input
     * @param baseline the reference input (often zeros)
     * @param output the output index
     * @param steps the number of path points
     * @return one value per feature, adding up to about {@code f(input) - f(baseline)}
     */
    public static float[] integratedGradients(Model model, float[] input, float[] baseline, int output, int steps) {
        int d = input.length;
        float[] path = new float[steps * d];
        for (int s = 0; s < steps; s++) {
            float a = (s + 0.5f) / steps;
            for (int i = 0; i < d; i++) {
                path[s * d + i] = baseline[i] + a * (input[i] - baseline[i]);
            }
        }
        float[] g = gradient(model, Tensor.of(Shape.of(steps, d), path), output);
        float[] out = new float[d];
        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < d; i++) {
                out[i] += g[s * d + i] / steps;
            }
        }
        for (int i = 0; i < d; i++) {
            out[i] *= input[i] - baseline[i];
        }
        return out;
    }

    /**
     * SmoothGrad: saliency averaged over Gaussian-perturbed copies of the input.
     *
     * @param model the model
     * @param input one input
     * @param output the output index
     * @param samples the number of noisy copies
     * @param sigma the noise standard deviation
     * @param seed the seed
     * @return one value per feature
     */
    public static float[] smoothGrad(Model model, float[] input, int output, int samples, float sigma, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = input.length;
        float[] batch = new float[samples * d];
        for (int s = 0; s < samples; s++) {
            for (int i = 0; i < d; i++) {
                batch[s * d + i] = input[i] + (float) r.nextGaussian() * sigma;
            }
        }
        float[] g = gradient(model, Tensor.of(Shape.of(samples, d), batch), output);
        float[] out = new float[d];
        for (int s = 0; s < samples; s++) {
            for (int i = 0; i < d; i++) {
                out[i] += g[s * d + i] / samples;
            }
        }
        return out;
    }
}
