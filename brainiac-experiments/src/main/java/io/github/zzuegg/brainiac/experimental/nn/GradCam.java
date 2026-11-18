package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Grad-CAM: a class-activation map from a convolutional layer's feature maps, each weighted
 * by the spatial mean of the class score's gradient with respect to it,
 * {@code relu(sum_c alpha_c A_c)} with {@code alpha_c = mean_ij d y / d A_cij}. It shows
 * where in the image the evidence for a class lies, at the layer's resolution;
 * {@link #upsample} maps it back onto the input pixels.
 *
 * <p>Paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
 * Gradient-based Localization" (2017), <a href="https://arxiv.org/abs/1610.02391">arXiv:1610.02391</a>.
 */
public final class GradCam {

    private GradCam() {
    }

    /**
     * The class-activation maps of a batch.
     *
     * @param model a sequential model
     * @param layer the index of a layer whose output is {@code [batch, channels, h, w]}
     * @param inputs the images
     * @param output the class (output index) to explain
     * @return {@code [batch, h, w]}, non-negative
     */
    public static Tensor map(Model model, int layer, Tensor inputs, int output) {
        Var a = Var.param(LayerHooks.activations(model, inputs).get(layer));
        Var y = LayerHooks.resume(model, layer, a).slice(1, output, output + 1).sum();
        Tensor g = Var.gradients(y, List.of(a), false).get(0).value();
        Tensor alpha = g.mean(3, true).mean(2, true);
        return alpha.mul(a.value()).sum(1, false).relu();
    }

    /**
     * Maps class-activation maps back onto the input grid by nearest neighbour: a map cell
     * covers {@code stride} input pixels per axis, and the map is centered (unpadded
     * convolutions shrink it evenly at every border).
     *
     * @param cam {@code [batch, h, w]}
     * @param side the input side
     * @param stride input pixels per map cell: 1 after stride-1 convolutions, 2 after one 2x2 pooling
     * @return {@code [batch, side, side]}
     */
    public static Tensor upsample(Tensor cam, int side, int stride) {
        int n = cam.shape().dim(0);
        int h = cam.shape().dim(1);
        int w = cam.shape().dim(2);
        float[] v = cam.toArray();
        float[] out = new float[n * side * side];
        int oy = (side - stride * h) / 2;
        int ox = (side - stride * w) / 2;
        for (int b = 0; b < n; b++) {
            for (int i = 0; i < side; i++) {
                int ci = Math.clamp(Math.floorDiv(i - oy, stride), 0, h - 1);
                for (int j = 0; j < side; j++) {
                    int cj = Math.clamp(Math.floorDiv(j - ox, stride), 0, w - 1);
                    out[(b * side + i) * side + j] = v[(b * h + ci) * w + cj];
                }
            }
        }
        return Tensor.of(Shape.of(n, side, side), out);
    }
}
