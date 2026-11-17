package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * "Clip denoised" for noise-predicting diffusion models, as a wrapper usable with core
 * {@code Diffusion.sample}: the wrapper turns the predicted noise into the implied clean
 * sample {@code x0 = (x - sqrt(1 - abar) eps) / sqrt(abar)}, clamps it to the data range,
 * and returns the noise consistent with the clamped {@code x0}. At the noisiest steps
 * {@code sqrt(abar)} is tiny and small errors in {@code eps} blow up {@code x0}; clamping
 * keeps sampling stable for data with known bounds such as images in [-1, 1] (Ho et al.,
 * 2020; dynamic thresholding in Imagen, <a href="https://arxiv.org/abs/2205.11487">2205.11487</a>,
 * generalizes it).
 */
public final class ClippedDenoiser {
    private ClippedDenoiser() {
    }

    /**
     * Wraps a noise predictor.
     *
     * @param model predicts the noise (core {@code Prediction.NOISE}); time is the step
     * @param schedule the schedule it was trained with
     * @param low the smallest data value
     * @param high the largest data value
     * @return the clipped predictor
     */
    public static Denoiser wrap(Denoiser model, NoiseSchedule schedule, float low, float high) {
        return (noisy, time) -> {
            Tensor eps = model.apply(noisy, time).value();
            float[] t = time.toArray();
            int batch = t.length;
            float[] a = new float[batch];
            float[] b = new float[batch];
            for (int i = 0; i < batch; i++) {
                double ab = schedule.alphaBar(Math.round(t[i]));
                a[i] = (float) Math.sqrt(ab);
                b[i] = (float) Math.sqrt(1 - ab);
            }
            int[] dims = new int[noisy.shape().rank()];
            java.util.Arrays.fill(dims, 1);
            dims[0] = batch;
            Tensor sa = Tensor.of(Shape.of(dims), a, noisy.value().device());
            Tensor sb = Tensor.of(Shape.of(dims), b, noisy.value().device());
            Tensor x = noisy.value();
            Tensor x0 = x.sub(eps.mul(sb)).div(sa).maximum(Tensor.full(Shape.of(), low))
                    .minimum(Tensor.full(Shape.of(), high));
            return Var.of(x.sub(x0.mul(sa)).div(sb));
        };
    }
}
