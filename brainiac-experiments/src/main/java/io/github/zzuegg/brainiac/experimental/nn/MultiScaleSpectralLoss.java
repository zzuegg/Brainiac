package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The multi-scale spectral loss of DDSP: for several FFT sizes, the mean absolute
 * difference of STFT magnitudes plus the mean absolute difference of their logs, summed
 * over scales. Phase is ignored, so a renderer is free to choose its own phases, and the
 * several resolutions trade time against frequency detail. Built on core {@link Signal#stft}.
 *
 * <p>Paper: Engel et al., "DDSP: Differentiable Digital Signal Processing" (ICLR 2020),
 * <a href="https://arxiv.org/abs/2001.04643">arXiv:2001.04643</a>; the multi-resolution STFT
 * loss also follows Yamamoto et al., "Parallel WaveGAN" (2020).
 */
public final class MultiScaleSpectralLoss {

    private MultiScaleSpectralLoss() {
    }

    /**
     * The STFT magnitude.
     *
     * @param x signals {@code [..., samples]}
     * @param nFft the frame length (hop is a quarter of it)
     * @return {@code [..., frames, nFft / 2 + 1]}
     */
    public static Var magnitude(Var x, int nFft) {
        return Signal.power(Signal.stft(x, nFft, nFft / 4)).add(1e-7f).sqrt();
    }

    /**
     * The loss.
     *
     * @param predicted signals {@code [b, samples]}
     * @param target signals of the same shape
     * @param scales FFT sizes, such as {@code {64, 128, 256}}
     * @return a scalar
     */
    public static Var loss(Var predicted, Var target, int... scales) {
        Var total = null;
        for (int n : scales) {
            Var p = magnitude(predicted, n);
            Var t = magnitude(target, n).detach();
            Var term = p.sub(t).abs().mean().add(p.add(1e-5f).log().sub(t.add(1e-5f).log()).abs().mean());
            total = total == null ? term : total.add(term);
        }
        return total;
    }

    /**
     * The loss's value for plain arrays, a distance for evaluation.
     *
     * @param predicted samples
     * @param target samples of the same length
     * @param scales FFT sizes
     * @return the distance
     */
    public static double distance(float[] predicted, float[] target, int... scales) {
        Var p = Var.of(Tensor.of(Shape.of(1, predicted.length), predicted));
        Var t = Var.of(Tensor.of(Shape.of(1, target.length), target));
        return Var.noGrad(() -> loss(p, t, scales)).value().item();
    }
}
