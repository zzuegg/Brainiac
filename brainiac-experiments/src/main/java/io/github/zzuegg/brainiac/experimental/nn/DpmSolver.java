package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * DPM-Solver++ (Lu et al., <a href="https://arxiv.org/abs/2211.01095">arXiv:2211.01095</a>),
 * the multistep second-order solver ("2M") for the diffusion ODE written in its data
 * prediction: with {@code x = x0 + sigma n} and {@code lambda = -ln sigma}, the ODE has the
 * exact solution {@code x_t = (sigma_t / sigma_s) x_s + sigma_t int e^lambda D dlambda}, so the
 * linear part is integrated exactly (an exponential integrator) and only the denoiser
 * {@code D} is approximated, by a linear extrapolation through the current and the previous
 * evaluation. One network evaluation per step, second order, and no Heun correction needed:
 * good samples in 10 to 20 evaluations. The last step is first order (as in the paper's
 * "lower order final" for few steps).
 *
 * <p>The model is any core {@link Denoiser} whose "time" is the noise level {@code sigma}
 * and whose output is the clean estimate, such as {@link EdmDenoiser}; {@link #fromVp}
 * converts a core noise-prediction diffusion model.
 */
public final class DpmSolver {
    private DpmSolver() {
    }

    /**
     * Samples with DPM-Solver++(2M).
     *
     * @param model the data-prediction model {@code D(x; sigma)}
     * @param noise standard normal noise {@code [batch, dim]}
     * @param sigmas descending noise levels ending in 0, such as {@link EdmDenoiser#karrasSigmas}
     * @return the samples; {@code sigmas.length - 1} model evaluations
     */
    public static Tensor sample(Denoiser model, Tensor noise, double[] sigmas) {
        int b = noise.shape().dim(0);
        Tensor x = noise.mul((float) sigmas[0]);
        Tensor previous = null;
        double previousH = 0;
        for (int i = 0; i + 1 < sigmas.length; i++) {
            double s = sigmas[i];
            double t = sigmas[i + 1];
            Tensor d = model.apply(Var.of(x), Tensor.full(Shape.of(b), (float) s)).value();
            if (t == 0) {
                x = d;
                break;
            }
            double h = Math.log(s / t);
            Tensor dHat = d;
            if (previous != null) {
                double r = previousH / h;
                dHat = d.mul((float) (1 + 1 / (2 * r))).sub(previous.mul((float) (1 / (2 * r))));
            }
            x = x.mul((float) (t / s)).sub(dHat.mul((float) Math.expm1(-h)));
            previous = d;
            previousH = h;
        }
        return x;
    }

    /**
     * Turns a variance-preserving noise-prediction model {@code eps(x_t, t)} with
     * {@code x_t = a x0 + sqrt(1 - a^2) n} into a data-prediction model in the
     * {@code x = x0 + sigma n} form that {@link #sample} expects, with
     * {@code sigma = sqrt(1 - a^2) / a}.
     *
     * @param epsilon the noise predictor, called with its own time per row
     * @param alphaOfSigma the signal level {@code a} for a noise level
     * @param timeOfSigma the model's time input for a noise level
     * @return the data-prediction model
     */
    public static Denoiser fromVp(Denoiser epsilon, java.util.function.DoubleUnaryOperator alphaOfSigma,
            java.util.function.DoubleUnaryOperator timeOfSigma) {
        return (x, sigma) -> {
            float[] s = sigma.toArray();
            float[] scale = new float[s.length];
            float[] time = new float[s.length];
            for (int i = 0; i < s.length; i++) {
                scale[i] = (float) alphaOfSigma.applyAsDouble(s[i]);
                time[i] = (float) timeOfSigma.applyAsDouble(s[i]);
            }
            Tensor a = GenerativeOps.column(scale);
            Var xt = x.mul(a);
            Var eps = epsilon.apply(xt, Tensor.of(Shape.of(s.length), time));
            return x.sub(eps.mul(GenerativeOps.column(s)));
        };
    }
}
