package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Temporal efficient training (TET, Deng et al., ICLR 2022,
 * <a href="https://arxiv.org/abs/2202.11946">arXiv:2202.11946</a>) for spiking networks: the
 * cross-entropy is applied to the output at every time step and averaged, instead of to the
 * time-averaged output, plus a regulariser pulling every step's output towards a constant
 * {@code φ}:
 *
 * <pre>
 * L = (1 - λ) · (1/T) Σ_t CE(O_t, y) + λ · (1/T) Σ_t MSE(O_t, φ)
 * </pre>
 */
public final class TemporalEfficientLoss {

    private TemporalEfficientLoss() {
    }

    /**
     * The TET loss.
     *
     * @param outputs per-step outputs {@code [batch, steps, classes]}
     * @param labels class ids {@code [batch]}
     * @param lambda the regulariser weight, e.g. 0.05 (0 for pure TET)
     * @param phi the regulariser's target, e.g. the firing threshold
     * @return the scalar loss
     */
    public static Var loss(Var outputs, Tensor labels, float lambda, float phi) {
        Shape s = outputs.shape();
        int b = s.dim(0);
        int steps = s.dim(1);
        int c = s.dim(2);
        float[] y = labels.toArray();
        float[] repeated = new float[b * steps];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < steps; t++) {
                repeated[i * steps + t] = y[i];
            }
        }
        Var ce = Functional.crossEntropy(outputs.reshape(b * steps, c), Tensor.of(Shape.of(b * steps), repeated));
        if (lambda == 0) {
            return ce;
        }
        Var mse = outputs.sub(Tensor.full(Shape.of(), phi, outputs.value().device())).square().mean();
        return ce.mul(1 - lambda).add(mse.mul(lambda));
    }

    /**
     * The usual baseline: cross-entropy of the time-averaged output.
     *
     * @param outputs per-step outputs {@code [batch, steps, classes]}
     * @param labels class ids {@code [batch]}
     * @return the scalar loss
     */
    public static Var meanOutput(Var outputs, Tensor labels) {
        return Functional.crossEntropy(outputs.mean(1, false), labels);
    }
}
