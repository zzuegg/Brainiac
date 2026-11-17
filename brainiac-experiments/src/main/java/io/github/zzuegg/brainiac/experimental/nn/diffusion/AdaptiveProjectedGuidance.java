package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Adaptive projected guidance (APG, Sadat et al., "Eliminating Oversaturation and Artifacts
 * of High Guidance Scales in Diffusion Models", <a href="https://arxiv.org/abs/2410.02416">2410.02416</a>):
 * classifier-free guidance whose update {@code cond - uncond} is, per example,
 *
 * <ol>
 *   <li>accumulated with a (negative) momentum {@code beta} across sampling steps,</li>
 *   <li>rescaled to a norm of at most {@code r},</li>
 *   <li>split into the component parallel to the conditional prediction and the orthogonal
 *       rest; the parallel part, which mostly scales the sample up (oversaturation), is
 *       down-weighted by {@code eta}.</li>
 * </ol>
 *
 * The result is {@code cond + (w - 1) (orthogonal + eta parallel)}; {@code eta = 1},
 * {@code r = infinity}, {@code beta = 0} is plain CFG. The paper applies it to the
 * <em>denoised</em> prediction, so the model should predict the clean sample (core
 * {@code Diffusion.Prediction.SAMPLE}). The momentum makes the guided denoiser stateful:
 * create one per sampling run, or call {@link Guided#reset()}.
 */
public final class AdaptiveProjectedGuidance {
    private final float weight;
    private final float eta;
    private final float maxNorm;
    private final float momentum;

    private AdaptiveProjectedGuidance(float weight, float eta, float maxNorm, float momentum) {
        this.weight = weight;
        this.eta = eta;
        this.maxNorm = maxNorm;
        this.momentum = momentum;
    }

    /**
     * Creates the rule.
     *
     * @param weight the guidance weight w
     * @param eta the weight of the parallel component (0 drops it)
     * @param maxNorm the largest update norm r, {@code Float.POSITIVE_INFINITY} for none
     * @param momentum beta; the paper uses -0.5
     * @return the rule
     */
    public static AdaptiveProjectedGuidance of(float weight, float eta, float maxNorm, float momentum) {
        return new AdaptiveProjectedGuidance(weight, eta, maxNorm, momentum);
    }

    /** A guided denoiser that carries the momentum between sampling steps. */
    public final class Guided implements Denoiser {
        private final Denoiser.Conditional model;
        private final Tensor condition;
        private final Tensor none;
        private final Tensor[] running = new Tensor[1];

        Guided(Denoiser.Conditional model, Tensor condition, Tensor none) {
            this.model = model;
            this.condition = condition;
            this.none = none;
        }

        /** Forgets the momentum, before a new sampling run. */
        public void reset() {
            running[0] = null;
        }

        @Override
        public Var apply(Var noisy, Tensor time) {
            Tensor c = model.apply(noisy, time, condition).value();
            Tensor u = model.apply(noisy, time, none).value();
            return Var.of(combine(c, u, running));
        }
    }

    /**
     * Guides a conditional model.
     *
     * @param model the model, trained with condition dropout
     * @param condition the condition per example
     * @param none the "no condition" value
     * @return a stateful denoiser for one sampling run
     */
    public Guided guided(Denoiser.Conditional model, Tensor condition, Tensor none) {
        return new Guided(model, condition, none);
    }

    /**
     * The guided prediction for one step, with explicit momentum state.
     *
     * @param cond the conditional prediction {@code [batch, ...]}
     * @param uncond the unconditional prediction
     * @param running holds the accumulated update so far ({@code null} at the start) at index 0;
     *     updated
     * @return the guided prediction
     */
    public Tensor combine(Tensor cond, Tensor uncond, Tensor[] running) {
        int batch = cond.shape().dim(0);
        int width = Math.toIntExact(cond.size() / batch);
        float[] c = cond.toArray();
        float[] diff = cond.sub(uncond).toArray();
        if (running[0] != null && momentum != 0) {
            float[] prev = running[0].toArray();
            for (int i = 0; i < diff.length; i++) {
                diff[i] += momentum * prev[i];
            }
        }
        running[0] = Tensor.of(cond.shape(), diff.clone());
        float[] out = new float[c.length];
        for (int b = 0; b < batch; b++) {
            int o = b * width;
            double norm = 0;
            for (int i = 0; i < width; i++) {
                norm += diff[o + i] * diff[o + i];
            }
            norm = Math.sqrt(norm);
            double scale = norm > maxNorm ? maxNorm / norm : 1;
            double dot = 0;
            double cc = 0;
            for (int i = 0; i < width; i++) {
                dot += diff[o + i] * scale * c[o + i];
                cc += c[o + i] * c[o + i];
            }
            double proj = cc > 1e-12 ? dot / cc : 0;
            for (int i = 0; i < width; i++) {
                double d = diff[o + i] * scale;
                double parallel = proj * c[o + i];
                double update = (d - parallel) + eta * parallel;
                out[o + i] = (float) (c[o + i] + (weight - 1) * update);
            }
        }
        return Tensor.of(Shape.of(cond.shape().toArray()), out);
    }
}
