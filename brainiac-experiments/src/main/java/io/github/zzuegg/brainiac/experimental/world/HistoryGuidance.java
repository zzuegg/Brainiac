package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * History guidance for video diffusion (Song et al., ICML 2025,
 * <a href="https://arxiv.org/abs/2502.06764">arXiv:2502.06764</a>). A model trained with
 * diffusion forcing ({@link FrameDenoiser.Training#DIFFUSION_FORCING}) gives the score of the
 * next frame for any history, including none (every history frame at the maximum noise level).
 * That makes classifier-free guidance on the history itself possible, with no extra training:
 *
 * <ul>
 *   <li>{@link #vanilla}: {@code v = v(empty) + w (v(history) - v(empty))}; {@code w = 1} is
 *       plain conditional sampling, {@code w > 1} extrapolates away from the history-free
 *       prediction, which sharpens the dependence on the past (motion and object identity).
 *   <li>{@link #temporal}: guidance across time, composing the scores conditioned on several
 *       subsets of the history (here the most recent {@code 1, 2, ..., n} frames):
 *       {@code v = v(empty) + w mean_s (v(h_s) - v(empty))}, so short and long contexts both
 *       pull, and a long history cannot dominate with stale information.
 * </ul>
 *
 * <p>The paper's frequency-domain variant (guiding with a low-pass of the history) is not
 * included. The guidance is applied through {@link FrameDenoiser.Guide}, so it plugs into
 * {@link FrameDenoiser#next} and {@link FrameDenoiser#rollout}.
 */
public final class HistoryGuidance implements FrameDenoiser.Guide {

    private final float weight;
    private final boolean temporal;

    private HistoryGuidance(float weight, boolean temporal) {
        this.weight = weight;
        this.temporal = temporal;
    }

    /**
     * Vanilla history guidance.
     *
     * @param weight the guidance weight {@code w}
     * @return the guide
     */
    public static HistoryGuidance vanilla(float weight) {
        return new HistoryGuidance(weight, false);
    }

    /**
     * Temporal-subset history guidance.
     *
     * @param weight the guidance weight {@code w}
     * @return the guide
     */
    public static HistoryGuidance temporal(float weight) {
        return new HistoryGuidance(weight, true);
    }

    /** @return the guidance weight */
    public float weight() {
        return weight;
    }

    @Override
    public float[] velocity(FrameDenoiser model, Tensor history, int available, Tensor x, float level,
            float historyLevel, SplittableRandom random) {
        int b = x.shape().dim(0);
        float[] empty = Var.noGrad(() -> model.velocity(history, model.historyLevels(b, 0, 1f), x, level, random))
                .value().toArray();
        if (available == 0) {
            return empty;
        }
        float[] guided = new float[empty.length];
        int subsets = temporal ? available : 1;
        for (int s = 1; s <= subsets; s++) {
            int keep = temporal ? s : available;
            float[] cond = Var.noGrad(() -> model.velocity(history, model.historyLevels(b, keep, historyLevel), x,
                    level, random)).value().toArray();
            for (int i = 0; i < guided.length; i++) {
                guided[i] += (cond[i] - empty[i]) / subsets;
            }
        }
        for (int i = 0; i < guided.length; i++) {
            guided[i] = empty[i] + weight * guided[i];
        }
        return guided;
    }

    @Override
    public String toString() {
        return (temporal ? "TemporalHistoryGuidance(" : "HistoryGuidance(") + weight + ")";
    }
}
