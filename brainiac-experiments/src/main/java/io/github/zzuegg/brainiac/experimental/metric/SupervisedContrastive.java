package io.github.zzuegg.brainiac.experimental.metric;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Supervised contrastive learning, SupCon (Khosla et al., NeurIPS 2020): InfoNCE extended to
 * labels. Every other sample of the batch with the same label (including the other augmented
 * view of the same image) is a positive, every sample of another class a negative. With
 * normalised embeddings {@code z} and temperature {@code tau}, the loss of anchor {@code i} is
 * {@code -1/|P(i)| sum_{p in P(i)} log (exp(z_i.z_p / tau) / sum_{a != i} exp(z_i.z_a / tau))}:
 * the average over positives sits <em>outside</em> the log (the paper's better
 * "L_out" variant). Anchors without a positive are skipped.
 *
 * <p>As a core {@link Loss} the prediction is the embedding batch {@code [n, d]} (both views
 * stacked) and the target the class id of every row.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2004.11362">arXiv:2004.11362</a>.
 */
public final class SupervisedContrastive implements Loss {

    private final float temperature;

    /**
     * Creates the loss.
     *
     * @param temperature the softmax temperature, 0.1 in the paper
     */
    public SupervisedContrastive(float temperature) {
        this.temperature = temperature;
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        int n = prediction.shape().dim(0);
        int[] y = MetricMath.labels(target);
        Var z = SelfSupervisedLosses.normalize(prediction);
        float[] self = new float[n * n];
        float[] positive = new float[n * n];
        float[] rowWeight = new float[n];
        int anchors = 0;
        for (int i = 0; i < n; i++) {
            self[i * n + i] = -1e4f;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (j != i && y[j] == y[i]) {
                    positive[i * n + j] = 1;
                    count++;
                }
            }
            if (count > 0) {
                rowWeight[i] = 1f / count;
                anchors++;
            }
        }
        if (anchors == 0) {
            return prediction.mul(0f).sum();
        }
        var device = prediction.value().device();
        Var sim = z.matmul(z.transpose()).div(temperature).add(Tensor.of(Shape.of(n, n), self, device));
        Var logProb = sim.sub(sim.logsumexp(1, true));
        Var perAnchor = logProb.mul(Tensor.of(Shape.of(n, n), positive, device)).sum(1, false)
                .mul(Tensor.of(Shape.of(n), rowWeight, device));
        return perAnchor.sum().mul(-1f / anchors);
    }
}
