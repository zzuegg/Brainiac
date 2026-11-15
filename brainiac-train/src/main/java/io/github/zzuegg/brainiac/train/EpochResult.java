package io.github.zzuegg.brainiac.train;

import java.util.Map;

/**
 * What happened in one training epoch.
 *
 * @param epoch the epoch number, starting at 1
 * @param loss the mean training loss over the epoch
 * @param metrics training metrics by name
 * @param validationLoss the loss on the validation data, or {@code NaN} without validation
 * @param validationMetrics validation metrics by name; empty without validation
 * @param learningRate the optimizer's learning rate during the epoch
 */
public record EpochResult(int epoch, float loss, Map<String, Float> metrics,
                          float validationLoss, Map<String, Float> validationMetrics, float learningRate) {

    /** @return whether validation data was evaluated */
    public boolean hasValidation() {
        return !Float.isNaN(validationLoss);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(String.format("epoch %d  loss %.5f", epoch, loss));
        metrics.forEach((k, v) -> sb.append(String.format("  %s %.4f", k, v)));
        if (hasValidation()) {
            sb.append(String.format("  val_loss %.5f", validationLoss));
            validationMetrics.forEach((k, v) -> sb.append(String.format("  val_%s %.4f", k, v)));
        }
        return sb.toString();
    }
}
