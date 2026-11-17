package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A classifier trained one batch at a time by some learning rule, so that rules without
 * a backward pass (forward-forward, predictive coding, feedback alignment, ...) and
 * ordinary backpropagation ({@link BackpropLearner}) can be swapped and compared on the
 * same data.
 */
public interface LocalLearner {

    /**
     * Learns from one batch.
     *
     * @param inputs {@code [batch, features]}
     * @param labels {@code [batch]} class ids
     * @return a loss for the batch, as the rule defines it ({@code NaN} if it has none)
     */
    float trainBatch(Tensor inputs, Tensor labels);

    /**
     * Scores every class.
     *
     * @param inputs {@code [batch, features]}
     * @return {@code [batch, classes]} scores, highest for the predicted class
     */
    Tensor predict(Tensor inputs);

    /**
     * The share of rows whose top score is at the label.
     *
     * @param inputs {@code [batch, features]}
     * @param labels {@code [batch]} class ids
     * @return the accuracy in {@code [0, 1]}
     */
    default double accuracy(Tensor inputs, Tensor labels) {
        return LearningMath.accuracy(predict(inputs), labels);
    }

    /**
     * Trains for some epochs over a fixed data set in order, batch by batch.
     *
     * @param inputs all inputs
     * @param labels all class ids
     * @param epochs passes over the data
     * @param batch the batch size
     * @return the mean loss of the last epoch
     */
    default float fit(Tensor inputs, Tensor labels, int epochs, int batch) {
        int n = inputs.shape().dim(0);
        float last = Float.NaN;
        for (int e = 0; e < epochs; e++) {
            double sum = 0;
            int count = 0;
            for (int s = 0; s < n; s += batch) {
                int end = Math.min(n, s + batch);
                sum += trainBatch(inputs.slice(0, s, end), labels.slice(0, s, end));
                count++;
            }
            last = (float) (sum / count);
        }
        return last;
    }
}
