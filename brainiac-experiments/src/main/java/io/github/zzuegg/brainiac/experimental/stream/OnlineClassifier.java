package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

/**
 * A classifier that learns from a stream one example at a time, in the test-then-train protocol
 * of {@link Prequential}: it first predicts an example, then learns from its label.
 */
public interface OnlineClassifier {

    /**
     * Predicts class probabilities.
     *
     * @param sample the example (its label must not be used)
     * @return one probability per class
     */
    double[] predict(DriftingStreams.Sample sample);

    /**
     * Learns from a labelled example.
     *
     * @param sample the example
     */
    void learn(DriftingStreams.Sample sample);

    /** @return an estimate of the model's memory, in bytes */
    default long bytes() {
        return 0;
    }

    /**
     * The most probable class.
     *
     * @param p class probabilities
     * @return the argmax
     */
    static int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }
}
