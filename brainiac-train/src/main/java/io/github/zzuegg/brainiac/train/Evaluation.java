package io.github.zzuegg.brainiac.train;

import java.util.Map;

/**
 * A model's loss and metrics on a dataset.
 *
 * @param loss the mean loss
 * @param metrics metric values by name
 */
public record Evaluation(float loss, Map<String, Float> metrics) {

    /**
     * Returns one metric.
     *
     * @param metric the metric
     * @return its value
     * @throws IllegalArgumentException if the metric was not computed
     */
    public float get(Metric metric) {
        Float v = metrics.get(metric.name());
        if (v == null) {
            throw new IllegalArgumentException("metric " + metric.name() + " was not computed; add it with metric(...)");
        }
        return v;
    }
}
