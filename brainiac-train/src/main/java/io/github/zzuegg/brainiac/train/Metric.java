package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** A human-friendly score reported during training, such as accuracy. */
public interface Metric {

    /**
     * Share of correct predictions. With several outputs the largest output must match the
     * one-hot target (or the class id, for targets given as ids); with one output, values above 0.5 count as 1.
     */
    Metric ACCURACY = new Metric() {
        @Override
        public String name() {
            return "accuracy";
        }

        @Override
        public float compute(Tensor prediction, Tensor target) {
            return value(prediction, target).item();
        }

        @Override
        public Tensor value(Tensor prediction, Tensor target) {
            Tensor correct;
            if (prediction.shape().dim(-1) == 1) {
                Tensor half = Tensor.full(Shape.of(), 0.5f, prediction.device());
                correct = prediction.greater(half).equalTo(target.greater(half));
            } else if (io.github.zzuegg.brainiac.nn.Loss.classIds(prediction.shape(), target.shape())) {
                Tensor best = prediction.argmax(-1);
                correct = best.equalTo(target.reshape(best.shape().toArray()));
            } else {
                correct = prediction.argmax(-1).equalTo(target.argmax(-1));
            }
            return correct.mean();
        }
    };

    /** Mean absolute difference between prediction and target. */
    Metric MEAN_ABSOLUTE_ERROR = new Metric() {
        @Override
        public String name() {
            return "mae";
        }

        @Override
        public float compute(Tensor prediction, Tensor target) {
            return value(prediction, target).item();
        }

        @Override
        public Tensor value(Tensor prediction, Tensor target) {
            return prediction.sub(target).abs().mean();
        }
    };

    /**
     * Returns the name used in results and progress output.
     *
     * @return the name
     */
    String name();

    /**
     * Scores a batch of predictions.
     *
     * @param prediction model outputs, shape {@code [batch, outputs]}
     * @param target expected outputs, same shape
     * @return the score
     */
    float compute(Tensor prediction, Tensor target);

    /**
     * Computes the metric as a single-value tensor on the predictions' device, so a training
     * loop can sum it without waiting for the device after every batch. The default wraps
     * {@link #compute}.
     *
     * @param prediction the model's output
     * @param target the expected output
     * @return the metric, a tensor with one value
     */
    default Tensor value(Tensor prediction, Tensor target) {
        return Tensor.of(compute(prediction, target));
    }
}
