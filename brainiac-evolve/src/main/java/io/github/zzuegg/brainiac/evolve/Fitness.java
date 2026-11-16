package io.github.zzuegg.brainiac.evolve;

/**
 * Turns how a briefly trained candidate did into one number to maximize.
 *
 * <pre>{@code
 * // accuracy, minus 0.01 per thousand parameters
 * Fitness f = Fitness.metric().minusPerParameter(0.01 / 1000);
 * }</pre>
 */
@FunctionalInterface
public interface Fitness {

    /**
     * How a candidate did on the validation data after its training budget.
     *
     * @param metric the task's metric: accuracy for classification, mean absolute error
     *     for regression
     * @param higherIsBetter whether a higher metric is better (false for errors)
     * @param loss the validation loss
     * @param parameters the parameters the design really uses ({@link Genome#effectiveParameters})
     * @param connections the hidden connections
     * @param seconds the training time
     */
    record Score(double metric, boolean higherIsBetter, double loss, long parameters, long connections,
                 double seconds) {
    }

    /**
     * @param score how the candidate did
     * @return its fitness; higher is better. NaN counts as worst.
     */
    double of(Score score);

    /**
     * The task's metric, negated when lower is better: accuracy for classification,
     * minus the mean absolute error for regression.
     *
     * @return the fitness
     */
    static Fitness metric() {
        return s -> s.higherIsBetter() ? s.metric() : -s.metric();
    }

    /**
     * Minus the validation loss.
     *
     * @return the fitness
     */
    static Fitness loss() {
        return s -> -s.loss();
    }

    /**
     * This fitness minus a penalty per parameter, to prefer smaller designs.
     *
     * @param penalty subtracted per parameter, e.g. {@code 0.01 / 1000} to trade one
     *     percent of accuracy for a thousand parameters
     * @return the fitness
     */
    default Fitness minusPerParameter(double penalty) {
        return s -> of(s) - penalty * s.parameters();
    }

    /**
     * This fitness minus a penalty per second of training, to prefer faster designs.
     *
     * @param penalty subtracted per second
     * @return the fitness
     */
    default Fitness minusPerSecond(double penalty) {
        return s -> of(s) - penalty * s.seconds();
    }
}
