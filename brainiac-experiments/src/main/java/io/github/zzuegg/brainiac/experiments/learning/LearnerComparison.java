package io.github.zzuegg.brainiac.experiments.learning;

import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.BackpropLearner;
import io.github.zzuegg.brainiac.experimental.train.LocalLearner;
import io.github.zzuegg.brainiac.train.Optimizer;

/**
 * Runs a learning rule and the backpropagation baseline on the same {@link ToyTasks} data
 * with the same budget and reports train and test accuracy, the experiment behind every
 * "rule X vs backprop" comparison in this package.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class LearnerComparison {

    private LearnerComparison() {
    }

    /**
     * One learner's outcome.
     *
     * @param name the learner's name
     * @param trainAccuracy accuracy on the training data
     * @param testAccuracy accuracy on held-out data
     * @param millis wall time of training
     */
    public record Result(String name, double trainAccuracy, double testAccuracy, long millis) {
        @Override
        public String toString() {
            return String.format("%-28s train %.3f  test %.3f  (%d ms)", name, trainAccuracy, testAccuracy, millis);
        }
    }

    /**
     * Trains a learner and measures it.
     *
     * @param name a label for the report
     * @param learner the learner
     * @param train the training data
     * @param test the held-out data
     * @param epochs passes over the training data
     * @param batch the batch size
     * @return the result
     */
    public static Result run(String name, LocalLearner learner, ToyTasks.Data train, ToyTasks.Data test,
            int epochs, int batch) {
        long t0 = System.nanoTime();
        learner.fit(train.inputs(), train.ids(), epochs, batch);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return new Result(name, learner.accuracy(train.inputs(), train.ids()),
                learner.accuracy(test.inputs(), test.ids()), ms);
    }

    /**
     * The backpropagation baseline: an MLP of the given sizes trained with Adam.
     *
     * @param train the training data
     * @param test the held-out data
     * @param epochs passes over the data
     * @param batch the batch size
     * @param learningRate Adam's learning rate
     * @param seed the seed
     * @param hidden the hidden sizes
     * @return the result
     */
    public static Result backprop(ToyTasks.Data train, ToyTasks.Data test, int epochs, int batch,
            float learningRate, long seed, int... hidden) {
        return run("backprop", new BackpropLearner(SimpleMlp.tanh(seed, sizes(train, hidden)),
                Optimizer.adam(learningRate)), train, test, epochs, batch);
    }

    /**
     * @param data the data, for its feature and class counts
     * @param hidden the hidden sizes
     * @return input size, hidden sizes and class count
     */
    public static int[] sizes(ToyTasks.Data data, int... hidden) {
        int[] s = new int[hidden.length + 2];
        s[0] = data.features();
        System.arraycopy(hidden, 0, s, 1, hidden.length);
        s[s.length - 1] = data.classes();
        return s;
    }
}
