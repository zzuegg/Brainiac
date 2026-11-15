package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.train.Evaluation;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Progress;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.SplittableRandom;

/**
 * Two sequence tasks: forecasting a noisy wave with a bidirectional LSTM, and reversing a
 * sequence of digits with a small transformer.
 *
 * <p>{@code ./gradlew :brainiac-examples:run --args=sequence}
 */
public final class SequenceExample {

    private SequenceExample() {
    }

    /** Steps the forecaster sees. */
    public static final int WINDOW = 24;

    /** Length of the digit sequences to reverse. */
    public static final int LENGTH = 8;

    /** Number of distinct digits. */
    public static final int DIGITS = 6;

    /**
     * Windows of a sum of two noisy sine waves; the target is the next value.
     *
     * @param samples the number of windows
     * @param seed the random seed
     * @return features {@code [WINDOW]} (one value per step) and one target
     */
    public static Dataset waves(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] x = new float[samples][WINDOW];
        float[][] y = new float[samples][1];
        for (int s = 0; s < samples; s++) {
            double phase = random.nextDouble(0, 2 * Math.PI);
            double speed = random.nextDouble(0.15, 0.35);
            for (int t = 0; t <= WINDOW; t++) {
                double v = 0.7 * Math.sin(phase + speed * t) + 0.3 * Math.sin(2.3 * (phase + speed * t))
                        + random.nextGaussian() * 0.02;
                if (t < WINDOW) {
                    x[s][t] = (float) v;
                } else {
                    y[s][0] = (float) v;
                }
            }
        }
        return Dataset.of(x, y);
    }

    /**
     * Builds the forecaster.
     *
     * @return an untrained model: {@code [WINDOW, 1]} to one value
     */
    public static Model forecaster() {
        return Model.sequential()
                .input(WINDOW, 1)
                .add(Recurrent.lstm(24).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT))
                .gru(24, false)
                .dense(1)
                .seed(3)
                .build();
    }

    /**
     * Random digit sequences, one-hot, with the reversed sequence as one-hot targets.
     *
     * @param samples the number of sequences
     * @param seed the random seed
     * @return features {@code [LENGTH * DIGITS]} and targets {@code [LENGTH * DIGITS]}
     */
    public static Dataset reversal(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] x = new float[samples][LENGTH * DIGITS];
        float[][] y = new float[samples][LENGTH * DIGITS];
        for (int s = 0; s < samples; s++) {
            for (int t = 0; t < LENGTH; t++) {
                int d = random.nextInt(DIGITS);
                x[s][t * DIGITS + d] = 1;
                y[s][(LENGTH - 1 - t) * DIGITS + d] = 1;
            }
        }
        return Dataset.of(x, y);
    }

    /**
     * Builds the transformer: project digits, add positions, two encoder blocks, and a
     * softmax over digits at every step.
     *
     * @return an untrained model: {@code [LENGTH, DIGITS]} to {@code [LENGTH, DIGITS]}
     */
    public static Model reverser() {
        return Model.sequential()
                .input(LENGTH, DIGITS)
                .dense(32)
                .positionalEncoding()
                .add(TransformerBlock.of(4).ffnSize(64))
                .add(TransformerBlock.of(4).ffnSize(64))
                .dense(DIGITS, Activation.SOFTMAX)
                .seed(5)
                .build();
    }

    /**
     * Trains the forecaster.
     *
     * @param model the model
     * @param data the training windows
     * @param epochs the number of epochs
     * @param verbose whether to print progress
     * @return the trainer, for evaluation
     */
    public static Trainer trainForecaster(Model model, Dataset data, int epochs, boolean verbose) {
        Trainer trainer = Trainer.of(model)
                .loss(Loss.MSE)
                .optimizer(Optimizer.adam(0.01f))
                .clipGradientNorm(1)
                .batchSize(32)
                .epochs(epochs)
                .metric(Metric.MEAN_ABSOLUTE_ERROR)
                .seed(1);
        if (verbose) {
            trainer.listener(Progress.every(5));
        }
        trainer.fit(data);
        return trainer;
    }

    /**
     * Trains the reverser.
     *
     * @param model the model
     * @param data the training sequences
     * @param epochs the number of epochs
     * @param verbose whether to print progress
     * @return the trainer, for evaluation
     */
    public static Trainer trainReverser(Model model, Dataset data, int epochs, boolean verbose) {
        Trainer trainer = Trainer.of(model)
                .loss(Loss.CROSS_ENTROPY)
                .optimizer(Optimizer.adam(0.003f))
                .clipGradientNorm(1)
                .batchSize(32)
                .epochs(epochs)
                .metric(Metric.ACCURACY)
                .seed(1);
        if (verbose) {
            trainer.listener(Progress.every(5));
        }
        trainer.fit(data);
        return trainer;
    }

    /**
     * Runs both tasks.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Model forecaster = forecaster();
        System.out.println(forecaster.summary());
        Trainer t1 = trainForecaster(forecaster, waves(2000, 1), 30, true);
        Evaluation e1 = t1.evaluate(waves(500, 2));
        System.out.printf("%nforecast test MSE %.5f, mean absolute error %.4f%n%n", e1.loss(),
                e1.get(Metric.MEAN_ABSOLUTE_ERROR));

        Model reverser = reverser();
        System.out.println(reverser.summary());
        Trainer t2 = trainReverser(reverser, reversal(4000, 3), 30, true);
        Evaluation e2 = t2.evaluate(reversal(1000, 4));
        System.out.printf("%nreversal test accuracy per digit %.1f%%%n", 100 * e2.get(Metric.ACCURACY));
    }
}
