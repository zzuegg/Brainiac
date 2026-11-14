package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Progress;
import io.github.zzuegg.brainiac.train.Trainer;
import io.github.zzuegg.brainiac.train.TrainingResult;

import java.util.SplittableRandom;

/**
 * Three interleaved spiral arms: a non-linear classification problem that needs a
 * deeper network. Shows generated data, ReLU layers, softmax and cross-entropy.
 */
public final class SpiralExample {

    private SpiralExample() {
    }

    /**
     * Generates points on spiral arms with a little noise.
     *
     * @param pointsPerArm samples per class
     * @param arms the number of classes
     * @param seed the random seed
     * @return the dataset with one-hot targets
     */
    public static Dataset data(int pointsPerArm, int arms, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] x = new float[pointsPerArm * arms][];
        float[][] y = new float[pointsPerArm * arms][];
        for (int arm = 0; arm < arms; arm++) {
            for (int i = 0; i < pointsPerArm; i++) {
                double r = (double) i / pointsPerArm;
                double angle = arm * 2 * Math.PI / arms + r * 4 + random.nextGaussian() * 0.2;
                int row = arm * pointsPerArm + i;
                x[row] = new float[] {(float) (r * Math.sin(angle)), (float) (r * Math.cos(angle))};
                y[row] = new float[arms];
                y[row][arm] = 1;
            }
        }
        return Dataset.of(x, y);
    }

    /**
     * Builds the model.
     *
     * @param arms the number of classes
     * @return an untrained model
     */
    public static Model model(int arms) {
        return Model.sequential()
                .input(2)
                .dense(64, Activation.RELU)
                .dense(64, Activation.RELU)
                .dense(arms, Activation.SOFTMAX)
                .seed(7)
                .build();
    }

    /**
     * Trains a model on spiral data.
     *
     * @param model the model
     * @param data the data
     * @param verbose whether to print progress
     * @return the training result
     */
    public static TrainingResult train(Model model, Dataset data, boolean verbose) {
        Trainer trainer = Trainer.of(model)
                .loss(Loss.CROSS_ENTROPY)
                .optimizer(Optimizer.adam(0.01f))
                .batchSize(32)
                .epochs(300)
                .metric(Metric.ACCURACY)
                .seed(7);
        if (verbose) {
            trainer.listener(Progress.every(50));
        }
        return trainer.fit(data);
    }

    /**
     * Runs the example.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Dataset data = data(100, 3, 1);
        Model model = model(3);
        System.out.println(model.summary());
        TrainingResult result = train(model, data, true);
        System.out.printf("%nfinal accuracy %.1f%%%n", 100 * result.last().metrics().get("accuracy"));
    }
}
