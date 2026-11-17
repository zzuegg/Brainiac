package io.github.zzuegg.brainiac.experiments.memory;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.SleepConsolidation;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Experiment X4: catastrophic forgetting across tasks, with and without sleep. A small
 * classifier meets several tasks one after the other (each task is two new classes of
 * Gaussian clusters, class-incremental). Awake, it learns only the current task online; with
 * sleep enabled, it sleeps after every task, replaying its pruned episodic store into the
 * weights. The result is the accuracy matrix {@code acc[i][j]} (on task {@code j} after
 * learning task {@code i}), the final average accuracy and the average forgetting
 * (best accuracy ever reached on a task minus its final accuracy).
 */
public final class SleepForgetting {

    /**
     * The outcome.
     *
     * @param accuracy {@code accuracy[i][j]}: accuracy on task {@code j} after learning task {@code i}
     * @param finalAccuracy the mean accuracy over all tasks at the end
     * @param forgetting the mean drop from each task's best accuracy to its final one
     */
    public record Result(double[][] accuracy, double finalAccuracy, double forgetting) {
    }

    private SleepForgetting() {
    }

    /**
     * Runs the experiment.
     *
     * @param tasks the number of tasks, two classes each
     * @param sleep whether the model sleeps after each task
     * @param seed the random seed
     * @return the result
     */
    public static Result run(int tasks, boolean sleep, long seed) {
        int features = 8;
        int classes = 2 * tasks;
        SplittableRandom random = new SplittableRandom(seed);
        float[][] centers = new float[classes][features];
        for (float[] c : centers) {
            for (int i = 0; i < features; i++) {
                c[i] = (float) (random.nextGaussian() * 2);
            }
        }
        Model model = Model.sequential().input(features).dense(32, Activation.RELU).dense(classes, Activation.SOFTMAX)
                .seed(seed).build();
        SleepConsolidation learner = SleepConsolidation.of(model, Loss.CROSS_ENTROPY, Optimizer.adam(0.01f), seed)
                .store(40 * tasks, SleepConsolidation.Pruning.BALANCED);
        Dataset[] tests = new Dataset[tasks];
        for (int t = 0; t < tasks; t++) {
            tests[t] = sample(centers, 2 * t, 200, random);
        }
        double[][] acc = new double[tasks][tasks];
        for (int t = 0; t < tasks; t++) {
            for (int step = 0; step < 60; step++) {
                learner.wake(sample(centers, 2 * t, 16, random));
            }
            if (sleep) {
                learner.sleep(150, 32);
            }
            for (int j = 0; j < tasks; j++) {
                acc[t][j] = accuracy(model, tests[j]);
            }
        }
        double fin = 0;
        double forget = 0;
        for (int j = 0; j < tasks; j++) {
            fin += acc[tasks - 1][j] / tasks;
            double best = 0;
            for (int i = j; i < tasks; i++) {
                best = Math.max(best, acc[i][j]);
            }
            if (j < tasks - 1) {
                forget += (best - acc[tasks - 1][j]) / (tasks - 1);
            }
        }
        return new Result(acc, fin, forget);
    }

    private static Dataset sample(float[][] centers, int firstClass, int n, SplittableRandom random) {
        float[][] x = new float[n][];
        float[][] y = new float[n][centers.length];
        for (int i = 0; i < n; i++) {
            int c = firstClass + random.nextInt(2);
            x[i] = new float[centers[c].length];
            for (int k = 0; k < x[i].length; k++) {
                x[i][k] = centers[c][k] + (float) random.nextGaussian();
            }
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    private static double accuracy(Model model, Dataset data) {
        Tensor p = model.predict(data.features()).argmax(1);
        Tensor t = data.targets().argmax(1);
        return p.equalTo(t).mean().item();
    }
}
