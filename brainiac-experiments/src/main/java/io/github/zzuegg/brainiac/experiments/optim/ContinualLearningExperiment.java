package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.experimental.train.ContinualBrain;
import io.github.zzuegg.brainiac.experimental.train.ForgettingTracker;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Experiment X21: a continual-learning {@link ContinualBrain} on a sequence of "permuted
 * digits" tasks (the MNIST-like {@link ToyTasks#digits} with a different fixed pixel
 * permutation per task, the classic permuted-MNIST protocol), measuring forgetting with a
 * {@link ForgettingTracker} for every protection mode.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ContinualLearningExperiment {

    private ContinualLearningExperiment() {
    }

    /**
     * One task's data.
     *
     * @param train the training split
     * @param test the test split
     */
    public record Task(ToyTasks.Data train, ToyTasks.Data test) {
    }

    /**
     * Permuted-digits tasks.
     *
     * @param tasks how many
     * @param perClass training samples per class
     * @param seed the seed
     * @return the tasks; the first is unpermuted
     */
    public static List<Task> permutedDigits(int tasks, int perClass, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, perClass, 8, 0.05, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.05, seed, seed + 1000);
        SplittableRandom r = new SplittableRandom(seed + 7);
        List<Task> out = new ArrayList<>();
        for (int t = 0; t < tasks; t++) {
            int[] perm = new int[train.features()];
            for (int i = 0; i < perm.length; i++) {
                perm[i] = i;
            }
            for (int i = perm.length - 1; t > 0 && i > 0; i--) {
                int j = r.nextInt(i + 1);
                int tmp = perm[i];
                perm[i] = perm[j];
                perm[j] = tmp;
            }
            out.add(new Task(permute(train, perm), permute(test, perm)));
        }
        return out;
    }

    private static ToyTasks.Data permute(ToyTasks.Data d, int[] perm) {
        float[][] x = new float[d.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = new float[perm.length];
            for (int k = 0; k < perm.length; k++) {
                x[i][k] = d.x()[i][perm[k]];
            }
        }
        return new ToyTasks.Data(x, d.y(), d.classes());
    }

    /**
     * Trains a continual brain on the tasks in order and records accuracies.
     *
     * @param tasks the tasks
     * @param protection the protection mode
     * @param regionSize neurons per region
     * @param epochs epochs per task
     * @param seed the seed
     * @return the accuracy matrix and its summaries
     */
    public static ForgettingTracker run(List<Task> tasks, ContinualBrain.Protection protection, int regionSize,
            int epochs, long seed) {
        Task first = tasks.get(0);
        ContinualBrain brain = new ContinualBrain(first.train().features(), regionSize, tasks.size(), 0.3,
                protection, 500f, seed);
        ForgettingTracker tracker = new ForgettingTracker();
        for (int t = 0; t < tasks.size(); t++) {
            Task task = tasks.get(t);
            int id = brain.addTask(task.train().classes());
            Tensor x = task.train().inputs();
            Tensor y = task.train().ids();
            brain.train(id, x, y, epochs, 32, Optimizer.adam(0.01f));
            brain.finishTask(id, x.slice(0, 0, Math.min(64, task.train().size())),
                    y.slice(0, 0, Math.min(64, task.train().size())));
            double[] acc = new double[t + 1];
            for (int j = 0; j <= t; j++) {
                acc[j] = brain.accuracy(j, tasks.get(j).test().inputs(), tasks.get(j).test().ids());
            }
            tracker.record(acc);
        }
        return tracker;
    }
}
