package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Trainer;
import io.github.zzuegg.brainiac.train.TrainingResult;

import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

class LearnedOptimizerTest {

    private static final Function<SplittableRandom, LearnedOptimizer.Task> TASKS = r -> r.nextBoolean()
            ? LearnedOptimizer.quadratic(8, r)
            : LearnedOptimizer.linearRegression(20, 6, r);

    private static double sgd(LearnedOptimizer.Task task, int steps, double lr) {
        double[] x = task.initial();
        double[] g = new double[x.length];
        double first = task.loss(x, g);
        double sum = 0;
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < x.length; i++) {
                x[i] -= lr * g[i];
            }
            double l = task.loss(x, g);
            if (!Double.isFinite(l) || l > 1e6 * first) {
                return 20;
            }
            sum += Math.log(Math.max(l, 1e-12) / first);
        }
        return sum / steps;
    }

    @Test
    void metaTrainedRuleBeatsItsStartAndSgdOnHeldOutTasks() {
        LearnedOptimizer learned = LearnedOptimizer.metaTrain(8, TASKS, 0.05f, 30, 4, 60, 3);
        LearnedOptimizer start = LearnedOptimizer.of(8, LearnedOptimizer.initialMeta(8, 3), 0.05f);
        SplittableRandom r = new SplittableRandom(12345);
        double l = 0;
        double s = 0;
        double sg = 0;
        for (int k = 0; k < 20; k++) {
            LearnedOptimizer.Task task = TASKS.apply(r);
            l += learned.unroll(task, 30);
            s += start.unroll(task, 30);
            sg += sgd(task, 30, 0.05);
        }
        System.out.printf("mean log loss ratio: learned %.3f, untrained %.3f, sgd %.3f%n", l / 20, s / 20, sg / 20);
        assertTrue(l < s - 0.5, "learned " + l / 20 + " vs untrained " + s / 20);
        assertTrue(l < sg, "learned " + l / 20 + " vs sgd " + sg / 20);
    }

    @Test
    void isADropInOptimizerForTheTrainer() {
        LearnedOptimizer learned = LearnedOptimizer.metaTrain(8, TASKS, 0.02f, 30, 2, 25, 5);
        Model model = Model.sequential().input(2).dense(8, Activation.TANH).dense(1).seed(1).build();
        float[][] x = new float[32][];
        float[][] y = new float[32][];
        SplittableRandom r = new SplittableRandom(2);
        for (int i = 0; i < 32; i++) {
            x[i] = new float[] {(float) r.nextGaussian(), (float) r.nextGaussian()};
            y[i] = new float[] {x[i][0] - 0.5f * x[i][1]};
        }
        TrainingResult result = Trainer.of(model).loss(Loss.MSE).optimizer(learned).batchSize(32).epochs(60)
                .fit(Dataset.of(x, y));
        float first = result.history().getFirst().loss();
        float last = result.history().getLast().loss();
        assertTrue(last < 0.5f * first, first + " -> " + last);
        assertTrue(!learned.state(model.parameters()).isEmpty());
    }
}
