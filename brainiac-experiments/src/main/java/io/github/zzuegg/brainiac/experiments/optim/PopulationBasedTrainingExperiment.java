package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.PopulationBasedTraining;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.HyperparameterSearch;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Population-based training against from-scratch hyperparameter search on a
 * learning-rate-sensitive task: the noisy teacher regression of
 * {@code TrainingTricksExperiment.schedules} (a tanh teacher plus label noise, where a
 * decaying learning rate wins), with Adam's learning rate searched over
 * {@code [1e-4, 0.3]} (log-uniform). Random search, Hyperband and TPE each train every
 * configuration from scratch at a constant rate; {@link PopulationBasedTraining} trains a
 * population once and evolves the rate during training. Scores are the test mean squared
 * error; compute is the total number of training steps.
 */
public final class PopulationBasedTrainingExperiment {

    /** Steps per unit of search budget. */
    public static final int UNIT = 50;

    /**
     * One search.
     *
     * @param score the best test error found
     * @param steps the training steps spent
     * @param detail what was found (a configuration or a schedule)
     */
    public record Outcome(double score, long steps, String detail) {
    }

    private PopulationBasedTrainingExperiment() {
    }

    private static Map<String, HyperparameterSearch.Range> space() {
        return Map.of("lr", new HyperparameterSearch.LogUniform(1e-4, 0.3));
    }

    /** A model, its optimizer and its data stream, trained a few steps at a time. */
    private static final class Regression implements PopulationBasedTraining.Member {
        private final Model model;
        private final Optimizer optimizer;
        private final Tensor teacher;
        private final SplittableRandom data;

        Regression(Model model, Optimizer optimizer, Tensor teacher, SplittableRandom data) {
            this.model = model;
            this.optimizer = optimizer;
            this.teacher = teacher;
            this.data = data;
        }

        static Regression fresh(long seed, long dataSeed) {
            Model m = Model.sequential().input(8).dense(32, Activation.TANH).dense(1, Activation.IDENTITY).seed(seed)
                    .build();
            Tensor teacher = Tensor.normal(Shape.of(8, 1), 0, 1, new SplittableRandom(99));
            return new Regression(m, Optimizer.adam(0.01f), teacher, new SplittableRandom(dataSeed));
        }

        @Override
        public double train(Map<String, Object> config, int steps) {
            optimizer.learningRate((float) (double) config.get("lr"));
            for (int s = 0; s < steps; s++) {
                Tensor x = Tensor.normal(Shape.of(16, 8), 0, 1, data);
                Tensor y = x.matmul(teacher).tanh().add(Tensor.normal(Shape.of(16, 1), 0, 0.3f, data));
                model.forward(Var.of(x)).sub(y).square().mean().backward();
                optimizer.step(model.parameters());
            }
            return testError();
        }

        double testError() {
            Tensor x = Tensor.normal(Shape.of(512, 8), 0, 1, new SplittableRandom(12345));
            return model.forward(Var.of(x)).sub(x.matmul(teacher).tanh()).square().mean().value().item();
        }

        @Override
        public PopulationBasedTraining.Member copy() {
            Model m = model.copy(model.device());
            Optimizer o = Optimizer.adam(optimizer.learningRate());
            o.loadState(m.parameters(), optimizer.state(model.parameters()));
            return new Regression(m, o, teacher, data.split());
        }
    }

    private static HyperparameterSearch.Objective objective(long seed, long[] steps) {
        return (config, budget) -> {
            steps[0] += (long) budget * UNIT;
            return Regression.fresh(seed, seed + 7).train(config, budget * UNIT);
        };
    }

    /**
     * Population-based training.
     *
     * @param population members
     * @param rounds exploit/explore rounds of {@link #UNIT} steps
     * @param seed the seed
     * @return the outcome, with the best member's learning-rate schedule
     */
    public static Outcome pbt(int population, int rounds, long seed) {
        SplittableRandom seeds = new SplittableRandom(seed);
        PopulationBasedTraining pbt = new PopulationBasedTraining(space(), seed);
        PopulationBasedTraining.Result r = pbt.run(() -> Regression.fresh(seed, seeds.nextLong()), population, rounds,
                UNIT);
        StringBuilder schedule = new StringBuilder("lr");
        for (Map<String, Object> c : r.schedule()) {
            schedule.append(String.format(" %.4f", (double) c.get("lr")));
        }
        return new Outcome(r.best(), r.steps(), schedule.toString());
    }

    /**
     * Hyperband from scratch.
     *
     * @param maxBudget the largest budget in units
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome hyperband(int maxBudget, long seed) {
        long[] steps = {0};
        HyperparameterSearch.Result r = new HyperparameterSearch(space(), seed).hyperband(objective(seed, steps),
                maxBudget, 3);
        return new Outcome(r.best().score(), steps[0], config(r));
    }

    /**
     * TPE from scratch.
     *
     * @param trials configurations
     * @param budget units per configuration
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome tpe(int trials, int budget, long seed) {
        long[] steps = {0};
        HyperparameterSearch.Result r = new HyperparameterSearch(space(), seed).tpe(objective(seed, steps), trials,
                budget, Math.max(2, trials / 3), 24);
        return new Outcome(r.best().score(), steps[0], config(r));
    }

    /**
     * Random search from scratch.
     *
     * @param trials configurations
     * @param budget units per configuration
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome random(int trials, int budget, long seed) {
        long[] steps = {0};
        HyperparameterSearch.Result r = new HyperparameterSearch(space(), seed).random(objective(seed, steps), trials,
                budget);
        return new Outcome(r.best().score(), steps[0], config(r));
    }

    private static String config(HyperparameterSearch.Result r) {
        List<HyperparameterSearch.Trial> ts = r.trials();
        return String.format("lr %.4f (%d trials)", (double) r.best().config().get("lr"), ts.size());
    }
}
