package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.evolve.Evolution;
import io.github.zzuegg.brainiac.evolve.Genome;
import io.github.zzuegg.brainiac.experimental.train.PopulationEnsemble;
import io.github.zzuegg.brainiac.experimental.train.PopulationEnsemble.Uncertainty;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * X22, population ensembles: a short neuroevolution run on three overlapping clusters, then
 * the best distinct designs of its last population are trained and evaluated together.
 * Their disagreement measures uncertainty: it is compared on the test data and on inputs
 * far from any training data, and the ensemble abstains on its least certain test inputs
 * (the selective prediction, or risk-coverage, curve).
 */
public final class PopulationEnsembles {

    /**
     * The outcome.
     *
     * @param members ensemble members
     * @param fullRisk the error rate answering every test input
     * @param halfRisk the error rate answering the most certain half
     * @param aurc the area under the risk-coverage curve
     * @param inDistributionDisagreement mean mutual information on test inputs
     * @param outOfDistributionDisagreement mean mutual information on far-away inputs
     */
    public record Outcome(int members, double fullRisk, double halfRisk, double aurc,
            double inDistributionDisagreement, double outOfDistributionDisagreement) {
    }

    private PopulationEnsembles() {
    }

    private static Dataset clusters(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] centers = {{-1, -1}, {1, -1}, {0, 0.8f}};
        float[][] x = new float[n][];
        float[][] y = new float[n][3];
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            x[i] = new float[] {centers[c][0] + (float) r.nextGaussian() * 0.6f,
                centers[c][1] + (float) r.nextGaussian() * 0.6f};
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    /**
     * Runs the experiment.
     *
     * @param members how many designs form the ensemble
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int members, long seed) {
        Dataset test = clusters(300, seed + 1);
        Task task = Task.classification("clusters", clusters(300, seed), test, Shape.of(2), Shape.of(3))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.03f)).batchSize(32).epochs(30));
        Evolution evolution = Evolution.of(task).template(Genome.dense(2, Activation.TANH, 8))
                .mutations("connections,neurons,activations").population(6).generations(3).epochs(10).seed(seed);
        Evolution.Result result = evolution.run();
        PopulationEnsemble ensemble = PopulationEnsemble.fromEvolution(evolution, result, task, members, seed);
        float[][] x = new float[test.size()][];
        int[] labels = new int[test.size()];
        for (int i = 0; i < test.size(); i++) {
            x[i] = test.featuresOf(i);
            float[] t = test.targetsOf(i);
            labels[i] = t[0] == 1 ? 0 : t[1] == 1 ? 1 : 2;
        }
        PopulationEnsemble.SelectiveCurve curve = ensemble.selective(x, labels, Uncertainty.ENTROPY);
        SplittableRandom r = new SplittableRandom(seed + 2);
        float[][] far = new float[100][];
        for (int i = 0; i < far.length; i++) {
            double angle = r.nextDouble(2 * Math.PI);
            far[i] = new float[] {(float) (8 * Math.cos(angle)), (float) (8 * Math.sin(angle))};
        }
        double in = mean(ensemble.predict(x).uncertainties(Uncertainty.MUTUAL_INFORMATION));
        double out = mean(ensemble.predict(far).uncertainties(Uncertainty.MUTUAL_INFORMATION));
        return new Outcome(ensemble.size(), curve.riskAt(1), curve.riskAt(0.5), curve.aurc(), in, out);
    }

    private static double mean(double[] a) {
        return java.util.Arrays.stream(a).average().orElse(0);
    }
}
