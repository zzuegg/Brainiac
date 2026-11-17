package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.DataPruningExperiment.Method;

import org.junit.jupiter.api.Test;

class DataPruningExperimentTest {

    @Test
    void difficultyScoresFindNoisyLabelsAndPruneAgainstRandom() {
        double el2nPrecision = DataPruningExperiment.noisePrecision(Method.EL2N, 0.15, 1);
        double forgetPrecision = DataPruningExperiment.noisePrecision(Method.FORGETTING, 0.15, 1);
        double full = DataPruningExperiment.accuracy(Method.RANDOM, 1, 0, 0.15, 1);
        double random = DataPruningExperiment.accuracy(Method.RANDOM, 0.5, 0, 0.15, 1);
        double hard = DataPruningExperiment.accuracy(Method.EL2N, 0.5, 0.15, 0.15, 1);
        double easy = DataPruningExperiment.accuracy(Method.EL2N, 0.5, 0.5, 0.15, 1);
        double forgetEasy = DataPruningExperiment.accuracy(Method.FORGETTING, 0.5, 0.5, 0.15, 1);
        System.out.printf("Data pruning (15%% flipped labels): flipped share of the top-15%% EL2N %.2f, forgetting %.2f "
                + "(chance 0.15); test acc keeping 50%%: EL2N hard %.3f, EL2N easy %.3f, forgetting easy %.3f vs random "
                + "%.3f (all data %.3f)%n", el2nPrecision, forgetPrecision, hard, easy, forgetEasy, random, full);
        assertTrue(el2nPrecision > 0.3 && forgetPrecision > 0.3, "the hardest examples are mostly mislabelled");
        assertTrue(Math.max(easy, forgetEasy) > random - 0.05, "score pruning is competitive with random at 50%");
    }
}
