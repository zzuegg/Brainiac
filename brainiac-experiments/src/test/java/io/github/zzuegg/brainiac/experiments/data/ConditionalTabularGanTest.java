package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.TableSource;
import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.TabularScore;

import org.junit.jupiter.api.Test;

class ConditionalTabularGanTest {

    @Test
    void ctganTrainOnSyntheticTestOnReal() {
        TabularScore gan = SyntheticTraining.tabular(TableSource.CTGAN, 800, 64, 1);
        TabularScore copula = SyntheticTraining.tabular(TableSource.GAUSSIAN_COPULA, 0, 64, 1);
        System.out.printf("CTGAN TSTR acc %.3f rare recall %.3f balanced %.3f DCR ratio %.2f; Gaussian copula "
                + "%.3f/%.3f/%.3f/%.2f%n", gan.accuracy(), gan.rareRecall(), gan.balancedAccuracy(),
                gan.closestRecordRatio(), copula.accuracy(), copula.rareRecall(), copula.balancedAccuracy(),
                copula.closestRecordRatio());
        assertTrue(gan.rareRecall() > copula.rareRecall(), "conditional sampling keeps the rare class");
        assertTrue(gan.balancedAccuracy() > 0.52);
        assertTrue(gan.closestRecordRatio() > 0.7, "no copying");
    }
}
