package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.PopulationBasedTrainingExperiment.Outcome;

import org.junit.jupiter.api.Test;

class PopulationBasedTrainingExperimentTest {

    @Test
    void populationBasedTrainingFindsAScheduleAtTheCostOfOneRun() {
        Outcome pbt = PopulationBasedTrainingExperiment.pbt(8, 10, 1);
        Outcome hyperband = PopulationBasedTrainingExperiment.hyperband(10, 1);
        Outcome tpe = PopulationBasedTrainingExperiment.tpe(8, 10, 1);
        Outcome random = PopulationBasedTrainingExperiment.random(8, 10, 1);
        System.out.printf("teacher regression test MSE: PBT %.4f (%d steps, %s) | Hyperband %.4f (%d steps, %s), "
                + "TPE %.4f (%d steps), random %.4f (%d steps)%n", pbt.score(), pbt.steps(), pbt.detail(),
                hyperband.score(), hyperband.steps(), hyperband.detail(), tpe.score(), tpe.steps(), random.score(),
                random.steps());
        assertTrue(pbt.score() < random.score() * 1.1, "PBT is at least as good as random search");
        assertTrue(pbt.steps() <= tpe.steps(), "at no more compute than TPE");
    }
}
