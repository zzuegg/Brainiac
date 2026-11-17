package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningExperiment.Learner;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class MetaLearningExperimentTest {

    @Test
    void metaLearnedInitializationsAdaptFasterToNewSineWaves() {
        Map<Learner, Float> mse = new EnumMap<>(Learner.class);
        for (Learner l : Learner.values()) {
            mse.put(l, MetaLearningExperiment.sine(l, 1000, 1));
        }
        System.out.println(mse);
        float scratch = mse.get(Learner.SCRATCH);
        for (Learner l : new Learner[] {Learner.FOMAML, Learner.MAML, Learner.REPTILE}) {
            assertTrue(mse.get(l) < 0.7f * scratch, l + " adapts better than scratch: " + mse);
        }
    }

    @Test
    void prototypicalNetworksGeneralizeToUnseenClasses() {
        float trained = MetaLearningExperiment.prototypical(200, 2);
        float untrained = MetaLearningExperiment.prototypical(0, 2);
        System.out.println("prototypical: trained " + trained + ", untrained " + untrained);
        assertTrue(trained > untrained + 0.05f && trained > 0.7f, trained + " vs " + untrained);
    }
}
