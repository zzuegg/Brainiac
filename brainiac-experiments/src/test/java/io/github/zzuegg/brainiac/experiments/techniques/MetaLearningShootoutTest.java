package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningShootout.Method;
import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class MetaLearningShootoutTest {

    @Test
    void matchingAnilAndMetaSgdLearnUnseenShapeClasses() {
        FewShotEpisodes source = FewShotEpisodes.shapes(1);
        Map<Method, Result> r = new EnumMap<>(Method.class);
        r.put(Method.PROTOTYPICAL, MetaLearningShootout.run(Method.PROTOTYPICAL, source, 5, 1500, 1));
        r.put(Method.MATCHING, MetaLearningShootout.run(Method.MATCHING, source, 5, 1500, 1));
        r.put(Method.MAML, MetaLearningShootout.run(Method.MAML, source, 5, 600, 1));
        r.put(Method.ANIL, MetaLearningShootout.run(Method.ANIL, source, 5, 600, 1));
        r.put(Method.META_SGD, MetaLearningShootout.run(Method.META_SGD, source, 5, 600, 1));
        System.out.printf("2-way 5-shot unseen shapes: matching %.3f, ANIL %.3f (%d weights adapted, %d ms train), "
                + "Meta-SGD %.3f | baselines: proto %.3f, MAML %.3f (%d weights, %d ms train)%n",
                r.get(Method.MATCHING).accuracy(), r.get(Method.ANIL).accuracy(), r.get(Method.ANIL).adaptedWeights(),
                r.get(Method.ANIL).trainMillis(), r.get(Method.META_SGD).accuracy(),
                r.get(Method.PROTOTYPICAL).accuracy(), r.get(Method.MAML).accuracy(),
                r.get(Method.MAML).adaptedWeights(), r.get(Method.MAML).trainMillis());
        for (Result x : r.values()) {
            assertTrue(x.accuracy() > 0.6, r.toString());
        }
        assertTrue(r.get(Method.ANIL).adaptedWeights() < r.get(Method.MAML).adaptedWeights() / 10);
    }
}
