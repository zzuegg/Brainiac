package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningShootout.Method;
import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningShootout.Result;

import org.junit.jupiter.api.Test;

class SnailMetaLearnerTest {

    @Test
    void snailClassifiesUnseenShapesFromTheEpisodeSequence() {
        FewShotEpisodes source = FewShotEpisodes.shapes(1);
        Result snail = MetaLearningShootout.run(Method.SNAIL, source, 5, 600, 1);
        Result proto = MetaLearningShootout.run(Method.PROTOTYPICAL, source, 5, 600, 1);
        System.out.printf("SNAIL 2-way 5-shot unseen shapes: %.3f (%.2f ms/episode) vs prototypical %.3f%n",
                snail.accuracy(), snail.millisPerEpisode(), proto.accuracy());
        assertTrue(snail.accuracy() > 0.6, "SNAIL learns from context: " + snail);
    }
}
