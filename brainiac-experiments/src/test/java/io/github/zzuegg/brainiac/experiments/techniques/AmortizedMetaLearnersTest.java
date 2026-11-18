package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningShootout.Method;
import io.github.zzuegg.brainiac.experiments.techniques.MetaLearningShootout.Result;

import org.junit.jupiter.api.Test;

class AmortizedMetaLearnersTest {

    @Test
    void leoAdaptsInLatentSpaceAndStaysStableWithMoreSteps() {
        FewShotEpisodes source = FewShotEpisodes.shapes(1);
        Result leo1 = MetaLearningShootout.leo(source, 1, 200, 1, 5);
        Result leo5 = MetaLearningShootout.leo(source, 5, 200, 1, 5);
        Result leo5long = MetaLearningShootout.leo(source, 5, 200, 1, 20);
        Result maml1 = MetaLearningShootout.maml(Method.MAML, source, 1, 200, 1, 3);
        Result maml5 = MetaLearningShootout.maml(Method.MAML, source, 5, 200, 1, 3);
        Result maml5long = MetaLearningShootout.maml(Method.MAML, source, 5, 200, 1, 20);
        Result anil5 = MetaLearningShootout.maml(Method.ANIL, source, 5, 200, 1, 3);
        System.out.printf("LEO 1-shot %.3f, 5-shot %.3f (20 inner steps %.3f, %d latent dims) | MAML 1-shot %.3f, "
                + "5-shot %.3f (20 steps %.3f), ANIL 5-shot %.3f%n", leo1.accuracy(), leo5.accuracy(),
                leo5long.accuracy(), leo5.adaptedWeights(), maml1.accuracy(), maml5.accuracy(), maml5long.accuracy(),
                anil5.accuracy());
        assertTrue(leo5.accuracy() > 0.55 && leo1.accuracy() > 0.5, leo1 + " " + leo5);
        assertTrue(leo5long.accuracy() > leo5.accuracy() - 0.05, "latent inner loop is stable: " + leo5long);
    }

    @Test
    void cnapsAdaptsAcrossDomainsWithoutGradientSteps() {
        FewShotEpisodes source = FewShotEpisodes.mixed(1);
        Result cnaps = MetaLearningShootout.cnaps(source, 5, 1500, 1, true);
        Result fixed = MetaLearningShootout.cnaps(source, 5, 1500, 1, false);
        Result proto = MetaLearningShootout.run(Method.PROTOTYPICAL, source, 5, 1500, 1);
        Result maml = MetaLearningShootout.run(Method.MAML, source, 5, 400, 1);
        System.out.printf("CNAPs mixed shapes+digits 2-way 5-shot: %.3f (%.2f ms/episode), no FiLM %.3f | "
                + "prototypical %.3f, MAML %.3f (%.2f ms/episode)%n", cnaps.accuracy(), cnaps.millisPerEpisode(),
                fixed.accuracy(), proto.accuracy(), maml.accuracy(), maml.millisPerEpisode());
        assertTrue(cnaps.accuracy() > 0.65, cnaps.toString());
    }
}
