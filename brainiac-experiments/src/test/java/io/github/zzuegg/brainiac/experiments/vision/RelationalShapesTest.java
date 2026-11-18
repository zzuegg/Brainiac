package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.RelationalShapes.Architecture;

import org.junit.jupiter.api.Test;

class RelationalShapesTest {

    @Test
    void relationNetworkAnswersRelationalQuestions() {
        double[] rn = RelationalShapes.run(Architecture.RELATION_NETWORK, 1000, 1);
        double[] mlp = RelationalShapes.run(Architecture.CNN_MLP, 1000, 1);
        System.out.printf("Sort-of-CLEVR-lite accuracy non-relational/relational: RN %.3f/%.3f, CNN+MLP %.3f/%.3f%n",
                rn[0], rn[1], mlp[0], mlp[1]);
        assertTrue(rn[0] > 0.7, "the relation network answers the presence questions");
        assertTrue(rn[1] > mlp[1], "it answers 'same class?' better than CNN+MLP");
    }
}
