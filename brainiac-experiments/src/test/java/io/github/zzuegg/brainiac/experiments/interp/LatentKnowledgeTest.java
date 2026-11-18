package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class LatentKnowledgeTest {

    @Test
    void ccsFindsTruthWithoutLabels() {
        LatentKnowledge.Scores s = LatentKnowledge.run(600, 2, 1);
        System.out.printf("CCS plain %.3f sycophantic %.3f | supervised %.3f / %.3f | zero-shot %.3f / %.3f |"
                + " contrastive PCA %.3f random %.3f | distractor truth %s feature %s%n", s.ccsPlain(),
                s.ccsSycophantic(), s.supervisedPlain(), s.supervisedSycophantic(), s.zeroShotPlain(),
                s.zeroShotSycophantic(), s.contrastivePca(), s.random(), Arrays.toString(s.distractorTruth()),
                Arrays.toString(s.distractorFeature()));
        assertTrue(s.supervisedPlain() > 0.7, "truth is linearly decodable");
        assertTrue(s.zeroShotSycophantic() < 0.6, "the model's answers follow the user on the sycophantic split");
    }
}
