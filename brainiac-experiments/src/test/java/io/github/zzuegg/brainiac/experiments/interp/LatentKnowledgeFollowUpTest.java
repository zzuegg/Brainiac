package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 10.3.17: CCS with per-template normalisation, prompt ensembles and CRC (wave 9 follow-up). */
class LatentKnowledgeFollowUpTest {

    @Test
    void templatesNormalisationAndEnsembles() {
        List<LatentKnowledge.FollowUpRow> rows = LatentKnowledge.followUp(300, new int[] {1, 4}, 100, 1);
        for (LatentKnowledge.FollowUpRow r : rows) {
            System.out.println(r + " (wave 9: CCS 0.575, random 0.570, supervised 0.780)");
        }
        for (LatentKnowledge.FollowUpRow r : rows) {
            assertTrue(r.supervised() > 0.6, "truth stays linearly decodable: " + r);
            assertTrue(r.ccsPerTemplate() >= 0.5 && r.crcEnsemble() >= 0.5, "sign-free accuracies are valid");
        }
    }
}
