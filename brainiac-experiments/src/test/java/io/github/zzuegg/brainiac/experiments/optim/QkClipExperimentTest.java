package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class QkClipExperimentTest {

    @Test
    void qkClipCapsAttentionLogitsAtHighMuonRates() {
        QkClipExperiment.Result muon = QkClipExperiment.run(0, 0.05f, 300, 1);
        QkClipExperiment.Result clip = QkClipExperiment.run(10, 0.05f, 300, 1);
        System.out.printf("MuonClip (tau 10): max logit %.1f, loss %.3f, acc %.3f (%d clipped steps) vs Muon max logit "
                + "%.1f, loss %.3f, acc %.3f%n", clip.maxLogit(), clip.finalLoss(), clip.accuracy(), clip.clips(),
                muon.maxLogit(), muon.finalLoss(), muon.accuracy());
        assertTrue(clip.maxLogit() <= 10.01f && muon.maxLogit() > 50, "QK-clip holds the logits at tau");
        assertTrue(clip.finalLoss() < muon.finalLoss() && clip.accuracy() > 0.95f, "and the task is still solved");
    }
}
