package io.github.zzuegg.brainiac.experiments.multimodal;

import io.github.zzuegg.brainiac.experimental.nn.ContrastivePairLoss;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ZeroShotDigitsTest {

    @Test
    void imageTextTowersClassifyZeroShot() {
        ZeroShotDigits z = ZeroShotDigits.create();
        double clip = z.contrastive(ContrastivePairLoss.clip(0.1f), 1000, 1);
        double sigLip = z.contrastive(ContrastivePairLoss.sigLip(10, -10), 1000, 1);
        double probe = z.simclrLinearProbe(1000, 5, 1);
        System.out.printf("zero-shot digits with an unseen caption phrasing: CLIP %.3f, SigLIP %.3f; SimCLR + linear "
                + "probe on 5 labels per class %.3f (chance 0.100)%n", clip, sigLip, probe);
        assertTrue(clip > 0.7 && sigLip > 0.7, "both contrastive losses classify zero-shot");
        assertTrue(clip > probe && sigLip > probe, "caption supervision beats SimCLR with a few labels");
    }
}
