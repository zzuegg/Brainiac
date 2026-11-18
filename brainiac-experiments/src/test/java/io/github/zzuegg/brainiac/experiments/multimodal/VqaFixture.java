package io.github.zzuegg.brainiac.experiments.multimodal;

/**
 * The frozen LM and vision encoder shared by the VQA bridge test classes: pretrained once per
 * test JVM, so the first class to run pays for it (about 15 s).
 */
final class VqaFixture {

    private static FusionShootout.VqaSetup setup;

    private VqaFixture() {
    }

    static synchronized FusionShootout.VqaSetup setup() {
        if (setup == null) {
            setup = FusionShootout.vqaSetup(500, 100, 512, 200, 1);
        }
        return setup;
    }

    static FusionShootout.VqaScores run(FusionShootout.Bridge bridge) {
        FusionShootout.VqaSetup s = setup();
        FusionShootout.VqaScores v = FusionShootout.vqa(s, bridge, 150, 1);
        System.out.printf("VQA %-12s acc %.3f (count %.3f exists %.3f left %.3f) trainable %d text loss %.3f"
                + " (frozen LM %.3f) %d ms%n", bridge, v.accuracy(), v.count(), v.exists(), v.left(), v.trainable(),
                v.textLoss(), s.textLoss(), v.millis());
        return v;
    }
}
