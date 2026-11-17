package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.ClassicCnnShootout.Architecture;
import io.github.zzuegg.brainiac.experiments.vision.ClassicCnnShootout.Result;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ResidualRegularizersTest {

    @Test
    void shakeRegularisersTrainResidualNets() {
        Map<Architecture, Result> results = ClassicCnnShootoutTest.runAll("residual CNNs", Architecture.RESNET,
                Architecture.SHAKE_SHAKE, Architecture.SHAKE_DROP, Architecture.CONVNEXT,
                Architecture.CONVNEXT_DROPPATH);
        assertTrue(results.get(Architecture.SHAKE_SHAKE).parameters() > results.get(Architecture.RESNET).parameters(),
                "Shake-Shake has two branches");
    }
}
