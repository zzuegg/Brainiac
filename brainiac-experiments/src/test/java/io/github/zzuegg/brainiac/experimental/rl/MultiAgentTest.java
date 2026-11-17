package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Schedule;
import org.junit.jupiter.api.Test;

class MultiAgentTest {

    private static double train(ValueDecomposition.Mixer mixer) {
        TwoStepGame game = new TwoStepGame();
        ValueDecomposition learner = ValueDecomposition.of(mixer, game, RlNetworkFactory.mlp(Activation.RELU, 32))
                .epsilon(Schedule.constant(1)).seed(3);
        learner.fit(game, 2400);
        double score = learner.evaluate(game, 1);
        System.out.println("two-step game, " + mixer + ": " + score);
        return score;
    }

    @Test
    void qmixFindsTheCoordinatedOptimumThatAdditiveMixingMisses() {
        double qmix = train(ValueDecomposition.Mixer.QMIX);
        double vdn = train(ValueDecomposition.Mixer.VDN);
        double independent = train(ValueDecomposition.Mixer.INDEPENDENT);
        assertEquals(8, qmix, 1e-6);
        assertTrue(vdn <= 7 && independent <= 7, vdn + " " + independent);
    }
}
