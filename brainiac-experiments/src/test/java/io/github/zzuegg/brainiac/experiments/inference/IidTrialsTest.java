package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class IidTrialsTest {

    @Test
    void oneLikelihoodAndOneScoreServeEveryTrialCount() {
        Map<String, double[]> m = LikelihoodFree.iidTrials(1500, new int[] {1, 10}, 2);
        System.out.printf("i.i.d. trials, Gaussian linear, n=1500 (13.5.1/13.5.3) C2ST: %s%n",
                NeuralLikelihoodTest.line(m));
        assertTrue(m.get("NLE k=10")[0] < 0.75, "one NLE serves ten i.i.d. trials");
        assertTrue(m.get("NPSE composed k=1")[0] < 0.75, "the score posterior fits one trial");
    }
}
