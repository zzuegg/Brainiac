package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.rl.MatrixGame;
import io.github.zzuegg.brainiac.experiments.multiagent.MatrixGameDynamics.ContinuousGame;
import io.github.zzuegg.brainiac.experiments.multiagent.MatrixGameDynamics.Pairing;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MatrixGameDynamicsTest {

    private static String format(Map<String, Double> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %.3f; ", k, v)));
        return sb.toString();
    }

    @Test
    void lookAheadMethodsConvergeWhereGradientDescentAscentCycles() {
        Map<String, Double> bilinear = MatrixGameDynamics.minMax(ContinuousGame.BILINEAR, 500, 0.1f);
        Map<String, Double> dirac = MatrixGameDynamics.minMax(ContinuousGame.DIRAC_GAN, 500, 0.1f);
        System.out.printf("bilinear, 500 steps, distance from equilibrium (start 2.83): %s%n", format(bilinear));
        System.out.printf("Dirac-GAN, 500 steps (start 1.41): %s%n", format(dirac));
        for (Map<String, Double> m : java.util.List.of(bilinear, dirac)) {
            double gda = m.get("simultaneous GDA");
            assertTrue(m.get("extragradient") < 0.5 * gda, "extragradient converges");
            assertTrue(m.get("optimistic GDA") < 0.5 * gda, "optimistic GDA converges");
            assertTrue(m.get("consensus") < 0.5 * gda, "consensus converges");
        }
    }

    @Test
    void magnetConvergesInTheLastIterate() {
        Map<String, Double> pennies = MatrixGameDynamics.lastIterate(MatrixGame.matchingPennies(), 1000);
        Map<String, Double> rps = MatrixGameDynamics.lastIterate(MatrixGame.biasedRockPaperScissors(), 1000);
        System.out.printf("last-iterate exploitability, matching pennies: %s%n", format(pennies));
        System.out.printf("last-iterate exploitability, biased RPS: %s%n", format(rps));
        for (Map<String, Double> m : java.util.List.of(pennies, rps)) {
            assertTrue(m.get("R-NaD") < 0.01 && m.get("multiplicative weights") > 0.3, "magnet stops cycling");
        }
    }

    @Test
    void opponentShapingFindsReciprocity() {
        Map<String, Pairing> r = MatrixGameDynamics.shaping(100, 3, 40);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.2f/%.2f (P(C|CC) %.2f, P(C|CD) %.2f); ", k, v.first(),
                v.second(), v.cooperateAfterCooperation(), v.cooperateAfterDefection())));
        System.out.printf("IPD reward per step (CC -1, DD -2), 3 seeds: %s%n", sb);
        assertTrue(r.get("LOLA vs LOLA").first() > r.get("naive vs naive").first() + 0.5, "LOLA cooperates");
        assertTrue(r.get("M-FOS vs naive").first() > r.get("naive vs naive").first() + 0.3, "M-FOS shapes");
    }
}
