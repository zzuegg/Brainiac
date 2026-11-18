package io.github.zzuegg.brainiac.experiments.markets;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.markets.LearnedAuctions;

import org.junit.jupiter.api.Test;

import java.util.Map;

class MechanismDesignTest {

    @Test
    void learnedAuctions() {
        LearnedAuctions.Kind[] net = {LearnedAuctions.Kind.REGRET_NET};
        Map<String, double[]> one = MechanismDesign.auction(1, 2, net, 200, 21);
        Map<String, double[]> two = MechanismDesign.auction(2, 2, LearnedAuctions.Kind.values(), 100, 23);
        System.out.printf("12.4.17 {revenue, mean regret, max regret, params} 1x2: %s 2x2: %s%n", Lines.of(one),
                Lines.of(two));
        assertTrue(one.get("Manelli-Vincent")[0] > 0.54 && one.get("Manelli-Vincent")[0] < 0.56,
                Lines.of(one));
        assertTrue(one.get("REGRET_NET")[0] > 0.4 && one.get("REGRET_NET")[2] < 0.03, Lines.of(one));
        assertTrue(two.get("REGRET_FORMER")[0] > 0.4, Lines.of(two));
    }
}
