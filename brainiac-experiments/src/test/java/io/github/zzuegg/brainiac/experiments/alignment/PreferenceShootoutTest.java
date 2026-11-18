package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.5 to 9.2.7: link-function DPO variants, length bias and exploration. */
class PreferenceShootoutTest {

    @Test
    void likelihoodDisplacement() {
        Map<PreferenceShootout.Objective, List<PreferenceShootout.Checkpoint>> r = PreferenceShootout.displacement(
                400, 80, 3, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, cps) -> {
            PreferenceShootout.Checkpoint last = cps.get(cps.size() - 1);
            sb.append(String.format(" %s drift %.2f win %.3f gold %.3f kl %.2f;", k, last.chosenDrift(),
                    last.winRate(), last.gold(), last.kl()));
        });
        System.out.printf("9.2.5 chosen log-prob drift and gold after training (DPO is the baseline):%s%n", sb);
        assertEquals(PreferenceShootout.Objective.values().length, r.size());
    }

    @Test
    void lengthControlledWinRateRemovesALengthEffect() {
        double[] diff = new double[400];
        double[] wins = new double[400];
        for (int i = 0; i < diff.length; i++) {
            diff[i] = i % 2 == 0 ? 1 : -1;
            wins[i] = diff[i] > 0 ? (i % 10 < 8 ? 1 : 0) : (i % 10 < 3 ? 1 : 0);
        }
        double lc = PreferenceShootout.lengthControlledWinRate(diff, wins);
        System.out.printf("9.2.6 length-controlled win rate on length-only wins %.3f (raw 0.5)%n", lc);
        assertTrue(lc > 0.3 && lc < 0.7);
    }

    @Test
    void lengthRegularizedDpo() {
        Map<String, PreferenceShootout.LengthRow> r = PreferenceShootout.lengthBias(1.0, 400, 120, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s len %.2f acc %.3f win %.3f LC %.3f;", k, row.length(),
                row.accuracy(), row.winRate(), row.lengthControlled())));
        System.out.printf("9.2.6 length, accuracy, raw and length-controlled win rate (DPO is the baseline):%s%n", sb);
        assertTrue(r.size() >= 2);
    }

    @Test
    void exploratoryPreferenceOptimization() {
        Map<String, PreferenceShootout.ExploreRow> r = PreferenceShootout.exploration(20, 32, 3, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s gold %.3f target p %.4f queries %d;", k,
                row.gold()[row.gold().length - 1], row.target(), row.queries())));
        System.out.printf("9.2.7 final gold and probability of the rare best response (online DPO is the baseline):%s%n",
                sb);
        assertTrue(r.size() >= 4);
    }
}
