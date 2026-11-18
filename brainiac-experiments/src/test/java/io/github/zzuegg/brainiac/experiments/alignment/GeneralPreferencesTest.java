package io.github.zzuegg.brainiac.experiments.alignment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.3 and 9.2.4: general preference models and self-play preference optimization. */
class GeneralPreferencesTest {

    @Test
    void preferenceModelsOnCyclicPreferences() {
        Map<String, GeneralPreferences.ModelRow> r = GeneralPreferences.models(true, 800, 150, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s held-out %.3f cyclic %.3f;", k, row.heldOut(),
                row.cyclicTriples())));
        System.out.printf("9.2.3 preference accuracy and share of cyclic triples captured (BT is the baseline):%s%n",
                sb);
        assertTrue(r.get("Bradley-Terry").cyclicTriples() <= 2.0 / 3 + 1e-9,
                "a transitive model gets at most two of the three pairs of a cycle right");
    }

    @Test
    void truePreferenceIsSkewSymmetric() {
        int[] a = {1, 2};
        int[] b = {3};
        assertEquals(1.0, GeneralPreferences.truePreference(a, b, true) + GeneralPreferences.truePreference(b, a, true),
                1e-9);
    }

    @Test
    void selfPlayPolicies() {
        Map<String, GeneralPreferences.PolicyRow> r = GeneralPreferences.policies(true, 600, 60, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s worst-case %.3f;", k, row.worstCase())));
        System.out.printf("9.2.4 worst-case win rate against a style panel (DPO is the baseline, 0.5 is Nash):%s%n",
                sb);
        assertTrue(r.size() >= 5);
    }
}
