package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.LongTailShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.LongTailShootout.Result;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class LongTailShootoutTest {

    @Test
    void logitAdjustmentMarginsAndDecouplingAtRatioOneHundred() {
        List<Result> rows = LongTailShootout.run(100, 500, 1);
        Map<Method, Result> by = new EnumMap<>(Method.class);
        for (Result r : rows) {
            by.put(r.method(), r);
            System.out.printf("LongTail ratio 100 %-22s balanced %.3f head %.3f medium %.3f tail %.3f param %.2f%n",
                    r.method(), r.balanced(), r.head(), r.medium(), r.tail(), r.parameter());
        }
        double ce = by.get(Method.CE).balanced();
        assertTrue(by.get(Method.CE).tail() < by.get(Method.CE).head() - 0.2, "CE neglects the tail");
        assertTrue(by.get(Method.POSTHOC_ADDITIVE).balanced() > ce, "post-hoc additive adjustment helps");
        assertTrue(by.get(Method.LA_LOSS).balanced() > ce, "the logit-adjusted loss helps");
        assertTrue(by.get(Method.LA_LOSS).tail() > by.get(Method.CE).tail(), "the logit-adjusted loss lifts the tail");
        for (Result r : rows) {
            assertTrue(r.balanced() > 0.2, "every method trains: " + r);
        }
    }
}
