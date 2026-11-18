package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DomainGeneralizationTest {

    @Test
    void leaveOneDomainOut() {
        Map<DomainGeneralization.Method, Double> acc = new EnumMap<>(DomainGeneralization.Method.class);
        for (DomainGeneralization.Method m : new DomainGeneralization.Method[] {DomainGeneralization.Method.ERM,
                DomainGeneralization.Method.MIXSTYLE, DomainGeneralization.Method.VREX}) {
            acc.put(m, DomainGeneralization.leaveOneOut(m, 1, 180, 1));
        }
        System.out.printf("DG held-out TEXTURED accuracy %s%n", acc);
        acc.forEach((m, a) -> assertTrue(a > 0.24, m + " is above chance (1/6): " + a));
    }
}
