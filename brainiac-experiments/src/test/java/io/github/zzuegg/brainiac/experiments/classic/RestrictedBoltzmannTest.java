package io.github.zzuegg.brainiac.experiments.classic;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RestrictedBoltzmannTest {

    private static <R> R row(List<R> rows, java.util.function.Predicate<R> p) {
        return rows.stream().filter(p).findFirst().orElseThrow();
    }

    @Test
    void trainersAgainstMixtureAndIndependentPixels() {
        List<ClassicLearners.RbmRow> rows = ClassicLearners.rbmDigits(1200, 1);
        StringBuilder b = new StringBuilder("11.4.12 RBM digits test LL / AIS error / probe:");
        rows.forEach(r -> b.append(String.format(" %s %.2f/%.2f/%.3f;", r.method(), r.testLogLikelihood(),
                r.aisError(), r.probe())));
        System.out.println(b);
        ClassicLearners.RbmRow independent = row(rows, r -> r.method().equals("independent"));
        for (String m : new String[] {"CD-1", "PCD", "PT"}) {
            ClassicLearners.RbmRow r = row(rows, x -> x.method().equals(m));
            assertTrue(r.testLogLikelihood() > independent.testLogLikelihood(), m + " beats independent pixels");
            assertTrue(Math.abs(r.aisError()) < 2, m + ": AIS log Z within 2 nats of the exact value");
        }
    }
}
