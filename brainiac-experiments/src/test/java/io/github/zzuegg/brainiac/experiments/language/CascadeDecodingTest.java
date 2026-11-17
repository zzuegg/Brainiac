package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class CascadeDecodingTest {

    private static CascadeDecoding.Point find(List<CascadeDecoding.Point> r, String method, double alpha) {
        return r.stream().filter(p -> p.method().equals(method) && p.alpha() == alpha).findFirst().orElseThrow();
    }

    @Test
    void speculativeCascadesTradeQualityForLatencyBetterThanCascades() {
        List<CascadeDecoding.Point> r = CascadeDecoding.run(32, 8, new double[] {0, 0.1, 0.3, 0.5}, 1);
        StringBuilder line = new StringBuilder("Speculative cascades, accuracy/cost/large passes:");
        for (CascadeDecoding.Point p : r) {
            line.append(String.format(" %s a=%.1f %.3f/%.2f/%.2f;", p.method(), p.alpha(), p.accuracy(), p.cost(),
                    p.largePasses()));
        }
        System.out.println(line);
        CascadeDecoding.Point spec = find(r, "speculative cascade (token)", 0);
        CascadeDecoding.Point large = find(r, "large alone", -1);
        assertTrue(Math.abs(spec.accuracy() - large.accuracy()) < 1e-9, "alpha 0 is lossless speculative decoding");
        CascadeDecoding.Point loose = find(r, "speculative cascade (token)", 0.3);
        assertTrue(loose.largePasses() < 0.95 * spec.largePasses(), line.toString());
        assertTrue(loose.accuracy() >= spec.accuracy() - 0.02, "valid small-model tokens stand: " + line);
        // at matched tolerance the speculative cascade is cheaper than the sequential cascade at no loss
        CascadeDecoding.Point casc = find(r, "confidence cascade", 0.3);
        assertTrue(loose.accuracy() >= casc.accuracy() - 0.01 && loose.cost() < casc.cost(), line.toString());
        assertTrue(spec.cost() < large.cost(), line.toString());
    }
}
