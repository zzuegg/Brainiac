package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TabularPretrainingTest {

    @Test
    void pretrainingUnderLabelScarcity() {
        Map<String, double[]> r = TabularPretraining.labelScarcity(1000, 300, new double[] {0.05, 0.2}, 2, 120, 100, 1);
        r.forEach((k, v) -> System.out.printf("tabular SSL %s: acc %.3f (seed spread %.3f)%n", k, v[0], v[1]));
        double bestSsl = 0;
        for (String o : new String[] {"VIME", "SCARF", "SUBTAB", "MASK_RECOVER"}) {
            bestSsl = Math.max(bestSsl, Math.max(r.get(o + " fine-tune @ 20%")[0], r.get(o + " probe @ 20%")[0]));
        }
        // a genuine negative on this trees-favouring suite: SSL encoders learn but stay near the scratch MLP
        assertTrue(bestSsl > 0.55, "pretrained encoders learn at 20% labels: " + bestSsl);
        assertTrue(r.get("GBT @ 20%")[0] > r.get("scratch MLP @ 20%")[0], "trees beat the MLP at 20%");
        Map<String, double[]> m = TabularPretraining.missingness(800, 300, 0.3, 120, 100, 2);
        m.forEach((k, v) -> System.out.printf("tabular SSL missingness %s: clean %.3f MCAR %.3f MNAR %.3f%n", k, v[0],
                v[1], v[2]));
        assertTrue(m.get("random forest")[0] > 0.55, "RF clean");
    }

    @Test
    void structuredCorruptionOnRedundantBlocks() {
        Map<String, double[]> r = TabularPretraining.structured(1000, 300, 0.2, 150, 3);
        r.forEach((k, v) -> System.out.printf("structured corruption %s: probe acc %.3f, pretext loss %.3f%n", k, v[0],
                v[1]));
        assertTrue(r.get("duplicates grouped with their source")[0] >= 0.75, "MI grouping finds the duplicates");
        assertTrue(r.get("group marginal")[1] > r.get("random")[1], "group corruption leaves a harder pretext task");
    }
}
