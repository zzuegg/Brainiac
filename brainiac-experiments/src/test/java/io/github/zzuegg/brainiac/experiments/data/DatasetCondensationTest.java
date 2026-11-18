package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.Condensed;
import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.Condenser;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DatasetCondensationTest {

    @Test
    void matchingBeatsRandomRealImagesAtOneImagePerClass() {
        Map<Condenser, Condensed> r = new EnumMap<>(Condenser.class);
        for (Condenser m : Condenser.values()) {
            r.put(m, SyntheticTraining.condensation(m, 1, 40, 2, 1));
        }
        System.out.printf("condensation 1 img/class (MLP acc / tanh-MLP transfer): GM %.3f/%.3f DM %.3f/%.3f "
                + "random %.3f/%.3f k-means %.3f/%.3f herding %.3f/%.3f k-center %.3f/%.3f full %.3f/%.3f%n",
                r.get(Condenser.GRADIENT_MATCHING).accuracy(), r.get(Condenser.GRADIENT_MATCHING).transfer(),
                r.get(Condenser.DISTRIBUTION_MATCHING).accuracy(), r.get(Condenser.DISTRIBUTION_MATCHING).transfer(),
                r.get(Condenser.RANDOM).accuracy(), r.get(Condenser.RANDOM).transfer(),
                r.get(Condenser.KMEANS).accuracy(), r.get(Condenser.KMEANS).transfer(),
                r.get(Condenser.HERDING).accuracy(), r.get(Condenser.HERDING).transfer(),
                r.get(Condenser.KCENTER).accuracy(), r.get(Condenser.KCENTER).transfer(),
                r.get(Condenser.FULL).accuracy(), r.get(Condenser.FULL).transfer());
        double random = r.get(Condenser.RANDOM).accuracy();
        assertTrue(r.get(Condenser.GRADIENT_MATCHING).accuracy() > random + 0.1, "gradient matching beats random");
        assertTrue(r.get(Condenser.DISTRIBUTION_MATCHING).accuracy() > random + 0.1, "distribution matching too");
        for (Condenser m : Condenser.values()) {
            assertTrue(m == Condenser.FULL || r.get(Condenser.FULL).accuracy() > r.get(m).accuracy(), m.name());
        }
    }

    @Test
    void herdingAndKCenterPickDistinctRows() {
        double[][] x = {{0, 0}, {1, 0}, {0, 1}, {10, 10}, {0.4, 0.4}};
        int[] h = SyntheticTraining.herding(x, 2);
        int[] k = SyntheticTraining.kCenter(x, 2);
        assertTrue(h[0] != h[1] && k[0] != k[1]);
        assertTrue(k[1] == 3, "farthest-first reaches the outlier");
    }
}
