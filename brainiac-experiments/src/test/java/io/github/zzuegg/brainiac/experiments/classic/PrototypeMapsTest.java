package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class PrototypeMapsTest {

    @Test
    void mapsPreserveTopologyAtNearKMeansQuantisationError() {
        Map<String, double[]> r = PrototypeMaps.run(1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s QE %.3f TE %.3f; ", k, v[0], v[1])));
        System.out.printf("36 prototypes: %s%n", sb);
        for (String data : new String[] {"rings", "S-curve"}) {
            double km = r.get(data + " / k-means++")[0];
            assertTrue(r.get(data + " / SOM")[0] < km * 1.6, sb.toString());
            assertTrue(r.get(data + " / growing neural gas")[0] < km * 1.3, sb.toString());
            assertTrue(r.get(data + " / growing neural gas")[1] < 0.2, sb.toString());
        }
        assertTrue(r.get("S-curve / SOM")[1] < 0.3, sb.toString());
    }
}
