package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RelationalGraphsLinkTest {

    static String line(String title, Map<String, double[]> r) {
        StringBuilder s = new StringBuilder(title).append(':');
        r.forEach((k, v) -> {
            s.append(' ').append(k);
            for (double x : v) {
                s.append(String.format(" %.3f", x));
            }
            s.append(';');
        });
        return s.toString();
    }

    @Test
    void relationalEncodersAndNbfnetTransductiveAndInductive() {
        Map<String, double[]> r = RelationalGraphs.linkPrediction(5, 100, 1);
        String line = line("KG link MRR/Hits10 transductive, inductive", r);
        System.out.println(line);
        double[] nbf = r.get("NBFNet");
        assertTrue(nbf[2] > r.get("RGCN_BASIS")[2], line);
        assertTrue(nbf[2] > r.get("common neighbours")[2], line);
        assertTrue(r.get("COMPLEX (shallow)")[0] > 0.3, line);
    }
}
