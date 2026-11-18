package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PrototypeLearnersTest {

    @Test
    void adaptiveResonanceClassIncremental() {
        for (String name : new String[] {"digits", "spirals"}) {
            PrototypeLearners.Data d = PrototypeLearners.toy(name, 1);
            Map<String, double[]> r = PrototypeLearners.classIncremental(d, name.equals("digits") ? 2 : 1, 200, 1);
            r.forEach((k, v) -> System.out.printf("ART class-incremental %s %s: acc %.3f forgetting %.3f memory %.0f%n",
                    name, k, v[0], v[1], v[2]));
            assertTrue(r.get("Fuzzy ARTMAP (rho 0.6)")[1] < r.get("MLP, sequential")[1], "ARTMAP forgets less");
            assertTrue(r.get("Fuzzy ARTMAP (rho 0.6)")[0] > r.get("MLP, sequential")[0], "ARTMAP beats sequential MLP");
        }
        Map<String, double[]> s = PrototypeLearners.stream(3000, 2);
        s.forEach((k, v) -> System.out.printf(
                "ART drifting stream %s: before drift %.3f last quarter %.3f overall %.3f memory %.0f%n", k, v[0], v[1],
                v[2], v[3]));
    }

    @Test
    void learningVectorQuantization() {
        for (String name : new String[] {"irrelevant", "stress", "digits", "correlated"}) {
            PrototypeLearners.Data d = PrototypeLearners.lvqTask(name, 3);
            Map<String, double[]> r = PrototypeLearners.lvq(d, 1, 30, 3);
            r.forEach((k, v) -> System.out.printf("LVQ %s %s: acc %.3f prototypes %.0f relevance corr %.2f%n", name, k,
                    v[0], v[1], v[2]));
            if (name.equals("irrelevant")) {
                assertTrue(r.get("GMLVQ")[0] > r.get("GLVQ")[0], "GMLVQ beats Euclidean GLVQ with irrelevant features");
                assertTrue(r.get("GMLVQ")[2] > 0.8, "GMLVQ relevance recovers the planted features");
            }
            if (name.equals("correlated")) {
                assertTrue(r.get("LOCALISED_GMLVQ")[0] > r.get("GMLVQ")[0] + 0.1, "localised metrics on the X task");
            }
        }
    }
}
