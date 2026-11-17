package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class HyperparameterSearchTest {

    /** Best at lr 1e-3, x 0.3 and the "adam" choice; more budget lowers the score a little. */
    private static double objective(Map<String, Object> c, int budget) {
        double lr = Math.log10((double) c.get("lr")) + 3;
        double x = (double) c.get("x") - 0.3;
        double layers = ((int) c.get("layers") - 4) / 4.0;
        double opt = c.get("opt").equals("adam") ? 0 : 0.5;
        return lr * lr + x * x + layers * layers + opt + 1.0 / budget;
    }

    private static Map<String, HyperparameterSearch.Range> space() {
        Map<String, HyperparameterSearch.Range> s = new LinkedHashMap<>();
        s.put("lr", new HyperparameterSearch.LogUniform(1e-5, 1e-1));
        s.put("x", new HyperparameterSearch.Uniform(0, 1));
        s.put("layers", new HyperparameterSearch.IntRange(1, 8));
        s.put("opt", new HyperparameterSearch.Choice(List.of("sgd", "adam")));
        return s;
    }

    @Test
    void samplesStayInTheirRanges() {
        HyperparameterSearch h = new HyperparameterSearch(space(), 1);
        for (int i = 0; i < 200; i++) {
            Map<String, Object> c = h.sample();
            double lr = (double) c.get("lr");
            assertTrue(lr >= 1e-5 && lr <= 1e-1);
            int layers = (int) c.get("layers");
            assertTrue(layers >= 1 && layers <= 8);
        }
    }

    @Test
    void hyperbandRunsItsBracketsAndKeepsTheBest() {
        HyperparameterSearch.Result r = new HyperparameterSearch(space(), 2)
                .hyperband(HyperparameterSearchTest::objective, 27, 3);
        assertEquals(40 + 17 + 8 + 4, r.trials().size());
        assertEquals(27, r.best().budget());
        long full = r.trials().stream().filter(t -> t.budget() == 27).count();
        assertEquals(1 + 1 + 2 + 4, full);
        assertTrue(r.best().score() < 1.0, r.best().toString());
    }

    @Test
    void tpeFindsBetterConfigurationsThanItsRandomStart() {
        double tpe = 0;
        double random = 0;
        for (long seed = 0; seed < 5; seed++) {
            tpe += new HyperparameterSearch(space(), seed).tpe(HyperparameterSearchTest::objective, 60, 100, 15, 24)
                    .best().score();
            random += new HyperparameterSearch(space(), seed).random(HyperparameterSearchTest::objective, 60, 100)
                    .best().score();
        }
        assertTrue(tpe < random, "tpe " + tpe / 5 + ", random " + random / 5);
        assertTrue(tpe / 5 < 0.15, "tpe " + tpe / 5);
    }
}
