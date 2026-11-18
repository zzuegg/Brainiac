package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TabularShootoutTest {

    static String line(String title, Map<String, BinaryBench.Result> r) {
        StringBuilder s = new StringBuilder(title).append(" acc/AUC/ms:");
        r.forEach((k, v) -> s.append(String.format(" %s %.3f/%.3f/%d;", k, v.accuracy(), v.auc(), v.millis())));
        return s.toString();
    }

    @Test
    void tabularModelsAgainstMlpAndTrees() {
        Map<String, BinaryBench.Result> r = TabularShootout.shootout(250, 1);
        String line = line("Tabular", r);
        System.out.println(line);
        assertTrue(r.get("FT-Transformer").auc() > r.get("NAM").auc(), line);
        assertTrue(r.get("TabM (k=8)").auc() > 0.85, line);
        assertTrue(r.get("RealMLP").auc() > 0.85, line);
    }
}
