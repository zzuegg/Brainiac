package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BoostingShootoutLightGbmTest {

    static String line(String title, Map<String, double[]> r, String format) {
        StringBuilder s = new StringBuilder(title).append(':');
        r.forEach((k, v) -> {
            s.append(' ').append(k);
            for (double x : v) {
                s.append(String.format(format, x));
            }
            s.append(';');
        });
        return s.toString();
    }

    @Test
    void gossAndLeafWiseMatchFullDataAndEfbShrinksTheHistogram() {
        Map<String, double[]> r = BoostingShootout.lightGbm(2000, 60, 1);
        String line = line("LightGBM acc/AUC/logloss/ms/leaves-or-bundles", r, " %.3f");
        System.out.println(line);
        double full = r.get("leaf-wise (32 leaves)")[1];
        assertTrue(full > 0.8, line);
        assertTrue(r.get("leaf-wise + GOSS (a=0.2, b=0.1)")[1] > full - 0.03, line);
        assertTrue(r.get("level-wise (depth 5)")[1] > 0.8, line);
        assertTrue(r.get("sparse EFB (exclusive bundles)")[4] < r.get("sparse no bundling")[4] / 4, line);
        assertTrue(r.get("sparse EFB (exclusive bundles)")[1] > r.get("sparse no bundling")[1] - 0.02, line);
    }
}
