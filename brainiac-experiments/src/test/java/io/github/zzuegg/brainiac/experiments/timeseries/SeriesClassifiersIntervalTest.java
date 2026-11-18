package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SeriesClassifiersIntervalTest {

    @Test
    void quantOnIntervalTask() {
        Map<String, double[]> r = SeriesClassifiers.interval(SeriesClassifiers.split("interval", 120, 150, 128, 3), 3);
        r.forEach((k, v) -> System.out.printf("interval task %s: acc %.3f (%.0f ms)%n", k, v[0], v[1]));
        assertTrue(r.get("QUANT")[0] > 0.7, "QUANT " + r.get("QUANT")[0]);
        assertTrue(r.get("QUANT")[0] >= r.get("TSF")[0] - 0.05, "QUANT vs TSF");
    }

    @Test
    void ensemblesAndTheOracleGap() {
        SeriesClassifiers.EnsembleResult r = SeriesClassifiers.ensembles(
                SeriesClassifiers.split("dictionary", 90, 120, 128, 4), 3, 4);
        r.members().forEach((k, v) -> System.out.printf("ensemble dictionary member %s: acc %.3f%n", k, v));
        r.strategies().forEach((k, v) -> System.out.printf(
                "ensemble dictionary %s: acc %.3f (best member %.3f, oracle %.3f, gap closed %.2f)%n", k, v,
                r.best(), r.oracle(), r.gapClosed(k)));
        assertTrue(r.oracle() >= r.best(), "oracle bounds the best member");
        assertTrue(r.strategies().get("CAWPE") >= r.best(), "CAWPE " + r.strategies().get("CAWPE"));
        assertTrue(r.strategies().get("stack (extra trees)") > r.best(), "stacking gains over the best member");
    }
}
