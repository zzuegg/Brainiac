package io.github.zzuegg.brainiac.experiments.algorithms;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experiments.algorithms.CachingWithPredictions.CacheRow;
import io.github.zzuegg.brainiac.experiments.algorithms.CachingWithPredictions.SkiRow;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 11.2.17: predictive marking, Parrot-style imitation and ski rental with predictions. */
class CachingWithPredictionsTest {

    static CacheRow find(List<CacheRow> rows, String method) {
        return rows.stream().filter(r -> r.method().equals(method)).findFirst().orElseThrow();
    }

    static SkiRow ski(List<SkiRow> rows, String rule, double noise, double lambda) {
        return rows.stream().filter(r -> r.rule().equals(rule) && r.noise() == noise
                && (Double.isNaN(lambda) ? Double.isNaN(r.lambda()) : r.lambda() == lambda)).findFirst().orElseThrow();
    }

    @Test
    void pagingAndSkiRental() {
        double[] noises = {0, 1, 3};
        StringBuilder line = new StringBuilder("11.2.17 of Belady:");
        for (KeyStreams.TraceKind kind : KeyStreams.TraceKind.values()) {
            List<CacheRow> rows = CachingWithPredictions.paging(kind, 12_000, 1000, 120, 64, noises, 1);
            System.out.println(kind + ": " + rows);
            line.append(String.format(" %s LRU %.2f, PM(0) %.2f, PM(3) %.2f, FTP(3) %.2f, Parrot %.2f;", kind,
                    find(rows, "LRU").ofBelady(), find(rows, "predictive marking, noise 0.0").ofBelady(),
                    find(rows, "predictive marking, noise 3.0").ofBelady(),
                    find(rows, "follow predictions, noise 3.0").ofBelady(), find(rows, "Parrot imitation").ofBelady()));
        }
        List<SkiRow> skiRows = CachingWithPredictions.skiRental(100, 2000, new double[] {0, 0.5, 2},
                new double[] {0.2, 0.5, 1}, 2);
        skiRows.forEach(System.out::println);
        line.append(String.format(" ski mean (max) ratio: break-even %.3f, Karlin %.3f, det lambda .5 noise 0 %.3f (%.2f)"
                + " noise 2 %.3f (%.2f)", ski(skiRows, "break-even", 0, Double.NaN).meanRatio(),
                ski(skiRows, "Karlin randomised", 0, Double.NaN).meanRatio(),
                ski(skiRows, "deterministic", 0, 0.5).meanRatio(), ski(skiRows, "deterministic", 0, 0.5).maxRatio(),
                ski(skiRows, "deterministic", 2, 0.5).meanRatio(), ski(skiRows, "deterministic", 2, 0.5).maxRatio()));
        System.out.println(line);
        assertTrue(ski(skiRows, "deterministic", 2, 0.5).maxRatio() <= 1 + 1 / 0.5 + 1e-9,
                "the lambda-robust rule keeps its (1 + 1/lambda) bound");
    }
}
