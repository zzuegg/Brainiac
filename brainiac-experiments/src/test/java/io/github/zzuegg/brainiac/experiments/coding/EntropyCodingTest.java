package io.github.zzuegg.brainiac.experiments.coding;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class EntropyCodingTest {

    @Test
    void modelsCompressAndRoundTrip() {
        List<EntropyCoding.CodeRow> rows = EntropyCoding.lossless(6_000, 1);
        for (EntropyCoding.CodeRow r : rows) {
            System.out.printf("12.2.14 %-20s %-42s %.3f bits/symbol (%.3f with model), log loss %.3f, round trip %s,"
                    + " %8.0f symbols/s%n", r.data(), r.method(), r.bitsPerSymbol(), r.withModel(), r.logLoss(),
                    r.roundTrip(), r.symbolsPerSecond());
        }
        for (EntropyCoding.CodeRow r : rows) {
            assertTrue(r.roundTrip(), r.toString());
            if (!Double.isNaN(r.logLoss())) {
                assertTrue(r.bitsPerSymbol() < r.logLoss() + 0.02, "coder within 0.02 bits of the log loss: " + r);
            }
        }
        double deflate = pick(rows, "synthetic", "Deflate").bitsPerSymbol();
        double order3 = pick(rows, "synthetic", "adaptive order-3").bitsPerSymbol();
        assertTrue(order3 < deflate, "order-3 context model beats Deflate on text");
    }

    private static EntropyCoding.CodeRow pick(List<EntropyCoding.CodeRow> rows, String data, String method) {
        return rows.stream().filter(r -> r.data().startsWith(data) && r.method().startsWith(method)).findFirst()
                .orElseThrow();
    }

    @Test
    void bitsBackApproachesTheElbo() {
        List<EntropyCoding.ChainRow> rows = EntropyCoding.bitsBack(2_000, 300, 1_500, 2);
        for (EntropyCoding.ChainRow r : rows) {
            System.out.printf("12.2.16 %-42s %.2f bits/image, first image %.1f, second half %.2f bits/image,"
                    + " round trip %s%n", r.method(), r.bitsPerImage(), r.firstImage(), r.laterImages(), r.roundTrip());
        }
        for (EntropyCoding.ChainRow r : rows) {
            assertTrue(r.roundTrip(), r.toString());
        }
        double elbo = rows.get(0).bitsPerImage();
        EntropyCoding.ChainRow bb = rows.get(1);
        assertTrue(bb.laterImages() < elbo * 1.1, "BB-ANS approaches the negative ELBO after the first images");
    }
}
