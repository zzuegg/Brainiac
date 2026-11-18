package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TableQaLmTest {

    @Test
    void flatLanguageModelBaselineRuns() {
        TableQa.Scores s = TableQa.run(TableQa.Method.LM, 300, 200, 1);
        System.out.printf("Table QA %-6s small %.3f %s large %.3f %s (%d ms)%n", s.method(), s.smallAccuracy(),
                s.small(), s.largeAccuracy(), s.large(), s.millis());
        assertTrue(s.smallAccuracy() >= 0 && s.smallAccuracy() <= 1, "the baseline is scored");
    }
}
