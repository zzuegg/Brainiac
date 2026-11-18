package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TableQaTest {

    @Test
    void parserAndTapasAnswerTableQuestions() {
        TableQa.Scores parser = TableQa.run(TableQa.Method.PARSER, 300, 200, 1);
        TableQa.Scores tapas = TableQa.run(TableQa.Method.TAPAS, 300, 200, 1);
        for (TableQa.Scores s : new TableQa.Scores[] {parser, tapas}) {
            System.out.printf("Table QA %-6s small %.3f %s large %.3f %s (%d ms)%n", s.method(), s.smallAccuracy(),
                    s.small(), s.largeAccuracy(), s.large(), s.millis());
        }
        assertTrue(parser.smallAccuracy() > 0.9, "the enumerated programs answer exactly");
        // chance over the answer vocabulary is far below this; TAPAS stays weak on lookups and sums at this budget
        assertTrue(tapas.smallAccuracy() > 0.2, "TAPAS beats chance");
    }
}
