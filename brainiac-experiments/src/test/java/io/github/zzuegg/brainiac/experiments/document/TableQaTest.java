package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TableQaTest {

    @Test
    void parserAndTapasAnswerTableQuestions() {
        Map<TableQa.Method, TableQa.Scores> all = new EnumMap<>(TableQa.Method.class);
        for (TableQa.Method m : TableQa.Method.values()) {
            TableQa.Scores s = TableQa.run(m, 300, 200, 1);
            all.put(m, s);
            System.out.printf("Table QA %-6s small %.3f %s large %.3f %s (%d ms)%n", m, s.smallAccuracy(), s.small(),
                    s.largeAccuracy(), s.large(), s.millis());
        }
        assertTrue(all.get(TableQa.Method.PARSER).smallAccuracy() > 0.9, "the enumerated programs answer exactly");
        assertTrue(all.get(TableQa.Method.TAPAS).smallAccuracy() > 0.3, "TAPAS beats chance");
    }
}
