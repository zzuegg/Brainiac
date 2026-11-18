package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogicInferenceAspTest {

    private static LogicInference.Row find(List<LogicInference.Row> rows, String task, String method) {
        return rows.stream().filter(r -> r.task().startsWith(task) && r.method().equals(method)).findFirst()
            .orElseThrow();
    }

    @Test
    void answerSets() {
        List<LogicInference.Row> rows = LogicInference.answerSets(5, 1);
        System.out.printf("13.2.14 ASP (value vs ref, size = ground rules / clauses / nodes): %s%n",
            rows.stream().map(LogicInference.Row::toString).collect(Collectors.joining("; ")));
        LogicInference.Row loops = find(rows, "hamilton n=10 enumerated", "ASP + loop formulas");
        assertEquals(loops.reference(), loops.value(), "loop formulas leave only Hamiltonian cycles");
        LogicInference.Row plain = find(rows, "hamilton n=10 enumerated", "plain completion");
        assertTrue(plain.value() < plain.reference(), "plain completion admits sub-cycles");
        assertEquals(92, find(rows, "queens-8 all models", "ASP").value(), "all 8-queens solutions");
        LogicInference.Row col = find(rows, "colouring n=12 all models", "ASP");
        assertEquals(col.reference(), col.value(), "ASP and the direct CNF have the same colourings");
        LogicInference.Row blocks = find(rows, "blocks-5", "ASP");
        assertEquals(blocks.reference(), blocks.value(), "shortest ASP plan matches sequential SATPlan");
    }
}
