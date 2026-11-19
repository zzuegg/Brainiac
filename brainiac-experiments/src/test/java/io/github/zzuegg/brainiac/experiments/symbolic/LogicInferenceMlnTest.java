package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LogicInferenceMlnTest {

    private static LogicInference.Row find(List<LogicInference.Row> rows, String task, String method) {
        return rows.stream().filter(r -> r.task().startsWith(task) && r.method().startsWith(method)).findFirst()
            .orElseThrow();
    }

    @Test
    void markovLogic() {
        List<LogicInference.Row> rows = LogicInference.markovLogic(60, 0.5, 300, 1);
        System.out.printf("13.2.13 MLN (AUROC, ref = CLL; learning value = rule signs right): %s%n",
            rows.stream().map(LogicInference.Row::toString).collect(Collectors.joining("; ")));
        assertTrue(find(rows, "smokers n=5", "MC_SAT").value() < 0.1, "MC-SAT matches exact marginals");
        assertTrue(find(rows, "smokers n=60", "MC_SAT").value() > 0.6, "MC-SAT ranks hidden atoms");
        assertTrue(find(rows, "smokers learn", "PSEUDO_LIKELIHOOD").value() == 1, "PL recovers the rule signs");
    }
}
