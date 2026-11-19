package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.experimental.logic.ResolutionProver;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogicInferenceProvingTest {

    private static LogicInference.Row find(List<LogicInference.Row> rows, String task, String method) {
        return rows.stream().filter(r -> r.task().startsWith(task) && r.method().equals(method)).findFirst()
            .orElseThrow();
    }

    @Test
    void soundOnSatisfiableInput() {
        ResolutionProver.Problem p = ResolutionProver.Problem.builder("sat").axiom("~r(X,Y) | ~r(Y,Z) | r(X,Z)")
            .axiom("r(a,b)").axiom("r(b,c)").goal("~r(c,a)").build();
        ResolutionProver.Result r = ResolutionProver.prove(p, ResolutionProver.Selection.AGE,
            ResolutionProver.Limits.of(200), null);
        assertFalse(r.proved(), "r(c,a) does not follow");
        assertTrue(r.saturated(), "the clause set saturates");
    }

    @Test
    void theoremProving() {
        List<LogicInference.Row> rows = LogicInference.proving(300, 1);
        System.out.printf("13.2.12 proving (proved of n, size = mean given clauses): %s%n",
            rows.stream().map(LogicInference.Row::toString).collect(Collectors.joining("; ")));
        double age = find(rows, "held-out", "AGE").value();
        assertTrue(age > 0, "breadth-first proves something");
        assertTrue(find(rows, "held-out", "AGE_WEIGHT").value() >= age, "AGE_WEIGHT beats pure age");
        assertTrue(find(rows, "held-out", "LEARNED").value() >= age, "the learned selector is no worse than age");
    }
}
