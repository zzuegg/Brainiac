package io.github.zzuegg.brainiac.experiments.synthesis;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.synthesis.ProgramSynthesisShootout.Row;
import io.github.zzuegg.brainiac.experiments.synthesis.ProgramSynthesisShootout.Task;

import java.util.List;
import java.util.StringJoiner;

import org.junit.jupiter.api.Test;

/** Item 7.6.7 (DeepCoder) against enumeration, library learning and evolution. */
class ProgramSynthesisShootoutTest {

    static final List<Task> TASKS = ProgramSynthesisShootout.tasks(30, 2, 5, 11);

    static String line(List<Row> rows) {
        StringJoiner j = new StringJoiner("; ");
        rows.forEach(r -> j.add(r.toString()));
        return j.toString();
    }

    @Test
    void deepCoderGuidanceCutsTheSearch() {
        List<Row> rows = ProgramSynthesisShootout.searchers(TASKS, 20_000, 600, 1);
        System.out.println("Synthesis, 30 tasks of 2-5 primitives, budget 20k: " + line(rows));
        Row enumerate = rows.get(0);
        Row ordered = rows.get(2);
        assertTrue(ordered.solved() >= enumerate.solved(), "ordering keeps the solve rate (the search is complete)");
        assertTrue(ordered.meanTried() < enumerate.meanTried(), "guidance tries fewer programs");
    }
}
