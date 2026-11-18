package io.github.zzuegg.brainiac.experiments.synthesis;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.synthesis.ProgramSynthesisShootout.Row;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 7.6.8: RobustFill-style program generation against neural program induction (and, in the DeepCoder test, search). */
class RobustFillTest {

    @Test
    void robustFillWritesProgramsThatGeneralise() {
        List<Row> rows = ProgramSynthesisShootout.robustFill(ProgramSynthesisShootoutTest.TASKS, 80, new int[] {1, 8}, 1);
        Row induction = ProgramSynthesisShootout.induction(ProgramSynthesisShootoutTest.TASKS, 60, 1);
        System.out.println("RobustFill: " + ProgramSynthesisShootoutTest.line(rows) + "; " + induction);
        assertTrue(rows.get(1).solved() >= rows.get(0).solved(), "a wider beam solves at least as many");
        assertTrue(rows.get(1).solved() > induction.solved(), "programs generalise better than induction");
    }
}
