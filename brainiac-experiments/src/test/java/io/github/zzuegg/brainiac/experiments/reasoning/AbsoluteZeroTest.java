package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import org.junit.jupiter.api.Test;

class AbsoluteZeroTest {

    @Test
    void executorRunsTheDsl() {
        assertEquals(4, AbsoluteZero.execute("+3*2", 4));
        assertEquals(7, AbsoluteZero.execute("-3", 0));
        assertEquals(-1, AbsoluteZero.execute("+", 0));
        CharacterTokenizer tok = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
        assertNull(AbsoluteZero.parseProposal(tok.encode("+3*>4#")));
        assertEquals("+3*2", AbsoluteZero.parseProposal(tok.encode("+3*2>4#"))[0]);
    }

    @Test
    void selfPlayWithoutDataImprovesProposerAndSolver() {
        AbsoluteZero.Result r = AbsoluteZero.run(300, 1);
        System.out.println("Absolute Zero " + r);
        assertTrue(r.validAfter() > r.validBefore() + 0.3, r.toString());
        assertTrue(r.selfSolveAfter() > r.selfSolveBefore() + 0.2, r.toString());
        assertTrue(r.distinctTasks() > 3, r.toString());
    }
}
