package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.AnswerDebate;
import java.util.List;
import org.junit.jupiter.api.Test;

class DebatePanelTest {

    @Test
    void confidentAgentsPersuadeUnsureOnes() {
        int[] a = {1};
        int[] b = {2};
        int[] c = {3};
        AnswerDebate.Debater unsureA = (p, x) -> x[0] == 1 ? -0.6 : x[0] == 2 ? -0.7 : -0.8;
        AnswerDebate.Debater unsureC = (p, x) -> x[0] == 3 ? -0.6 : x[0] == 2 ? -0.7 : -0.8;
        AnswerDebate.Debater sureB = (p, x) -> x[0] == 2 ? -0.01 : -5;
        AnswerDebate panel = AnswerDebate.of(List.of(unsureA, unsureC, sureB)).agreement(0.5);
        assertArrayEquals(b, panel.productOfExperts(new int[0], List.of(a, b, c)));
        AnswerDebate.Verdict v = panel.debate(new int[0], List.of(a, b, c), 2);
        assertArrayEquals(new int[] {0, 2, 1}, v.rounds().get(0));
        assertArrayEquals(new int[] {1, 1, 1}, v.rounds().get(1));
        assertArrayEquals(b, v.answer());
        assertEquals(3, v.rounds().size());
    }

    @Test
    void panelsBeatTheirMembers() {
        DebatePanel.Result r = DebatePanel.run(5, 120, 2, 1);
        System.out.println("debate " + r);
        assertTrue(r.vote() > r.singleMean(), r.toString());
        assertTrue(r.product() > r.singleMean(), r.toString());
        assertTrue(r.debate() > r.singleMean(), r.toString());
    }
}
