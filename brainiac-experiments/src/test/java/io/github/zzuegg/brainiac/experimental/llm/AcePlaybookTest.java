package io.github.zzuegg.brainiac.experimental.llm;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AcePlaybookTest {

    static final Map<String, String> CAPITALS = Map.of("france", "paris", "spain", "madrid", "peru", "lima",
            "kenya", "nairobi", "japan", "tokyo", "chile", "santiago");
    static final List<String> TASKS = List.of("france", "spain", "peru", "spain", "kenya", "japan", "chile", "france");

    static int words(String s) {
        return s.isBlank() ? 0 : s.trim().split("\\s+").length;
    }

    static float[] embed(String s) {
        return VirtualContextPagerTest.embed(s);
    }

    /** Answers from the first (best-ranked) playbook line about the country. */
    static AcePlaybook.Attempt generate(String country, String playbook) {
        for (String line : playbook.lines().toList()) {
            String[] p = line.split(" ");
            if (p.length == 4 && p[1].equals(country) && p[2].equals("->")) {
                return new AcePlaybook.Attempt(p[3], List.of(p[0].substring(1, p[0].length() - 1)));
            }
        }
        return new AcePlaybook.Attempt("unknown", List.of());
    }

    /** Tags what was used; on a mistake, writes down the right answer. */
    static AcePlaybook.Reflection reflect(String country, AcePlaybook.Attempt a, String feedback) {
        boolean right = feedback.equals("correct");
        List<AcePlaybook.Tag> tags = a.used().stream().map(id -> new AcePlaybook.Tag(id, right)).toList();
        List<String> insights = right ? List.of()
                : List.of("capitals: " + country + " -> " + feedback.substring(feedback.lastIndexOf(' ') + 1));
        return new AcePlaybook.Reflection(insights, tags);
    }

    static double epoch(AcePlaybook book) {
        int right = 0;
        for (String c : TASKS) {
            AcePlaybook.Attempt a = book.step(c, ans -> ans.equals(CAPITALS.get(c)) ? "correct"
                    : "wrong, expected " + CAPITALS.get(c), AcePlaybookTest::generate, AcePlaybookTest::reflect,
                    AcePlaybook.Curator.standard(), 200, AcePlaybookTest::words);
            right += a.answer().equals(CAPITALS.get(c)) ? 1 : 0;
        }
        return right / (double) TASKS.size();
    }

    @Test
    void thePlaybookImprovesTheAgentWithoutTraining() {
        AcePlaybook book = AcePlaybook.create(AcePlaybookTest::embed).refineWith(0.95, 1);
        book.apply(List.of(new AcePlaybook.Add("capitals", "france -> lyon")));
        double first = epoch(book);
        double second = epoch(book);
        assertEquals(2 / 8.0, first, 1e-9, "only repeats are right the first time");
        assertEquals(1.0, second, 1e-9);
        assertEquals(6, book.bullets().size(), book.bullets().toString());
        assertFalse(book.render(200, AcePlaybookTest::words).contains("lyon"), "the harmful bullet is pruned");
        assertTrue(book.bullets().stream().allMatch(b -> b.helpful() >= 1));
    }

    @Test
    void refineMergesDuplicatesAndRenderKeepsTheBestWithinBudget() {
        AcePlaybook book = AcePlaybook.create(AcePlaybookTest::embed).refineWith(0.95, 3);
        book.apply(List.of(new AcePlaybook.Add("tips", "Check units first"),
                new AcePlaybook.Add("tips", "check UNITS first"),
                new AcePlaybook.Add("mistakes", "Never divide by the sample size twice"),
                new AcePlaybook.Tag("b1", true), new AcePlaybook.Tag("b1", true), new AcePlaybook.Tag("b2", true),
                new AcePlaybook.Tag("missing", true)));
        assertEquals(1, book.refine());
        AcePlaybook.Bullet merged = book.bullets().getFirst();
        assertEquals("b0", merged.id());
        assertEquals(2, merged.helpful());
        String small = book.render(5, AcePlaybookTest::words);
        assertEquals("## tips\n[b0] Check units first\n", small);
        assertTrue(book.render(100, AcePlaybookTest::words).contains("## mistakes\n[b2]"));
        book.apply(List.of(new AcePlaybook.Remove("b2")));
        assertEquals(1, book.bullets().size());
    }
}
