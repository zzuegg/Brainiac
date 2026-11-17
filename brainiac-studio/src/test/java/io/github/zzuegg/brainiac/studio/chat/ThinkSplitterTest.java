package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.Test;

/** A model's plain-text think block goes to the thinking, however the stream cuts it. */
class ThinkSplitterTest {

    private static String[] split(boolean open, List<String> pieces) throws IOException {
        StringBuilder text = new StringBuilder();
        StringBuilder thought = new StringBuilder();
        ThinkSplitter s = new ThinkSplitter(new ChatListener() {
            @Override
            public void text(String t) {
                text.append(t);
            }

            @Override
            public void thinking(String t) {
                thought.append(t);
            }
        }, open);
        for (String p : pieces) {
            s.text(p);
        }
        s.finish();
        return new String[] {thought.toString(), text.toString()};
    }

    @Test
    void aThinkBlockIsThinking() throws IOException {
        String[] r = split(false, List.of("<think>\nThe user greets me.\n</think>\n\nHello! How can I help?"));
        assertEquals("The user greets me.\n", r[0]);
        assertEquals("Hello! How can I help?", r[1]);
    }

    @Test
    void tagsCutAcrossPiecesAreFound() throws IOException {
        String all = "<think>a < b, so</think>\n\nIt is **a** <i>x</i>.";
        for (int size = 1; size <= 9; size++) {
            List<String> pieces = new java.util.ArrayList<>();
            for (int i = 0; i < all.length(); i += size) {
                pieces.add(all.substring(i, Math.min(all.length(), i + size)));
            }
            String[] r = split(false, pieces);
            assertEquals("a < b, so", r[0], "pieces of " + size);
            assertEquals("It is **a** <i>x</i>.", r[1], "pieces of " + size);
        }
    }

    @Test
    void aTemplateThatOpensTheBlockStartsInIt() throws IOException {
        String[] r = split(true, List.of("Let me think.", "</thi", "nk>", "Done."));
        assertEquals("Let me think.", r[0]);
        assertEquals("Done.", r[1]);
    }

    @Test
    void anAnswerWithoutThinkingIsUntouched() throws IOException {
        String[] r = split(false, List.of("Use ", "`a < b`", " and <", "b>bold</b>"));
        assertEquals("", r[0]);
        assertEquals("Use `a < b` and <b>bold</b>", r[1]);
    }
}
