package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class PromptLookupTest {

    private static int argmax(float[] row) {
        int best = 0;
        for (int i = 1; i < row.length; i++) {
            if (row[i] > row[best]) {
                best = i;
            }
        }
        return best;
    }

    @Test
    void guessesWhatFollowedTheLatestLongestMatch() {
        PromptLookup draft = new PromptLookup(10, 2, 1);
        // "1 2" was followed by 3; the later lone "2" by 4, but longer matches win
        assertEquals(3, draft.guess(new int[] {1, 2, 3, 9, 2, 4, 1, 2}, 8));
        assertEquals(4, new PromptLookup(10, 1, 1).guess(new int[] {1, 2, 3, 9, 2, 4, 1, 2}, 8));
        assertEquals(-1, draft.guess(new int[] {1, 2, 3}, 3));
    }

    @Test
    void readsGuessesAndGoesBack() {
        PromptLookup draft = new PromptLookup(10);
        LanguageModel.State s = draft.newState(1);
        float[] l = draft.next(s, new int[][] {{5, 6, 7, 8, 5, 6}}, null).toArray();
        assertEquals(7, argmax(l));
        l = draft.next(s, new int[][] {{7}}, null).toArray();
        assertEquals(8, argmax(l));
        assertEquals(7, s.length());
        s = s.truncate(6);
        l = draft.next(s, new int[][] {{9}}, null).toArray();
        // no earlier 9: every token equally likely
        assertArrayEquals(new float[10], l);
    }
}
