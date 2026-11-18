package io.github.zzuegg.brainiac.audio;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Orpheus's prompt and token layout (the language model and SNAC are tested on their own). */
class OrpheusTest {

    @Test
    void promptWrapsTheTextInSpeechMarkers() {
        assertArrayEquals(new int[] {128259, 128000, 5, 6, 128009, 128260, 128261, 128257},
                OrpheusSpeech.prompt(new int[] {128000, 5, 6}));
    }

    @Test
    void tokensBecomeThreeLevelsOfCodes() {
        int base = OrpheusSpeech.AUDIO_BASE;
        int[] frame1 = {1, 2, 3, 4, 5, 6, 7};
        int[] frame2 = {10, 20, 30, 40, 50, 60, 70};
        int[] tokens = new int[3 + 21 + 2];
        int at = 0;
        tokens[at++] = 128261;
        tokens[at++] = 128257;
        tokens[at++] = 999;                                     // text tokens are skipped
        for (int[] f : new int[][] {frame1, {4095, 9999, 0, 0, 0, 0, 0}, frame2}) {
            for (int j = 0; j < 7; j++) {
                tokens[at++] = base + j * 4096 + f[j];          // the middle frame's second code is out of range
            }
        }
        tokens[at++] = 128258;
        tokens[at] = base + 5;                                  // after the end: ignored
        int[][] codes = OrpheusSpeech.codes(tokens);
        assertArrayEquals(new int[] {1, 10}, codes[0]);
        assertArrayEquals(new int[] {2, 5, 20, 50}, codes[1]);
        assertArrayEquals(new int[] {3, 4, 6, 7, 30, 40, 60, 70}, codes[2]);
    }

    @Test
    void textIsCutIntoPiecesAtSentences() {
        String s = "A".repeat(200) + ". " + "B".repeat(150) + "! Short one. <laugh> And another?";
        List<String> p = OrpheusSpeech.pieces(s);
        assertEquals(2, p.size());
        assertEquals("A".repeat(200) + ".", p.get(0));
        assertTrue(p.get(1).endsWith("Short one. <laugh> And another?"), p.get(1));
    }

    @Test
    void tagsAndVoicesAreListed() {
        assertTrue(OrpheusSpeech.TAGS.contains("<laugh>"));
        assertEquals("tara", OrpheusSpeech.VOICES.getFirst());
    }
}
