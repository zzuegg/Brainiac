package io.github.zzuegg.brainiac.audio;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KokoroSpeechTest {

    @Test
    void parsesVoiceBlends() {
        assertEquals(Map.of("af_heart", 1.0), KokoroSpeech.parseMix("af_heart"));
        Map<String, Double> mix = KokoroSpeech.parseMix("af_heart*3 + am_adam*1");
        assertEquals(0.75, mix.get("af_heart"), 1e-9);
        assertEquals(0.25, mix.get("am_adam"), 1e-9);
        assertEquals(Map.of("a", 0.5, "b", 0.5), KokoroSpeech.parseMix("a,b"));
        assertThrows(IllegalArgumentException.class, () -> KokoroSpeech.parseMix("a*x"));
        assertThrows(IllegalArgumentException.class, () -> KokoroSpeech.parseMix(" , "));
    }

    @Test
    void chunksBySentenceWithinTheLimit() {
        EnglishPhonemizer g2p = EnglishPhonemizer.american();
        assertEquals(2, KokoroSpeech.chunks(g2p, "Hello there. How are you?").size());
        String longText = "word, ".repeat(200) + "end.";
        List<String> pieces = KokoroSpeech.chunks(g2p, longText);
        assertTrue(pieces.size() > 1);
        for (String p : pieces) {
            assertTrue(p.length() <= KokoroSpeech.MAX_PHONEMES, p.length() + "");
        }
    }
}
