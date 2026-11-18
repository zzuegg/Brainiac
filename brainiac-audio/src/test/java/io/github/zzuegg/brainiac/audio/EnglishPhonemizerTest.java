package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The phonemizer against misaki with spaCy tags (see g2p/make_g2p.py). */
class EnglishPhonemizerTest {

    @Test
    void matchesMisaki() throws Exception {
        Map<String, Object> expected = Json.object(Json.read(WhisperTest.resource("g2p").resolve("expected.json")));
        List<Object> sentences = Json.array(expected.get("sentences"));
        int total = 0;
        int same = 0;
        for (String accent : new String[] {"us", "gb"}) {
            EnglishPhonemizer g2p = accent.equals("us") ? EnglishPhonemizer.american() : EnglishPhonemizer.british();
            List<Object> want = Json.array(expected.get(accent));
            for (int i = 0; i < sentences.size(); i++) {
                String[] w = ((String) want.get(i)).split(" ");
                String[] got = g2p.phonemize((String) sentences.get(i)).split(" ");
                if (w.length != got.length) {
                    System.out.println("G2P " + accent + " length: " + want.get(i) + "\n    got " + String.join(" ", got));
                    total += w.length;
                    continue;
                }
                for (int k = 0; k < w.length; k++) {
                    if (w[k].contains("❓")) {
                        continue;
                    }
                    total++;
                    if (w[k].replace('ɾ', 'T').equals(got[k])) {
                        same++;
                    } else {
                        System.out.println("G2P " + accent + " " + w[k] + " vs " + got[k] + "   in " + sentences.get(i));
                    }
                }
            }
        }
        System.out.println("G2P " + same + " of " + total + " words match");
        assertTrue(same >= 0.95 * total, same + " of " + total);
    }

    @Test
    void readsUnknownWords() {
        EnglishPhonemizer g2p = EnglishPhonemizer.american();
        String ps = g2p.phonemize("Kokoro and Brainiac");
        assertTrue(!ps.contains("❓") && ps.split(" ").length == 3, ps);
        assertEquals("", g2p.phonemize("   "));
    }
}
