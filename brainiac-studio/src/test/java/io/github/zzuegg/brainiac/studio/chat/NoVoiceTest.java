package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** A voice the model does not have is answered with the ones it has, the closest first. */
class NoVoiceTest {

    @Test
    void theRealVoicesComeBackTheClosestFirst() {
        Set<String> known = new LinkedHashSet<>(List.of("af_heart", "am_adam", "bf_emma"));
        String m = LocalSpeech.noVoice("kokoro", "en-GB_EmmaVoice", known, false);
        assertTrue(m.startsWith("kokoro has no voice named en-GB_EmmaVoice. Its voices are: bf_emma"), m);
    }

    @Test
    void aModelWithoutNamedVoicesSaysWhatToDoInstead() {
        String m = LocalSpeech.noVoice("f5-tts", "allison", Set.of(), true);
        assertTrue(m.contains("only in voices made from a recording") && m.contains("Kokoro or Orpheus"), m);
    }
}
