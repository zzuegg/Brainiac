package io.github.zzuegg.brainiac.studio.chat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * What a speech model offers, so the page only shows settings that do something.
 *
 * @param voices the voices' names
 * @param speed whether the pace can be set
 * @param instructions whether it takes instructions on how to speak
 * @param formats the file formats it can write
 */
public record SpeechOptions(List<String> voices, boolean speed, boolean instructions, List<String> formats) {

    /** Copies the lists. */
    public SpeechOptions {
        voices = List.copyOf(voices);
        formats = List.copyOf(formats);
    }

    Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("voices", voices);
        m.put("speed", speed);
        m.put("instructions", instructions);
        m.put("formats", formats);
        return m;
    }
}
