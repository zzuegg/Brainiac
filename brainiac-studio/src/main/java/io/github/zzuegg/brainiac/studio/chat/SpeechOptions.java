package io.github.zzuegg.brainiac.studio.chat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * What a speech model offers, so the page only shows settings that do something.
 *
 * @param voices the voices' ids
 * @param speed whether the pace can be set
 * @param instructions whether it takes instructions on how to speak
 * @param formats the file formats it can write
 * @param names a readable name for some voices, by id, such as "Heart (American female)"
 * @param mix whether a voice may be a blend, such as {@code af_heart*0.7+am_adam*0.3}
 * @param tags expression tags it understands inside the text, such as {@code laugh}
 * @param clones whether it speaks in a voice made from a recording
 */
public record SpeechOptions(List<String> voices, boolean speed, boolean instructions, List<String> formats,
        Map<String, String> names, boolean mix, List<String> tags, boolean clones) {

    /** Copies the lists. */
    public SpeechOptions {
        voices = List.copyOf(voices);
        formats = List.copyOf(formats);
        names = Map.copyOf(names);
        tags = List.copyOf(tags);
    }

    /**
     * Options of a model whose voices are plain names and do not blend.
     *
     * @param voices the voices' names
     * @param speed whether the pace can be set
     * @param instructions whether it takes instructions on how to speak
     * @param formats the file formats it can write
     */
    public SpeechOptions(List<String> voices, boolean speed, boolean instructions, List<String> formats) {
        this(voices, speed, instructions, formats, Map.of(), false, List.of(), false);
    }

    Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("voices", voices);
        m.put("speed", speed);
        m.put("instructions", instructions);
        m.put("formats", formats);
        // in the voices' order, so the page can show them as they come
        Map<String, String> n = new LinkedHashMap<>();
        voices.stream().filter(names::containsKey).forEach(v -> n.put(v, names.get(v)));
        m.put("names", n);
        m.put("mix", mix);
        m.put("tags", tags);
        m.put("clone", clones);
        return m;
    }
}
