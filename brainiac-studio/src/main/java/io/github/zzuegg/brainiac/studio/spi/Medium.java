package io.github.zzuegg.brainiac.studio.spi;

import java.util.Locale;

/**
 * What kind of thing a capability works with. The studio gives every medium its own color and
 * shows a task only the tools of its medium.
 */
public enum Medium {
    /** text in and out: chat, completion, embeddings */
    TEXT,
    /** pictures: generation, editing, upscaling */
    IMAGE,
    /** sound of voices: speech synthesis and transcription */
    SPEECH,
    /** understanding pictures and video frames */
    VISION,
    /** moving pictures */
    VIDEO,
    /** building, training and searching models */
    TRAINING,
    /** agents that plan and call other capabilities */
    AGENT;

    /**
     * Returns the lower-case name used in the studio's API, such as {@code "text"}.
     *
     * @return the API name
     */
    public String id() {
        return name().toLowerCase(Locale.ROOT);
    }
}
