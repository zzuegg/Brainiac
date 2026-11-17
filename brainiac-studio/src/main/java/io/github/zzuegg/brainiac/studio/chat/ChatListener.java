package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;

/** Receives an answer while it is written. */
public interface ChatListener {

    /**
     * Called with each new piece of the answer.
     *
     * @param text the piece
     * @throws IOException if passing it on fails, which stops the answer
     */
    void text(String text) throws IOException;

    /**
     * Called with each new piece of the model's visible reasoning, for models that show it.
     *
     * @param text the piece
     * @throws IOException if passing it on fails, which stops the answer
     */
    default void thinking(String text) throws IOException {
    }
}
