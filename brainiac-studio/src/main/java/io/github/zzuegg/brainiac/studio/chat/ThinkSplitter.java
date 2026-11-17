package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;

/**
 * Passes on an answer, sending what a model writes between {@code <think>} and {@code </think>}
 * (Qwen3, DeepSeek-R1 and other reasoning models that think in plain text) as thinking rather than
 * as the answer. Tags split across pieces of the stream are put back together; text that only might
 * start a tag is held until the next piece says whether it does.
 */
public final class ThinkSplitter implements ChatListener {

    private static final String OPEN = "<think>";
    private static final String CLOSE = "</think>";

    private final ChatListener to;
    private final StringBuilder held = new StringBuilder();
    private boolean thinking;
    private boolean trim;

    /**
     * Splits an answer.
     *
     * @param to receives the answer and the thinking
     * @param thinking true when the prompt already opened a {@code <think>} block, so the answer
     *     starts inside it
     */
    public ThinkSplitter(ChatListener to, boolean thinking) {
        this.to = to;
        this.thinking = thinking;
        this.trim = true;
    }

    @Override
    public void text(String text) throws IOException {
        held.append(text);
        while (true) {
            String tag = thinking ? CLOSE : OPEN;
            int at = held.indexOf(tag);
            if (at >= 0) {
                emit(held.substring(0, at));
                held.delete(0, at + tag.length());
                thinking = !thinking;
                trim = true;   // the blank lines around a think block are not part of either side
                continue;
            }
            int keep = partial(held, tag);
            emit(held.substring(0, held.length() - keep));
            held.delete(0, held.length() - keep);
            return;
        }
    }

    @Override
    public void thinking(String text) throws IOException {
        to.thinking(text);
    }

    /**
     * Passes on what is still held, at the end of the answer.
     *
     * @throws IOException if the listener fails
     */
    public void finish() throws IOException {
        emit(held.toString());
        held.setLength(0);
    }

    private void emit(String s) throws IOException {
        if (trim) {
            s = s.stripLeading();
            if (s.isEmpty()) {
                return;
            }
            trim = false;
        }
        if (s.isEmpty()) {
            return;
        }
        if (thinking) {
            to.thinking(s);
        } else {
            to.text(s);
        }
    }

    // how many characters at the end could be the start of the tag
    private static int partial(CharSequence s, String tag) {
        for (int n = Math.min(tag.length() - 1, s.length()); n > 0; n--) {
            if (tag.startsWith(s.subSequence(s.length() - n, s.length()).toString())) {
                return n;
            }
        }
        return 0;
    }
}
