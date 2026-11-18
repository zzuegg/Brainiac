package io.github.zzuegg.brainiac.audio;

import java.util.function.BooleanSupplier;

/**
 * How to transcribe.
 *
 * @param language the spoken language as a code ({@code en}) or name ({@code english}),
 *     or {@code null} to detect it from the first 30 seconds
 * @param task transcribe in the spoken language, or translate to English
 * @param timestamps whether to split the text into timed segments (otherwise one per 30-second window)
 * @param temperature the first sampling temperature; 0 decodes greedily, and a window
 *     that looks wrong (repetitive or unsure) is retried at higher temperatures
 * @param prompt text that comes before the audio (names, spelling, style), or {@code null}
 * @param conditionOnPreviousText whether each window sees the text of the windows before it
 * @param cancelled polled while decoding: when it returns {@code true}, transcription stops
 *     with a {@link java.util.concurrent.CancellationException}
 */
public record TranscriptionOptions(String language, Task task, boolean timestamps, float temperature, String prompt,
        boolean conditionOnPreviousText, BooleanSupplier cancelled) {

    /** What to do with the speech. */
    public enum Task {
        /** Text in the spoken language. */
        TRANSCRIBE,
        /** English text, whatever the spoken language. */
        TRANSLATE
    }

    /**
     * Checks the values.
     *
     * @param language the language or {@code null}
     * @param task the task
     * @param timestamps whether to time segments
     * @param temperature the first temperature, 0 or more
     * @param prompt the prompt or {@code null}
     * @param conditionOnPreviousText whether windows see earlier text
     * @param cancelled the cancellation check, or {@code null} for never
     */
    public TranscriptionOptions {
        if (task == null) {
            task = Task.TRANSCRIBE;
        }
        if (!(temperature >= 0)) {
            throw new IllegalArgumentException("temperature must be 0 or more, got " + temperature);
        }
        if (cancelled == null) {
            cancelled = () -> false;
        }
        if (language != null && language.isBlank()) {
            language = null;
        }
    }

    /**
     * The defaults: detect the language, transcribe, timed segments, greedy decoding with
     * fallback, conditioned on earlier text.
     *
     * @return the options
     */
    public static TranscriptionOptions defaults() {
        return new TranscriptionOptions(null, Task.TRANSCRIBE, true, 0, null, true, null);
    }

    /**
     * Sets the language.
     *
     * @param code a code such as {@code de}, or {@code null} to detect it
     * @return new options
     */
    public TranscriptionOptions language(String code) {
        return new TranscriptionOptions(code, task, timestamps, temperature, prompt, conditionOnPreviousText, cancelled);
    }

    /**
     * Sets the task.
     *
     * @param t transcribe or translate
     * @return new options
     */
    public TranscriptionOptions task(Task t) {
        return new TranscriptionOptions(language, t, timestamps, temperature, prompt, conditionOnPreviousText, cancelled);
    }

    /**
     * Sets whether to time segments.
     *
     * @param on whether to
     * @return new options
     */
    public TranscriptionOptions timestamps(boolean on) {
        return new TranscriptionOptions(language, task, on, temperature, prompt, conditionOnPreviousText, cancelled);
    }

    /**
     * Sets the first temperature.
     *
     * @param t 0 for greedy
     * @return new options
     */
    public TranscriptionOptions temperature(float t) {
        return new TranscriptionOptions(language, task, timestamps, t, prompt, conditionOnPreviousText, cancelled);
    }

    /**
     * Sets a prompt.
     *
     * @param text earlier text, or {@code null}
     * @return new options
     */
    public TranscriptionOptions prompt(String text) {
        return new TranscriptionOptions(language, task, timestamps, temperature, text, conditionOnPreviousText, cancelled);
    }

    /**
     * Sets whether windows see earlier text.
     *
     * @param on whether they do
     * @return new options
     */
    public TranscriptionOptions conditionOnPreviousText(boolean on) {
        return new TranscriptionOptions(language, task, timestamps, temperature, prompt, on, cancelled);
    }

    /**
     * Sets the cancellation check.
     *
     * @param check returns {@code true} to stop
     * @return new options
     */
    public TranscriptionOptions cancelled(BooleanSupplier check) {
        return new TranscriptionOptions(language, task, timestamps, temperature, prompt, conditionOnPreviousText, check);
    }
}
