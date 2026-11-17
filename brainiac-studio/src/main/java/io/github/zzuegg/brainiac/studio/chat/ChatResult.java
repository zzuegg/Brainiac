package io.github.zzuegg.brainiac.studio.chat;

/**
 * How an answer ended.
 *
 * @param stop why it ended: {@code end}, {@code length}, {@code refusal}, or the provider's own word
 * @param inputTokens tokens read, or -1 if unknown
 * @param outputTokens tokens written, or -1 if unknown
 */
public record ChatResult(String stop, long inputTokens, long outputTokens) {
}
