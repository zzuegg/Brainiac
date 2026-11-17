package io.github.zzuegg.brainiac.experimental.llm;

/**
 * The user side of a multi-turn conversation with a language model, for training and
 * evaluating assistants beyond single turns (CollabLLM's user simulator, Lost-in-Conversation's
 * sharded instructions). The user holds an <em>intent</em> (the fully specified task) that the
 * assistant can only learn from the user's messages; the user opens the conversation, answers
 * each assistant response or ends the conversation, and finally scores the outcome.
 *
 * <p>Messages and responses are token arrays; the conversation's context is their
 * concatenation, starting with the opening message.
 *
 * @param <I> the intent type
 */
public interface SimulatedUser<I> {

    /**
     * @param intent what the user wants
     * @return the first user message
     */
    int[] opening(I intent);

    /**
     * @param intent what the user wants
     * @param context the conversation so far, ending with {@code response}
     * @param response the assistant's latest response
     * @param turn how many assistant responses there have been, this one included
     * @return the user's next message, or {@code null} when the conversation is over
     */
    int[] reply(I intent, int[] context, int[] response, int turn);

    /**
     * @param intent what the user wants
     * @param context the whole conversation
     * @param finalResponse the assistant's last response
     * @return the task score, for example 1 for a correct final answer
     */
    float outcome(I intent, int[] context, int[] finalResponse);
}
