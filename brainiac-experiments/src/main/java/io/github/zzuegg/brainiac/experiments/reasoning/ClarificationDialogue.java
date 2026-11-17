package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.SimulatedUser;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A toy multi-turn arithmetic dialogue with a simulated user, shared by the CollabLLM and
 * Lost-in-Conversation experiments. The user wants {@code a + b}. A fully specified request
 * opens with {@code a+b=}; an underspecified one with {@code a+?=}, and the second operand is
 * revealed only in a later user message, {@code b=}. The assistant answers with digits and
 * {@code #}, or asks with {@code ?}.
 *
 * <ul>
 *   <li>{@link Mode#COLLABORATIVE}: an answer ends the conversation; a question gets the
 *       missing operand as the reply (CollabLLM's setting: asking is the only way to know
 *       {@code b}).</li>
 *   <li>{@link Mode#SHARDED}: the instruction is split into shards that the user reveals one
 *       per turn whatever the assistant says, and only the answer after the last shard counts
 *       (Lost in Conversation's sharded simulation). An early answer stays in the context.</li>
 * </ul>
 */
public final class ClarificationDialogue implements SimulatedUser<ClarificationDialogue.Intent> {

    /** How the user behaves. */
    public enum Mode {
        /** Answer questions; an answer ends the conversation. */
        COLLABORATIVE,
        /** Reveal one shard per turn regardless of the response. */
        SHARDED
    }

    /**
     * What the user wants.
     *
     * @param a the first operand
     * @param b the second operand
     * @param specified whether the opening message states {@code b}
     */
    public record Intent(int a, int b, boolean specified) {
    }

    private final ArithmeticTask task;
    private final CharacterTokenizer tokenizer;
    private final int maxOperand;
    private final Mode mode;

    /**
     * @param maxOperand operands are drawn from {@code 0..maxOperand} (single digits)
     * @param mode the user's behaviour
     */
    public ClarificationDialogue(int maxOperand, Mode mode) {
        if (maxOperand > 9) {
            throw new IllegalArgumentException("single-digit operands only");
        }
        this.task = ArithmeticTask.addition(maxOperand);
        this.tokenizer = task.tokenizer();
        this.maxOperand = maxOperand;
        this.mode = mode;
    }

    /**
     * @param m the other mode
     * @return the same dialogue with another user behaviour
     */
    public ClarificationDialogue withMode(Mode m) {
        return new ClarificationDialogue(maxOperand, m);
    }

    /** @return the largest operand */
    public int maxOperand() {
        return maxOperand;
    }

    /** @return the vocabulary size */
    public int vocabSize() {
        return task.vocabSize();
    }

    /** @return the tokens that end a response: {@code #} and {@code ?} */
    public int[] stops() {
        return new int[] {task.eos(), tokenizer.encode("?")[0]};
    }

    /** @return the most tokens in a response */
    public int maxResponseTokens() {
        return task.maxAnswerTokens();
    }

    /**
     * @param random the random source
     * @param specifiedShare the chance that the request is fully specified
     * @return a random intent
     */
    public Intent sample(SplittableRandom random, double specifiedShare) {
        return new Intent(random.nextInt(maxOperand + 1), random.nextInt(maxOperand + 1),
                random.nextDouble() < specifiedShare);
    }

    /**
     * @param specified whether the requests are fully specified
     * @return every intent of that kind
     */
    public List<Intent> all(boolean specified) {
        List<Intent> out = new ArrayList<>();
        for (int a = 0; a <= maxOperand; a++) {
            for (int b = 0; b <= maxOperand; b++) {
                out.add(new Intent(a, b, specified));
            }
        }
        return out;
    }

    /**
     * @param intent an intent
     * @return the gold answer tokens, end token included
     */
    public int[] answer(Intent intent) {
        return tokenizer.encode((intent.a() + intent.b()) + "#");
    }

    /**
     * @param intent an intent
     * @return the tokens of the message that reveals {@code b}
     */
    public int[] reveal(Intent intent) {
        return tokenizer.encode(intent.b() + "=");
    }

    /** @return the question token as a response */
    public int[] question() {
        return tokenizer.encode("?");
    }

    @Override
    public int[] opening(Intent intent) {
        return tokenizer.encode(intent.a() + "+" + (intent.specified() ? String.valueOf(intent.b()) : "?") + "=");
    }

    private boolean asks(int[] response) {
        return response.length > 0 && response[response.length - 1] == question()[0];
    }

    @Override
    public int[] reply(Intent intent, int[] context, int[] response, int turn) {
        if (mode == Mode.SHARDED && !intent.specified()) {
            return turn == 1 ? reveal(intent) : null;   // the next shard, whatever the response
        }
        return asks(response) ? reveal(intent) : null;
    }

    @Override
    public float outcome(Intent intent, int[] context, int[] finalResponse) {
        Integer v = task.parse(finalResponse);
        return v != null && v == intent.a() + intent.b() ? 1 : 0;
    }
}
