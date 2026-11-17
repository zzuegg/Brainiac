package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic arithmetic puzzles with a programmatic verifier, the toy stand-in for "RL with
 * verifiable rewards": prompts like {@code 3+4=}, answers like {@code 7#} ({@code #} ends
 * the answer). The reward is 1 for an exactly right answer, a small format reward for a
 * well-formed wrong one (digits then the end token, as in DeepSeek-R1's format reward) and 0
 * otherwise.
 */
public final class ArithmeticTask {

    /** The characters of every arithmetic experiment: digits, operators, tool brackets, end. */
    public static final String CHARACTERS = "0123456789+-*=[]>?#";

    private final CharacterTokenizer tokenizer = new CharacterTokenizer(CHARACTERS);
    private final int maxOperand;
    private final char op;
    private final float formatReward;

    private ArithmeticTask(int maxOperand, char op, float formatReward) {
        this.maxOperand = maxOperand;
        this.op = op;
        this.formatReward = formatReward;
    }

    /**
     * @param maxOperand operands are drawn from {@code 0..maxOperand}
     * @return addition puzzles
     */
    public static ArithmeticTask addition(int maxOperand) {
        return new ArithmeticTask(maxOperand, '+', 0.1f);
    }

    /**
     * @param maxOperand operands are drawn from {@code 0..maxOperand}
     * @return multiplication puzzles
     */
    public static ArithmeticTask multiplication(int maxOperand) {
        return new ArithmeticTask(maxOperand, '*', 0.1f);
    }

    /** @return the tokenizer */
    public CharacterTokenizer tokenizer() {
        return tokenizer;
    }

    /** @return the vocabulary size */
    public int vocabSize() {
        return tokenizer.size();
    }

    /** @return the end token */
    public int eos() {
        return tokenizer.encode("#")[0];
    }

    /** @return the longest answer, in tokens, including the end token */
    public int maxAnswerTokens() {
        return String.valueOf(apply(maxOperand, maxOperand)).length() + 1;
    }

    private int apply(int a, int b) {
        return op == '+' ? a + b : a * b;
    }

    /**
     * @param a the first operand
     * @param b the second operand
     * @return the prompt tokens
     */
    public int[] prompt(int a, int b) {
        return tokenizer.encode(a + String.valueOf(op) + b + "=");
    }

    /**
     * @param random the random source
     * @return a random prompt
     */
    public int[] samplePrompt(SplittableRandom random) {
        return prompt(random.nextInt(maxOperand + 1), random.nextInt(maxOperand + 1));
    }

    /** @return every prompt, for evaluation */
    public List<int[]> allPrompts() {
        List<int[]> out = new ArrayList<>();
        for (int a = 0; a <= maxOperand; a++) {
            for (int b = 0; b <= maxOperand; b++) {
                out.add(prompt(a, b));
            }
        }
        return out;
    }

    /**
     * @param prompt a prompt
     * @return the right answer as a number
     */
    public int answer(int[] prompt) {
        String text = tokenizer.decode(prompt);
        int at = text.indexOf(op);
        int a = Integer.parseInt(text.substring(0, at));
        int b = Integer.parseInt(text.substring(at + 1, text.length() - 1));
        return apply(a, b);
    }

    /**
     * @param prompt a prompt
     * @return the right completion tokens, end token included
     */
    public int[] solution(int[] prompt) {
        return tokenizer.encode(answer(prompt) + "#");
    }

    /**
     * Parses the number a completion states.
     *
     * @param completion the completion
     * @return the number, or {@code null} if it is not digits followed by the end token
     */
    public Integer parse(int[] completion) {
        String s = tokenizer.decode(completion);
        int end = s.indexOf('#');
        if (end <= 0 || end > 4) {
            return null;
        }
        String digits = s.substring(0, end);
        for (int i = 0; i < digits.length(); i++) {
            if (!Character.isDigit(digits.charAt(i))) {
                return null;
            }
        }
        return Integer.parseInt(digits);
    }

    /**
     * @param prompt the prompt
     * @param completion the completion
     * @return whether it states the right answer
     */
    public boolean correct(int[] prompt, int[] completion) {
        Integer v = parse(completion);
        return v != null && v == answer(prompt);
    }

    /**
     * The verifiable reward.
     *
     * @param prompt the prompt
     * @param completion the completion
     * @return 1 if right, the format reward if well-formed, else 0
     */
    public float reward(int[] prompt, int[] completion) {
        Integer v = parse(completion);
        if (v == null) {
            return 0;
        }
        return v == answer(prompt) ? 1 : formatReward;
    }
}
