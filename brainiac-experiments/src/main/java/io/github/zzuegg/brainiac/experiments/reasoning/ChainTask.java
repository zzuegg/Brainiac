package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Multi-step arithmetic with a visible chain of thought: {@code 3+4+2=} is solved as
 * {@code 7>9#}, one partial sum per step ({@code >} ends a step, {@code #} the solution).
 * Each step can be checked on its own, which is what process reward models and step-level
 * search need.
 */
public final class ChainTask {

    private final CharacterTokenizer tokenizer = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private final int maxOperand;

    /**
     * @param maxOperand the three operands are drawn from {@code 0..maxOperand}
     */
    public ChainTask(int maxOperand) {
        this.maxOperand = maxOperand;
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

    /** @return the step token */
    public int stepEnd() {
        return tokenizer.encode(">")[0];
    }

    /** @return the longest solution in tokens */
    public int maxSolutionTokens() {
        return 2 * String.valueOf(3 * maxOperand).length() + 2;
    }

    /** @return every prompt */
    public List<int[]> allPrompts() {
        List<int[]> out = new ArrayList<>();
        for (int a = 0; a <= maxOperand; a++) {
            for (int b = 0; b <= maxOperand; b++) {
                for (int c = 0; c <= maxOperand; c++) {
                    out.add(tokenizer.encode(a + "+" + b + "+" + c + "="));
                }
            }
        }
        return out;
    }

    private int[] operands(int[] prompt) {
        String[] parts = tokenizer.decode(prompt).replace("=", "").split("\\+");
        return new int[] {Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2])};
    }

    /**
     * @param prompt a prompt
     * @return the right final answer
     */
    public int answer(int[] prompt) {
        int[] o = operands(prompt);
        return o[0] + o[1] + o[2];
    }

    /**
     * @param prompt a prompt
     * @return the right chain of thought, end token included
     */
    public int[] solution(int[] prompt) {
        int[] o = operands(prompt);
        return tokenizer.encode((o[0] + o[1]) + ">" + (o[0] + o[1] + o[2]) + "#");
    }

    /**
     * @param completion a solution
     * @return its final answer, or {@code null} if malformed
     */
    public Integer finalAnswer(int[] completion) {
        String s = tokenizer.decode(completion);
        if (!s.endsWith("#")) {
            return null;
        }
        String last = s.substring(s.lastIndexOf('>') + 1, s.length() - 1);
        if (last.isEmpty() || last.length() > 3 || !last.chars().allMatch(Character::isDigit)) {
            return null;
        }
        return Integer.parseInt(last);
    }

    /**
     * @param prompt a prompt
     * @param completion a solution
     * @return whether its final answer is right
     */
    public boolean correct(int[] prompt, int[] completion) {
        Integer v = finalAnswer(completion);
        return v != null && v == answer(prompt);
    }

    /**
     * Labels each step: 1 while every step so far is right, 0 from the first mistake on.
     * Steps are cut as {@code ProcessRewardModel.stepPrefixes} cuts them.
     *
     * @param prompt a prompt
     * @param completion a solution
     * @return one label per step
     */
    public float[] stepLabels(int[] prompt, int[] completion) {
        int[] o = operands(prompt);
        int[] expected = {o[0] + o[1], o[0] + o[1] + o[2]};
        String s = tokenizer.decode(completion);
        List<String> steps = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '>' || ch == '#' || i == s.length() - 1) {
                steps.add(s.substring(start, i + 1));
                start = i + 1;
            }
        }
        float[] labels = new float[steps.size()];
        boolean ok = true;
        for (int k = 0; k < labels.length; k++) {
            String step = steps.get(k);
            boolean closed = step.endsWith(">") || step.endsWith("#");
            String digits = closed ? step.substring(0, step.length() - 1) : step;
            boolean isLast = step.endsWith("#");
            ok &= k < 2 && closed && !digits.isEmpty() && digits.length() <= 3
                    && digits.chars().allMatch(Character::isDigit)
                    && Integer.parseInt(digits) == expected[k] && (isLast == (k == 1));
            labels[k] = ok ? 1 : 0;
        }
        return labels;
    }
}
