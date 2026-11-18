package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A step-segmented version of {@link ChainTask} with a variable number of steps and error
 * injection, for process supervision (wave 9, 9.2.8 to 9.2.12). The prompt lists {@code n}
 * digits, {@code 3+4+2+5=}; the solution writes the running sum modulo 10 after each further
 * operand, one digit per step, {@code 7>9>4#} ({@code >} ends a step, {@code #} the solution).
 * Every step is one token plus its terminator and can be checked exactly, so step labels are
 * exact, chains can be made long without long numbers, and an error can be injected at a known
 * step ({@link #inject}): the later steps then continue correctly from the wrong value, so the
 * final answer is wrong while every later step looks locally consistent.
 */
public final class StepChainTask {

    private final CharacterTokenizer tokenizer = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private final int minOperands;
    private final int maxOperands;
    private final int maxOperand;

    /**
     * @param minOperands the fewest operands (steps = operands - 1)
     * @param maxOperands the most operands
     * @param maxOperand operands are drawn from {@code 0..maxOperand}
     */
    public StepChainTask(int minOperands, int maxOperands, int maxOperand) {
        if (minOperands < 2 || maxOperands < minOperands || maxOperand > 9) {
            throw new IllegalArgumentException("bad chain settings");
        }
        this.minOperands = minOperands;
        this.maxOperands = maxOperands;
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
        return 2 * (maxOperands - 1);
    }

    /** @return the longest prompt in tokens */
    public int maxPromptTokens() {
        return 2 * maxOperands;
    }

    /**
     * @param random the random source
     * @return a random prompt
     */
    public int[] samplePrompt(SplittableRandom random) {
        int n = minOperands + random.nextInt(maxOperands - minOperands + 1);
        int[] p = new int[2 * n];
        for (int i = 0; i < n; i++) {
            p[2 * i] = random.nextInt(maxOperand + 1);
            p[2 * i + 1] = tokenizer.encode(i == n - 1 ? "=" : "+")[0];
        }
        return p;
    }

    /**
     * @param count how many
     * @param seed the seed
     * @return distinct random prompts (as many as exist, if fewer)
     */
    public List<int[]> prompts(int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> out = new ArrayList<>();
        java.util.Set<String> seen = new java.util.HashSet<>();
        for (int tries = 0; out.size() < count && tries < 50 * count; tries++) {
            int[] p = samplePrompt(r);
            if (seen.add(java.util.Arrays.toString(p))) {
                out.add(p);
            }
        }
        return out;
    }

    /**
     * @param prompt a prompt
     * @return its operands
     */
    public int[] operands(int[] prompt) {
        int[] o = new int[prompt.length / 2];
        for (int i = 0; i < o.length; i++) {
            o[i] = prompt[2 * i];
        }
        return o;
    }

    /**
     * @param prompt a prompt
     * @return its number of steps
     */
    public int steps(int[] prompt) {
        return prompt.length / 2 - 1;
    }

    /**
     * @param prompt a prompt
     * @return the right value of each step
     */
    public int[] partials(int[] prompt) {
        int[] o = operands(prompt);
        int[] s = new int[o.length - 1];
        int acc = o[0];
        for (int i = 1; i < o.length; i++) {
            acc = (acc + o[i]) % 10;
            s[i - 1] = acc;
        }
        return s;
    }

    /**
     * @param prompt a prompt
     * @return the right final answer
     */
    public int answer(int[] prompt) {
        int[] p = partials(prompt);
        return p[p.length - 1];
    }

    private int[] write(int[] values) {
        int[] out = new int[2 * values.length];
        for (int i = 0; i < values.length; i++) {
            out[2 * i] = values[i];
            out[2 * i + 1] = i == values.length - 1 ? eos() : stepEnd();
        }
        return out;
    }

    /**
     * @param prompt a prompt
     * @return the right solution, end token included
     */
    public int[] solution(int[] prompt) {
        return write(partials(prompt));
    }

    /**
     * A solution with an error at {@code step}: that step's value is off, and the later steps
     * continue correctly from it.
     *
     * @param prompt a prompt
     * @param step the wrong step (0-based)
     * @param random picks the wrong value
     * @return the solution
     */
    public int[] inject(int[] prompt, int step, SplittableRandom random) {
        int[] o = operands(prompt);
        int[] v = partials(prompt);
        int acc = v[step];
        acc = (acc + 1 + random.nextInt(9)) % 10;
        v[step] = acc;
        for (int i = step + 1; i < v.length; i++) {
            acc = (acc + o[i + 1]) % 10;
            v[i] = acc;
        }
        return write(v);
    }

    /**
     * A solution in which each step independently goes wrong with probability {@code rate}
     * (errors propagate as in {@link #inject}).
     *
     * @param prompt a prompt
     * @param rate the per-step error rate
     * @param random the random source
     * @return the solution
     */
    public int[] noisySolution(int[] prompt, double rate, SplittableRandom random) {
        int[] o = operands(prompt);
        int[] v = new int[o.length - 1];
        int acc = o[0];
        for (int i = 1; i < o.length; i++) {
            acc = (acc + o[i]) % 10;
            if (random.nextDouble() < rate) {
                acc = (acc + 1 + random.nextInt(9)) % 10;
            }
            v[i - 1] = acc;
        }
        return write(v);
    }

    /**
     * @param prompt a prompt
     * @param completion a solution
     * @return whether it is well formed and ends with the right answer
     */
    public boolean correct(int[] prompt, int[] completion) {
        int n = completion.length;
        return n >= 2 && completion[n - 1] == eos() && completion[n - 2] < 10 && completion[n - 2] == answer(prompt)
                && (n == 2 || completion[n - 3] == stepEnd());
    }

    /**
     * Labels each step of a (possibly partial) solution: 1 while every step so far is right
     * (the right digit, then {@code >}, or {@code #} for the last step), 0 from the first
     * mistake on. Steps are cut after {@code >} and {@code #}, and a trailing unfinished step
     * counts as one, as {@code ProcessRewardModel.stepPrefixes} cuts them.
     *
     * @param prompt a prompt
     * @param completion a solution
     * @return one label per step
     */
    public float[] stepLabels(int[] prompt, int[] completion) {
        int[] expected = partials(prompt);
        List<int[]> steps = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < completion.length; i++) {
            if (completion[i] == stepEnd() || completion[i] == eos() || i == completion.length - 1) {
                steps.add(java.util.Arrays.copyOfRange(completion, start, i + 1));
                start = i + 1;
            }
        }
        float[] labels = new float[steps.size()];
        boolean ok = true;
        for (int k = 0; k < labels.length; k++) {
            int[] s = steps.get(k);
            int terminator = k == expected.length - 1 ? eos() : stepEnd();
            ok &= k < expected.length && s.length == 2 && s[0] == expected[k] && s[1] == terminator;
            labels[k] = ok ? 1 : 0;
        }
        return labels;
    }
}
