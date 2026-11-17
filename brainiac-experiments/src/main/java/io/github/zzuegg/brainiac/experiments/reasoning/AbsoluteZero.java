package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Absolute Zero (Zhao et al., <a href="https://arxiv.org/abs/2505.03335">2505.03335</a>) on a
 * toy DSL: one model plays two roles with zero human data. As <em>proposer</em> it writes a
 * program and an input ({@code +3*2>4#}: add 3, times 2, mod 10, on input 4); an executor
 * validates it and computes the output. As <em>solver</em> it gets the deduction task
 * ({@code +3*2>4=}) and must answer ({@code 4#}). The solver is rewarded for being right;
 * the proposer is rewarded by learnability, {@code 1 - success rate} of the solver on its
 * task, and 0 for tasks that are always or never solved, so it proposes tasks at the edge
 * of the solver's ability. Both roles train with {@link GrpoTrainer} on the same weights. The
 * only data is one seed proposal ({@code +0>0#}), like the paper's identity-function seed.
 */
public final class AbsoluteZero {

    private static final CharacterTokenizer TOK = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private static final int EOS = TOK.encode("#")[0];

    /**
     * The outcome.
     *
     * @param solverBefore solver accuracy on held-out one-step programs before self-play
     * @param solverAfter and after
     * @param validBefore fraction of valid proposals before
     * @param validAfter and after
     * @param selfSolveBefore accuracy of the initial model on tasks the final proposer writes
     * @param selfSolveAfter accuracy of the final solver on them
     * @param distinctTasks distinct tasks among 64 final proposals
     */
    public record Result(double solverBefore, double solverAfter, double validBefore, double validAfter,
            double selfSolveBefore, double selfSolveAfter, int distinctTasks) {
    }

    private AbsoluteZero() {
    }

    /**
     * Runs a program of the DSL.
     *
     * @param program ops like {@code +3*2}
     * @param input the input digit
     * @return the output digit, or -1 if the program is malformed
     */
    public static int execute(String program, int input) {
        if (program.isEmpty() || program.length() % 2 != 0 || program.length() > 4) {
            return -1;
        }
        int x = input;
        for (int i = 0; i < program.length(); i += 2) {
            char op = program.charAt(i);
            char d = program.charAt(i + 1);
            if (!Character.isDigit(d)) {
                return -1;
            }
            int v = d - '0';
            switch (op) {
                case '+' -> x = (x + v) % 10;
                case '-' -> x = Math.floorMod(x - v, 10);
                case '*' -> x = (x * v) % 10;
                default -> {
                    return -1;
                }
            }
        }
        return x;
    }

    /**
     * Parses a proposal {@code program>input#}.
     *
     * @param proposal the proposer's completion
     * @return {program, input} or {@code null} if invalid
     */
    public static String[] parseProposal(int[] proposal) {
        String s = TOK.decode(proposal);
        if (!s.endsWith("#") || s.indexOf('>') < 0) {
            return null;
        }
        String program = s.substring(0, s.indexOf('>'));
        String input = s.substring(s.indexOf('>') + 1, s.length() - 1);
        if (input.length() != 1 || !Character.isDigit(input.charAt(0))) {
            return null;
        }
        return execute(program, input.charAt(0) - '0') < 0 ? null : new String[] {program, input};
    }

    /** How many leading characters of a proposal follow the format {@code (op digit){1,2} > digit #}. */
    static int validPrefix(int[] proposal) {
        String s = TOK.decode(proposal);
        int i = 0;
        int ops = 0;
        while (i + 1 < s.length() && "+-*".indexOf(s.charAt(i)) >= 0 && Character.isDigit(s.charAt(i + 1))
                && ops < 2) {
            i += 2;
            ops++;
        }
        if (ops == 0) {
            return i;
        }
        if (i < s.length() && s.charAt(i) == '>') {
            i++;
            if (i < s.length() && Character.isDigit(s.charAt(i))) {
                i++;
                if (i < s.length() && s.charAt(i) == '#') {
                    i++;
                }
            }
        }
        return i;
    }

    static int[] deduction(String program, String input) {
        return TOK.encode(program + ">" + input + "=");
    }

    static boolean solved(String program, String input, int[] answer) {
        return TOK.decode(answer).equals(execute(program, input.charAt(0) - '0') + "#");
    }

    static double solverAccuracy(WindowLm lm, List<String[]> tasks) {
        int[][] prompts = new int[tasks.size()][];
        for (int i = 0; i < prompts.length; i++) {
            prompts[i] = deduction(tasks.get(i)[0], tasks.get(i)[1]);
        }
        int[][] out = lm.sample(prompts, 2, EOS, 0, new SplittableRandom(0));
        int right = 0;
        for (int i = 0; i < out.length; i++) {
            right += solved(tasks.get(i)[0], tasks.get(i)[1], out[i]) ? 1 : 0;
        }
        return (double) right / tasks.size();
    }

    static double validity(WindowLm lm, SplittableRandom random) {
        int[][] ps = new int[64][];
        Arrays.fill(ps, TOK.encode("[+0>0]"));
        int valid = 0;
        for (int[] p : lm.sample(ps, 7, EOS, 1.0, random)) {
            valid += parseProposal(p) != null ? 1 : 0;
        }
        return valid / 64.0;
    }

    /**
     * @param iterations self-play iterations
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int iterations, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        WindowLm lm = WindowLm.of(TOK.size(), 14, 96, seed);
        List<String[]> heldOut = new ArrayList<>();
        for (char op : new char[] {'+', '-', '*'}) {
            for (int d = 0; d <= 9; d++) {
                heldOut.add(new String[] {op + String.valueOf(d), String.valueOf(random.nextInt(10))});
            }
        }
        double solverBefore = solverAccuracy(lm, heldOut);
        WindowLm initial = lm.copy();
        double validBefore = validity(lm, new SplittableRandom(seed + 1));
        GrpoTrainer proposerRl = GrpoTrainer.of(lm).optimizer(Optimizer.adam(3e-3f)).klCoefficient(0.05f)
                .innerEpochs(1).completions(7, EOS);
        GrpoTrainer solverRl = GrpoTrainer.of(lm).optimizer(Optimizer.adam(1e-2f)).klCoefficient(0.05f)
                .innerEpochs(1).completions(2, EOS);
        int[] seedPrompt = TOK.encode("[+0>0]");
        // the only seed, as in the paper: one identity-like triple, so the format is known
        Optimizer seedOpt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 5; s++) {
            lm.supervisedStep(List.of(seedPrompt), List.of(TOK.encode("+0>0#")), seedOpt);
        }
        int proposals = 24;
        java.util.ArrayDeque<String> buffer = new java.util.ArrayDeque<>();
        buffer.add("+0>0");
        int solves = 8;
        for (int it = 0; it < iterations; it++) {
            // as in AZR, the proposer sees a reference task from the buffer and writes a new one
            String reference = buffer.stream().skip(random.nextInt(buffer.size())).findFirst().orElseThrow();
            int[] proposerPrompt = TOK.encode("[" + reference + "]");
            int[][] ps = new int[proposals][];
            Arrays.fill(ps, proposerPrompt);
            int[][] proposed = lm.sample(ps, 7, EOS, 1.0, random);
            float[] proposerReward = new float[proposals];
            List<int[]> solverPrompts = new ArrayList<>();
            List<int[][]> solverGroups = new ArrayList<>();
            List<float[]> solverRewards = new ArrayList<>();
            for (int i = 0; i < proposals; i++) {
                String[] task = parseProposal(proposed[i]);
                if (task == null) {
                    proposerReward[i] = -1 + 0.1f * validPrefix(proposed[i]);   // graded format penalty
                    continue;
                }
                int[] prompt = deduction(task[0], task[1]);
                int[][] group = new int[solves][];
                Arrays.fill(group, prompt);
                int[][] answers = lm.sample(group, 2, EOS, 1.0, random);
                float[] r = new float[solves];
                int ok = 0;
                for (int g = 0; g < solves; g++) {
                    r[g] = solved(task[0], task[1], answers[g]) ? 1 : 0;
                    ok += (int) r[g];
                }
                double rate = (double) ok / solves;
                proposerReward[i] = rate == 0 || rate == 1 ? 0 : (float) (1 - rate);
                // AZR asks the proposer for tasks different from earlier ones; here a bonus for
                // tasks not in the recent buffer keeps the proposals diverse
                String key = task[0] + ">" + task[1];
                if (!buffer.contains(key)) {
                    proposerReward[i] += 0.3f;
                }
                buffer.addLast(key);
                if (buffer.size() > 100) {
                    buffer.removeFirst();
                }
                solverPrompts.add(prompt);
                solverGroups.add(answers);
                solverRewards.add(r);
            }
            proposerRl.update(List.of(proposerPrompt), List.<int[][]>of(proposed), List.of(proposerReward));
            if (!solverGroups.isEmpty()) {
                solverRl.update(solverPrompts, solverGroups, solverRewards);
            }
        }
        int[][] ps = new int[64][];
        Object[] refs = buffer.toArray();
        for (int i = 0; i < ps.length; i++) {
            ps[i] = TOK.encode("[" + refs[random.nextInt(refs.length)] + "]");
        }
        List<String[]> own = new ArrayList<>();
        java.util.Set<String> distinct = new java.util.HashSet<>();
        for (int[] p : lm.sample(ps, 7, EOS, 1.0, random)) {
            String[] task = parseProposal(p);
            if (task != null) {
                own.add(task);
                distinct.add(task[0] + ">" + task[1]);
            }
        }
        return new Result(solverBefore, solverAccuracy(lm, heldOut), validBefore,
                validity(lm, new SplittableRandom(seed + 2)), own.isEmpty() ? 0 : solverAccuracy(initial, own),
                own.isEmpty() ? 0 : solverAccuracy(lm, own), distinct.size());
    }
}
