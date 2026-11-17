package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.llm.ReflexionLoop;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Reflexion with a tiny language model on multiplication: a half-trained model guesses, a
 * checker says "too high" or "too low", the reflector writes that down as a lesson ("not 56
 * or more"), and the next attempt draws samples from the model until one agrees with every
 * lesson. Plain retrying, with the same number of trials and the same feedback but no memory,
 * repeats its mistakes.
 */
public final class ReflectiveSolver {

    /**
     * @param firstTry the share of puzzles solved at the first attempt
     * @param retry the share solved within the trial budget by plain retrying
     * @param reflexion the share solved with reflections
     * @param meanTrialsReflexion the mean trials Reflexion needed when it succeeded
     */
    public record Result(double firstTry, double retry, double reflexion, double meanTrialsReflexion) {
    }

    private ReflectiveSolver() {
    }

    /**
     * @param sftSteps supervised steps for the model (few, so it stays unreliable)
     * @param trials the trial budget
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int sftSteps, int trials, long seed) {
        ArithmeticTask task = ArithmeticTask.multiplication(9);
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        List<int[]> prompts = task.allPrompts();
        List<int[]> answers = new ArrayList<>();
        for (int[] p : prompts) {
            answers.add(task.solution(p));
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < sftSteps; s++) {
            lm.supervisedStep(prompts, answers, opt);
        }
        SplittableRandom random = new SplittableRandom(seed);
        int first = 0;
        int retried = 0;
        int reflected = 0;
        int reflexionTrials = 0;
        for (int[] prompt : prompts) {
            first += solve(lm, task, prompt, 0, 1, random).success() ? 1 : 0;
            retried += solve(lm, task, prompt, 0, trials, random).success() ? 1 : 0;
            ReflexionLoop.Outcome<Integer> o = solve(lm, task, prompt, trials, trials, random);
            if (o.success()) {
                reflected++;
                reflexionTrials += o.trials();
            }
        }
        double n = prompts.size();
        return new Result(first / n, retried / n, reflected / n, (double) reflexionTrials / Math.max(1, reflected));
    }

    private static ReflexionLoop.Outcome<Integer> solve(WindowLm lm, ArithmeticTask task, int[] prompt, int memory,
            int trials, SplittableRandom random) {
        int truth = task.answer(prompt);
        ReflexionLoop.Actor<Integer> actor = (lessons, r) -> {
            int[][] batch = new int[16][];
            for (int i = 0; i < batch.length; i++) {
                batch[i] = prompt;
            }
            Integer fallback = null;
            for (int[] completion : lm.sample(batch, task.maxAnswerTokens(), task.eos(), 1.0, r)) {
                Integer v = task.parse(completion);
                if (fallback == null) {
                    fallback = v;
                }
                if (v != null && consistent(v, lessons)) {
                    return v;
                }
            }
            return fallback;
        };
        ReflexionLoop.Evaluator<Integer> checker = v -> v == null ? new ReflexionLoop.Feedback(false, "malformed")
                : v == truth ? new ReflexionLoop.Feedback(true, "right")
                : new ReflexionLoop.Feedback(false, v > truth ? "too high" : "too low");
        ReflexionLoop.Reflector<Integer> reflector = (v, f) -> v == null ? "answer with digits"
                : f.signal().equals("too high") ? "below " + v : "above " + v;
        return ReflexionLoop.of(actor, checker, reflector).memorySize(memory).run(trials, random);
    }

    private static boolean consistent(int v, List<String> lessons) {
        for (String lesson : lessons) {
            String[] w = lesson.split(" ");
            if (w[0].equals("below") && v >= Integer.parseInt(w[1])) {
                return false;
            }
            if (w[0].equals("above") && v <= Integer.parseInt(w[1])) {
                return false;
            }
        }
        return true;
    }
}
