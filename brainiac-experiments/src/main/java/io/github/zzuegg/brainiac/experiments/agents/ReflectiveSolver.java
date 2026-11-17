package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.llm.MemoryStream;
import io.github.zzuegg.brainiac.experimental.llm.ReflectiveMemoryAgent;
import io.github.zzuegg.brainiac.experimental.llm.ReflexionLoop;
import io.github.zzuegg.brainiac.experimental.llm.VirtualContextPager;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Reflexion with a tiny language model on multiplication: a half-trained model guesses, a
 * checker says "too high" or "too low", the reflector writes that down as a lesson ("not 56
 * or more"), and the next attempt draws samples from the model until one agrees with every
 * lesson. Plain retrying, with the same number of trials and the same feedback but no memory,
 * repeats its mistakes.
 *
 * <p>{@link #lifelong} runs a long sequence of such tasks, where each product recurs and
 * appears in both operand orders, and compares Reflexion alone (lessons forgotten after each
 * task) with a {@link ReflectiveMemoryAgent}, which writes its lessons and successes to a
 * {@link MemoryStream}, retrieves them by relevance for the next task, and pages them into a
 * {@link VirtualContextPager} working context (item 5.10).
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

    /**
     * Solve rates over a long task sequence.
     *
     * @param reflexion share of tasks Reflexion alone solves within the budget
     * @param memoryAgent share the memory agent solves
     * @param reflexionFirstTry share Reflexion alone solves at the first trial
     * @param memoryFirstTry share the memory agent solves at the first trial
     * @param lastQuarterReflexion Reflexion's solve rate on the last quarter of the sequence
     * @param lastQuarterMemory the memory agent's solve rate on the last quarter
     */
    public record Lifelong(double reflexion, double memoryAgent, double reflexionFirstTry, double memoryFirstTry,
            double lastQuarterReflexion, double lastQuarterMemory) {
    }

    private static final ReflexionLoop.Reflector<Integer> REFLECTOR = (v, f) -> v == null ? "answer with digits"
            : f.signal().equals("too high") ? "below " + v : "above " + v;

    private ReflectiveSolver() {
    }

    private static WindowLm halfTrained(ArithmeticTask task, int sftSteps, long seed) {
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
        return lm;
    }

    /**
     * @param sftSteps supervised steps for the model (few, so it stays unreliable)
     * @param trials the trial budget
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int sftSteps, int trials, long seed) {
        ArithmeticTask task = ArithmeticTask.multiplication(9);
        WindowLm lm = halfTrained(task, sftSteps, seed);
        List<int[]> prompts = task.allPrompts();
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

    /**
     * @param sftSteps supervised steps for the model
     * @param tasks how many tasks in the sequence
     * @param trials the trial budget per task
     * @param seed the seed
     * @return the solve rates
     */
    public static Lifelong lifelong(int sftSteps, int tasks, int trials, long seed) {
        ArithmeticTask task = ArithmeticTask.multiplication(9);
        WindowLm lm = halfTrained(task, sftSteps, seed);
        SplittableRandom order = new SplittableRandom(seed + 1);
        MemoryStream store = MemoryStream.of(MemoryStream.hashedBagOfWords(64), 64).decay(0.999);
        VirtualContextPager context = VirtualContextPager.of(200, s -> s.split(" ").length,
                MemoryStream.hashedBagOfWords(64), 64);
        int plainSolved = 0;
        int plainFirst = 0;
        int memorySolved = 0;
        int memoryFirst = 0;
        int plainLast = 0;
        int memoryLast = 0;
        SplittableRandom random = new SplittableRandom(seed);
        for (int i = 0; i < tasks; i++) {
            int a = 2 + order.nextInt(8);
            int b = 2 + order.nextInt(8);
            int[] prompt = task.prompt(a, b);
            boolean last = i >= tasks - tasks / 4;
            ReflexionLoop.Outcome<Integer> plain = solve(lm, task, prompt, trials, trials, random);
            plainSolved += plain.success() ? 1 : 0;
            plainFirst += plain.success() && plain.trials() == 1 ? 1 : 0;
            plainLast += last && plain.success() ? 1 : 0;
            ReflectiveMemoryAgent<Integer> agent = ReflectiveMemoryAgent.of(
                    actor(lm, task, prompt, lessons -> relevantLessons(lessons, a, b)), store, context);
            ReflexionLoop.Outcome<Integer> mem = agent.solve("multiply " + a + " by " + b,
                    checker(task.answer(prompt)), REFLECTOR, trials, random).outcome();
            memorySolved += mem.success() ? 1 : 0;
            memoryFirst += mem.success() && mem.trials() == 1 ? 1 : 0;
            memoryLast += last && mem.success() ? 1 : 0;
        }
        double n = tasks;
        double q = tasks / 4;
        return new Lifelong(plainSolved / n, memorySolved / n, plainFirst / n, memoryFirst / n, plainLast / q,
                memoryLast / q);
    }

    /**
     * Keeps the retrieved lessons about this product, in either operand order, without their
     * task prefix (the actor's reading of its context); a success note becomes {@code is v}.
     */
    private static List<String> relevantLessons(List<String> lessons, int a, int b) {
        List<String> out = new ArrayList<>();
        for (String l : lessons) {
            int bar = l.indexOf(ReflectiveMemoryAgent.SEPARATOR);
            String[] w = l.substring(0, bar).split(" ");
            int x = Integer.parseInt(w[1]);
            int y = Integer.parseInt(w[3]);
            if ((x == a && y == b) || (x == b && y == a)) {
                String lesson = l.substring(bar + ReflectiveMemoryAgent.SEPARATOR.length());
                out.add(lesson.startsWith("solved by ") ? "is " + lesson.substring("solved by ".length()) : lesson);
            }
        }
        return out;
    }

    private static ReflexionLoop.Evaluator<Integer> checker(int truth) {
        return v -> v == null ? new ReflexionLoop.Feedback(false, "malformed")
                : v == truth ? new ReflexionLoop.Feedback(true, "right")
                : new ReflexionLoop.Feedback(false, v > truth ? "too high" : "too low");
    }

    private static ReflexionLoop.Actor<Integer> actor(WindowLm lm, ArithmeticTask task, int[] prompt,
            UnaryOperator<List<String>> read) {
        return (memory, r) -> {
            List<String> lessons = read.apply(memory);
            for (String l : lessons) {
                if (l.startsWith("is ")) {
                    return Integer.parseInt(l.substring(3));   // a remembered answer
                }
            }
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
    }

    private static ReflexionLoop.Outcome<Integer> solve(WindowLm lm, ArithmeticTask task, int[] prompt, int memory,
            int trials, SplittableRandom random) {
        return ReflexionLoop.of(actor(lm, task, prompt, lessons -> lessons), checker(task.answer(prompt)), REFLECTOR)
                .memorySize(memory).run(trials, random);
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
