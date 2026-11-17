package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiPredicate;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * STaR, the self-taught reasoner (Zelikman et al. 2022,
 * <a href="https://arxiv.org/abs/2203.14465">2203.14465</a>): bootstrap reasoning from a
 * handful of worked examples and a verifier of final answers. Each outer iteration
 * <ol>
 *   <li>samples a rationale and answer for every problem and keeps those whose answer is
 *       right;</li>
 *   <li><b>rationalizes</b> the failures: the problem is shown again with the right answer
 *       as a hint, and a rationale that reaches it is kept <em>without the hint</em>, as if
 *       the model had found it unaided;</li>
 *   <li>fine-tunes the <em>original</em> model on everything kept (plus the seed examples),
 *       which avoids drifting by training a model on its own outputs again and again.</li>
 * </ol>
 * Without step 2 it is rejection-sampling fine-tuning (RFT / ReST-EM): the problems the model
 * never solves contribute nothing, so it plateaus on what it can already do; rationalization
 * adds training signal exactly where the model fails.
 *
 * <pre>{@code
 * StarTrainer star = StarTrainer.of(lm, task::correct, task::hinted)
 *         .generation(maxTokens, eos, 1.0).samples(4).finetune(60, () -> Optimizer.adam(3e-3f));
 * for (int i = 0; i < 3; i++) {
 *     star.iterate(problems, random);
 * }
 * }</pre>
 */
public final class StarTrainer {

    /**
     * What one iteration did.
     *
     * @param solved problems whose own sample was right
     * @param rationalized failed problems recovered through a hint
     * @param kept training sequences after filtering (excluding the seed examples)
     */
    public record Round(int solved, int rationalized, int kept) {
    }

    private final WindowLm model;
    private final WindowLm initial;
    private final BiPredicate<int[], int[]> correct;
    private final UnaryOperator<int[]> hint;
    private boolean rationalize = true;
    private int samples = 4;
    private int maxTokens = 4;
    private int end = -1;
    private double temperature = 1;
    private int steps = 60;
    private Supplier<Optimizer> optimizer = () -> Optimizer.adam(3e-3f);
    private final List<int[]> seedPrompts = new ArrayList<>();
    private final List<int[]> seedCompletions = new ArrayList<>();

    private StarTrainer(WindowLm model, BiPredicate<int[], int[]> correct, UnaryOperator<int[]> hint) {
        this.model = Objects.requireNonNull(model);
        this.initial = model.copy();
        this.correct = Objects.requireNonNull(correct);
        this.hint = Objects.requireNonNull(hint);
    }

    /**
     * @param model the model; its current weights are the "original" model every
     *     iteration restarts from
     * @param correct whether a completion answers a problem right
     * @param hint the problem with its right answer given as a hint
     * @return the trainer
     */
    public static StarTrainer of(WindowLm model, BiPredicate<int[], int[]> correct, UnaryOperator<int[]> hint) {
        return new StarTrainer(model, correct, hint);
    }

    /**
     * @param on whether to rationalize failures (off gives rejection-sampling fine-tuning)
     * @return this trainer
     */
    public StarTrainer rationalization(boolean on) {
        this.rationalize = on;
        return this;
    }

    /**
     * @param n attempts per problem (and per hinted problem)
     * @return this trainer
     */
    public StarTrainer samples(int n) {
        this.samples = n;
        return this;
    }

    /**
     * @param maxNew the most tokens per rationale and answer
     * @param endToken the end token
     * @param t the sampling temperature
     * @return this trainer
     */
    public StarTrainer generation(int maxNew, int endToken, double t) {
        this.maxTokens = maxNew;
        this.end = endToken;
        this.temperature = t;
        return this;
    }

    /**
     * @param n full-batch fine-tuning steps per iteration
     * @param o a fresh optimizer for each iteration
     * @return this trainer
     */
    public StarTrainer finetune(int n, Supplier<Optimizer> o) {
        this.steps = n;
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * Adds worked examples that are part of every fine-tuning set: STaR's few-shot prompt.
     *
     * @param prompts the prompts (plain or hinted)
     * @param completions their rationales and answers
     * @return this trainer
     */
    public StarTrainer seed(List<int[]> prompts, List<int[]> completions) {
        seedPrompts.addAll(prompts);
        seedCompletions.addAll(completions);
        return this;
    }

    private int[] firstCorrect(int[] problem, int[] shown, SplittableRandom random) {
        int[][] group = new int[samples][];
        Arrays.fill(group, shown);
        for (int[] c : model.sample(group, maxTokens, end, temperature, random)) {
            if (c.length > 0 && correct.test(problem, c)) {
                return c;
            }
        }
        return null;
    }

    /**
     * One outer iteration: generate, rationalize, restart from the original model and
     * fine-tune.
     *
     * @param problems the problems (only their answers are checked, no rationales needed)
     * @param random the random source
     * @return what happened
     */
    public Round iterate(List<int[]> problems, SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call generation(maxNew, end, temperature) first");
        }
        List<int[]> ps = new ArrayList<>(seedPrompts);
        List<int[]> cs = new ArrayList<>(seedCompletions);
        int solved = 0;
        int rationalized = 0;
        for (int[] p : problems) {
            int[] c = firstCorrect(p, p, random);
            if (c != null) {
                solved++;
            } else if (rationalize) {
                c = firstCorrect(p, hint.apply(p), random);
                if (c != null) {
                    rationalized++;
                }
            }
            if (c != null) {
                ps.add(p);   // kept without the hint
                cs.add(c);
            }
        }
        model.load(initial);
        Optimizer o = optimizer.get();
        for (int s = 0; s < steps; s++) {
            model.supervisedStep(ps, cs, o);
        }
        return new Round(solved, rationalized, solved + rationalized);
    }
}
