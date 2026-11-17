package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * SEAL, self-adapting language models (Zweiger et al. 2025,
 * <a href="https://arxiv.org/abs/2506.10943">2506.10943</a>): a model learns to write its own
 * fine-tuning data. Given a passage, the editor generates a <em>self-edit</em> (a rewrite,
 * implications, question-answer pairs); a copy of the knowledge model is fine-tuned on that
 * self-edit, and the edit is rewarded by how well the updated model then answers questions
 * about the passage. The editor is trained with ReST-EM, as in the paper: sample several
 * edits per passage, keep the best-rewarded ones, and fine-tune the editor on them.
 *
 * <p>The point is that the raw passage is often a poor training example for the questions
 * that will be asked (the reversal curse: training on "B is A's value" does not teach
 * "A's value is B"), while a learned rewrite into the question's form is a good one. The
 * paper uses one model as editor and student with LoRA updates; here the editor and the
 * knowledge model are separate {@link WindowLm}s and the update is a few full fine-tuning
 * steps on a fresh copy, which keeps the toy fast.
 *
 * <pre>{@code
 * SelfEditTrainer seal = SelfEditTrainer.of(editor, (passage, edit) -> recallAfter(edit))
 *         .edits(3, end, 16, 1.0).finetune(30, () -> Optimizer.adam(3e-3f));
 * seal.round(passages, random);
 * }</pre>
 */
public final class SelfEditTrainer {

    /** Scores a self-edit: update a copy of the knowledge model on it and test it. */
    @FunctionalInterface
    public interface Evaluator {
        /**
         * @param passage the passage the edit was written from
         * @param edit the self-edit
         * @return the reward, higher is better
         */
        double reward(int[] passage, int[] edit);
    }

    /**
     * What one ReST-EM round did.
     *
     * @param meanReward the mean reward of every sampled edit
     * @param keptReward the mean reward of the edits kept for training
     */
    public record Round(double meanReward, double keptReward) {
    }

    private final WindowLm editor;
    private final Evaluator evaluator;
    private int maxTokens = 3;
    private int end = -1;
    private int candidates = 8;
    private double temperature = 1;
    private int steps = 30;
    private Supplier<Optimizer> optimizer = () -> Optimizer.adam(3e-3f);

    private SelfEditTrainer(WindowLm editor, Evaluator evaluator) {
        this.editor = Objects.requireNonNull(editor);
        this.evaluator = Objects.requireNonNull(evaluator);
    }

    /**
     * @param editor the model that writes self-edits (prompted with the passage)
     * @param evaluator applies an edit to the knowledge model and measures it
     * @return the trainer
     */
    public static SelfEditTrainer of(WindowLm editor, Evaluator evaluator) {
        return new SelfEditTrainer(editor, evaluator);
    }

    /**
     * @param maxNew the most tokens per edit
     * @param endToken the token that ends an edit
     * @param n edits sampled per passage
     * @param t the sampling temperature
     * @return this trainer
     */
    public SelfEditTrainer edits(int maxNew, int endToken, int n, double t) {
        this.maxTokens = maxNew;
        this.end = endToken;
        this.candidates = n;
        this.temperature = t;
        return this;
    }

    /**
     * @param n editor fine-tuning steps per round
     * @param o a fresh optimizer per round
     * @return this trainer
     */
    public SelfEditTrainer finetune(int n, Supplier<Optimizer> o) {
        this.steps = n;
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * Fine-tunes a copy of a model on one training sequence: the inner update that a
     * self-edit drives.
     *
     * @param base the knowledge model (not changed)
     * @param sequence the training text, learned as a continuation of the empty prompt
     * @param steps optimizer steps
     * @param optimizer the optimizer
     * @return the updated copy
     */
    public static WindowLm update(WindowLm base, int[] sequence, int steps, Optimizer optimizer) {
        return update(base, List.of(sequence), steps, optimizer);
    }

    /**
     * Fine-tunes a copy of a model on several training sequences at once (several self-edits,
     * or a self-edit among other documents).
     *
     * @param base the knowledge model (not changed)
     * @param sequences the training texts, each learned as a continuation of the empty prompt
     * @param steps full-batch optimizer steps
     * @param optimizer the optimizer
     * @return the updated copy
     */
    public static WindowLm update(WindowLm base, List<int[]> sequences, int steps, Optimizer optimizer) {
        WindowLm copy = base.copy();
        List<int[]> p = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            p.add(new int[0]);
        }
        for (int s = 0; s < steps; s++) {
            copy.supervisedStep(p, sequences, optimizer);
        }
        return copy;
    }

    /**
     * The editor's greedy self-edit for a passage.
     *
     * @param prompt the passage prompt
     * @return the edit
     */
    public int[] write(int[] prompt) {
        return editor.greedy(prompt, maxTokens, end);
    }

    /**
     * One ReST-EM round: sample edits, score each by the updated model's performance, keep
     * the best edit of each passage when it beats the passage's mean, fine-tune the editor.
     *
     * @param prompts the passage prompts (the passage plus an edit cue)
     * @param random the random source
     * @return what happened
     */
    public Round round(List<int[]> prompts, SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call edits(maxNew, end, n, temperature) first");
        }
        List<int[]> keepPrompts = new ArrayList<>();
        List<int[]> keepEdits = new ArrayList<>();
        double sum = 0;
        double kept = 0;
        int count = 0;
        for (int[] prompt : prompts) {
            int[][] group = new int[candidates][];
            Arrays.fill(group, prompt);
            int[][] edits = editor.sample(group, maxTokens, end, temperature, random);
            double best = Double.NEGATIVE_INFINITY;
            double mean = 0;
            int[] bestEdit = null;
            for (int[] e : edits) {
                double r = evaluator.reward(prompt, e);
                mean += r;
                if (r > best) {
                    best = r;
                    bestEdit = e;
                }
            }
            mean /= candidates;
            sum += mean * candidates;
            count += candidates;
            if (bestEdit != null && bestEdit.length > 0 && best > mean) {
                keepPrompts.add(prompt);
                keepEdits.add(bestEdit);
                kept += best;
            }
        }
        if (!keepEdits.isEmpty()) {
            Optimizer o = optimizer.get();
            for (int s = 0; s < steps; s++) {
                editor.supervisedStep(keepPrompts, keepEdits, o);
            }
        }
        return new Round(sum / count, keepEdits.isEmpty() ? 0 : kept / keepEdits.size());
    }
}
