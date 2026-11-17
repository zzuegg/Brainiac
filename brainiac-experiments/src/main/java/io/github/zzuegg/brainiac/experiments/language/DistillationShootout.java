package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.OnPolicyDistiller;
import io.github.zzuegg.brainiac.experimental.llm.SequenceDistillation;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.nn.DistillationLoss;
import io.github.zzuegg.brainiac.experimental.nn.FeatureDistillationLoss;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticRl;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Language-model distillation side by side: a wide {@link WindowLm} teacher fine-tuned on
 * two-digit addition is distilled into a narrow student with each method, at the same number
 * of student optimizer steps, and the student's greedy accuracy, perplexity on the right
 * answers and the teacher token positions spent are compared.
 *
 * <ul>
 *   <li>{@code SFT}: no teacher, the gold answers only.</li>
 *   <li>{@code LOGIT_KD}: offline token-level KD with {@link DistillationLoss} on the gold
 *       prefixes (the baseline).</li>
 *   <li>{@code GKD}: on-policy, the student samples and the teacher scores with JSD(0.5)
 *       ({@link OnPolicyDistiller}).</li>
 *   <li>{@code MINILLM}: on-policy sequence-level reverse KL.</li>
 *   <li>{@code SEQ_KD}: cross-entropy on the teacher's greedy outputs
 *       ({@link SequenceDistillation}), built once.</li>
 *   <li>{@code FEATURE_KD}: logit KD plus matching the last hidden layer through a learned
 *       projection ({@link FeatureDistillationLoss}).</li>
 * </ul>
 */
public final class DistillationShootout {

    /** The methods compared. */
    public enum Method {
        /** supervised fine-tuning on the gold answers, no teacher */
        SFT,
        /** offline forward-KL logit distillation on gold prefixes */
        LOGIT_KD,
        /** on-policy generalized KD with the Jensen-Shannon divergence */
        GKD,
        /** on-policy reverse KL with MiniLLM's policy gradient */
        MINILLM,
        /** sequence-level KD on the teacher's greedy outputs */
        SEQ_KD,
        /** logit KD plus hidden-state matching */
        FEATURE_KD
    }

    /**
     * @param method the method
     * @param accuracy the student's greedy accuracy on every puzzle
     * @param perplexity the student's per-token perplexity on the right answers
     * @param teacherTokens token positions the teacher scored during training
     */
    public record Result(Method method, double accuracy, double perplexity, long teacherTokens) {
    }

    private DistillationShootout() {
    }

    private static List<int[]> gold(ArithmeticTask task, List<int[]> prompts) {
        List<int[]> out = new ArrayList<>();
        for (int[] p : prompts) {
            out.add(task.solution(p));
        }
        return out;
    }

    /**
     * Fine-tunes the teacher on every puzzle.
     *
     * @param task the puzzles
     * @param hidden the teacher's width
     * @param steps full-batch supervised steps
     * @param seed the seed
     * @return the teacher
     */
    public static WindowLm teacher(ArithmeticTask task, int hidden, int steps, long seed) {
        WindowLm t = WindowLm.of(task.vocabSize(), 6, hidden, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<int[]> prompts = task.allPrompts();
        List<int[]> gold = gold(task, prompts);
        for (int s = 0; s < steps; s++) {
            t.supervisedStep(prompts, gold, opt);
        }
        return t;
    }

    /**
     * @param lm a model
     * @param task the puzzles
     * @return {@code exp} of the mean negative log-likelihood of the right answer tokens
     */
    public static double perplexity(WindowLm lm, ArithmeticTask task) {
        List<int[]> prompts = task.allPrompts();
        return Math.exp(-lm.logProbs(prompts, gold(task, prompts)).mean().value().item());
    }

    /**
     * Distils {@code teacher} into a fresh student with one method.
     *
     * @param teacher the teacher (not changed)
     * @param task the puzzles
     * @param method the method
     * @param studentHidden the student's width
     * @param steps student optimizer steps
     * @param seed the seed
     * @return the result
     */
    public static Result run(WindowLm teacher, ArithmeticTask task, Method method, int studentHidden, int steps,
            long seed) {
        WindowLm student = WindowLm.of(task.vocabSize(), 6, studentHidden, seed);
        SplittableRandom random = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<int[]> all = task.allPrompts();
        int batch = 32;
        long teacherTokens = 0;
        OnPolicyDistiller distiller = null;
        if (method == Method.GKD || method == Method.MINILLM) {
            distiller = OnPolicyDistiller.of(student, teacher).optimizer(opt).onPolicyFraction(1)
                    .completions(task.maxAnswerTokens(), task.eos())
                    .divergence(method == Method.GKD ? OnPolicyDistiller.Divergence.JSD
                            : OnPolicyDistiller.Divergence.REVERSE_KL_SEQUENCE, 0.5f)
                    .policyGradientWeight(0.1f);
        }
        List<int[]> seqTargets = null;
        if (method == Method.SEQ_KD) {
            SequenceDistillation.Data data = SequenceDistillation.greedy(teacher, all, task.maxAnswerTokens(),
                    task.eos());
            seqTargets = data.targets();
            teacherTokens = data.teacherTokens();
        }
        FeatureDistillationLoss feature = method == Method.FEATURE_KD
                ? new FeatureDistillationLoss(studentHidden, teacher.hiddenSize(), seed) : null;
        DistillationLoss kd = new DistillationLoss(2f);
        for (int s = 0; s < steps; s++) {
            List<int[]> prompts = new ArrayList<>();
            List<int[]> targets = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                int k = random.nextInt(all.size());
                prompts.add(all.get(k));
                targets.add(seqTargets != null ? seqTargets.get(k) : task.solution(all.get(k)));
            }
            switch (method) {
                case SFT, SEQ_KD -> student.supervisedStep(prompts, targets, opt);
                case GKD, MINILLM -> distiller.step(prompts, null, random);
                case LOGIT_KD, FEATURE_KD -> {
                    List<int[]> prefixes = OnPolicyDistiller.prefixes(prompts, targets);
                    teacherTokens += prefixes.size();
                    Var[] t = teacher.logitsAndHidden(prefixes);
                    Var[] st = student.logitsAndHidden(prefixes);
                    Var loss = kd.compute(st[1], t[1].value());
                    List<Var> params = new ArrayList<>(student.parameters());
                    if (feature != null) {
                        loss = loss.add(feature.compute(st[0], t[0].value()));
                        params.addAll(feature.parameters());
                    }
                    loss.backward();
                    opt.step(params);
                }
            }
        }
        if (distiller != null) {
            teacherTokens = distiller.teacherTokens();
        }
        return new Result(method, ArithmeticRl.accuracy(student, task), perplexity(student, task), teacherTokens);
    }
}
