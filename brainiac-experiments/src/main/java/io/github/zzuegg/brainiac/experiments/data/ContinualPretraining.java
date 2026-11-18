package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ContinualPretrainer;
import io.github.zzuegg.brainiac.experimental.train.ElasticWeightConsolidation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Continual pretraining (9.3.9): a {@link ContextLm} pretrained on domain A (a Markov
 * {@link DataCuration.Grammar}, warm-up then cosine decay), continued on domain B (a different
 * grammar, 30% of whose documents are corrupted, as web data is), with frequent evaluation on
 * held-out clean A and B.
 * <ul>
 *   <li>{@link Method#CONSTANT}: continue at the final pretraining rate, no replay;</li>
 *   <li>{@link Method#REWARM}: re-warm and re-decay ({@link ContinualPretrainer#rewarmRedecay}),
 *       no replay;</li>
 *   <li>{@link Method#REWARM_REPLAY}: the same with 5% replay of A;</li>
 *   <li>{@link Method#REWARM_REPLAY_SUBSET}: the same, trained for several epochs on the half of
 *       B with the lowest perplexity under a short reference run on B
 *       ({@link ContinualPretrainer#lowPerplexitySubset});</li>
 *   <li>{@link Method#EWC}: re-warming plus an {@link ElasticWeightConsolidation} penalty
 *       towards the pretrained weights;</li>
 *   <li>{@link Method#RETRAIN}: from scratch on A ∪ B for the pretraining plus continual budget
 *       (the compute upper bound).</li>
 * </ul>
 * The stability gap is tracked on the held-out A loss, every 10 steps
 * ({@link ContinualPretrainer.GapTracker}): its peak rise above the value at the switch, and how
 * much of that rise has been recovered by the end.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ContinualPretraining {

    private ContinualPretraining() {
    }

    /** The continual-pretraining recipe. */
    public enum Method {
        /** Constant rate at the end of pretraining, no replay. */
        CONSTANT,
        /** Re-warm and re-decay, no replay. */
        REWARM,
        /** Re-warm, re-decay and 5% replay. */
        REWARM_REPLAY,
        /** Re-warm, re-decay, 5% replay, several epochs on a low-perplexity half of B. */
        REWARM_REPLAY_SUBSET,
        /** Re-warm and re-decay with an EWC penalty. */
        EWC,
        /** Retrain on A ∪ B from scratch. */
        RETRAIN
    }

    /**
     * The outcome of one recipe.
     *
     * @param method the recipe
     * @param lossA final held-out loss on A
     * @param lossB final held-out loss on B
     * @param gapDepth the largest rise of the A loss above its value at the switch to B
     * @param gapRecovery how much of that rise is gone at the end (peak minus final A loss)
     * @param gapLength steps until the A loss is back at its start (-1 never, 0 no rise)
     * @param steps training steps spent (pretraining plus continual)
     */
    public record Run(Method method, double lossA, double lossB, double gapDepth, double gapRecovery, long gapLength,
            int steps) {
        @Override
        public String toString() {
            return String.format("%s: A %.4f, B %.4f, gap depth %.4f, recovered %.4f, length %d, %d steps", method,
                    lossA, lossB, gapDepth, gapRecovery, gapLength, steps);
        }
    }

    private static final int VOCAB = 16;
    private static final int BATCH = 32;

    private static void step(ContextLm lm, Optimizer opt, List<int[]> a, List<int[]> b, int fromA,
            SplittableRandom r, Supplier<Var> penalty) {
        List<int[]> docs = new ArrayList<>();
        for (int i = 0; i < BATCH; i++) {
            List<int[]> src = i < fromA ? a : b;
            docs.add(src.get(r.nextInt(src.size())));
        }
        Tensor[] e = ContextLm.examples(VOCAB, docs);
        int n = e[1].shape().dim(0);
        int[] rows = new int[BATCH];
        for (int i = 0; i < BATCH; i++) {
            rows[i] = r.nextInt(n);
        }
        Var loss = Functional.crossEntropy(lm.net().forward(Var.of(ContextLm.rows(e[0], rows))),
                ContextLm.rows(e[1], rows));
        if (penalty != null) {
            loss = loss.add(penalty.get());
        }
        loss.backward();
        opt.step(lm.net().parameters());
    }

    /**
     * Runs one recipe.
     *
     * @param method the recipe
     * @param pretrainSteps steps on A
     * @param continualSteps steps on B
     * @param seed the seed
     * @return the outcome
     */
    public static Run run(Method method, int pretrainSteps, int continualSteps, long seed) {
        DataCuration.Grammar ga = DataCuration.Grammar.random(VOCAB, seed);
        DataCuration.Grammar gb = DataCuration.Grammar.random(VOCAB, seed + 100);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> a = ga.documents(300, 16, r);
        List<int[]> b = new ArrayList<>();
        for (int[] d : gb.documents(300, 16, r)) {
            b.add(r.nextDouble() < 0.3 ? DataCuration.corrupt(d, 0.6, VOCAB, r) : d);
        }
        List<int[]> heldA = ga.documents(60, 16, r);
        List<int[]> heldB = gb.documents(60, 16, r);
        float peak = 1e-2f;
        float min = 1e-3f;
        ContextLm lm = new ContextLm(VOCAB, 64, seed + 2);
        if (method == Method.RETRAIN) {
            int total = pretrainSteps + continualSteps;
            Optimizer opt = Optimizer.adam(peak);
            LearningRateSchedule sched = LearningRateSchedule.warmup(total / 20, LearningRateSchedule.cosine(total
                    - total / 20, min), true);
            for (int s = 1; s <= total; s++) {
                opt.learningRate(sched.rate(s, peak));
                step(lm, opt, a, b, BATCH / 2, r, null);
            }
            return new Run(method, lm.loss(heldA), lm.loss(heldB), 0, 0, 0, total);
        }
        Optimizer opt = Optimizer.adam(peak);
        LearningRateSchedule pre = LearningRateSchedule.warmup(pretrainSteps / 20, LearningRateSchedule.cosine(
                pretrainSteps - pretrainSteps / 20, min), true);
        for (int s = 1; s <= pretrainSteps; s++) {
            opt.learningRate(pre.rate(s, peak));
            step(lm, opt, a, b, BATCH, r, null);
        }
        List<int[]> newData = b;
        if (method == Method.REWARM_REPLAY_SUBSET) {
            ContextLm reference = new ContextLm(VOCAB, 32, seed + 3).train(b, 150, BATCH, 1e-2f, seed + 4);
            double[] losses = new double[b.size()];
            for (int i = 0; i < losses.length; i++) {
                losses[i] = reference.loss(List.of(b.get(i)));
            }
            newData = new ArrayList<>();
            for (int i : ContinualPretrainer.lowPerplexitySubset(losses, 0.5)) {
                newData.add(b.get(i));
            }
        }
        ElasticWeightConsolidation ewc = null;
        if (method == Method.EWC) {
            ewc = new ElasticWeightConsolidation(lm.net().parameters(), 50f);
            List<Supplier<Var>> batches = new ArrayList<>();
            for (int k = 0; k < 8; k++) {
                List<int[]> docs = a.subList(k * 30, k * 30 + 30);
                batches.add(() -> {
                    Tensor[] e = ContextLm.examples(VOCAB, docs);
                    return Functional.crossEntropy(lm.net().forward(Var.of(e[0])), e[1]);
                });
            }
            ewc.consolidate(batches);
        }
        LearningRateSchedule cont = method == Method.CONSTANT ? (s, base) -> min
                : ContinualPretrainer.rewarmRedecay(continualSteps / 10, continualSteps, min);
        double replay = method == Method.REWARM_REPLAY || method == Method.REWARM_REPLAY_SUBSET ? 0.05 : 0;
        ContinualPretrainer.GapTracker gap = new ContinualPretrainer.GapTracker();
        gap.record(0, lm.loss(heldA));
        ElasticWeightConsolidation penalty = ewc;
        for (int s = 1; s <= continualSteps; s++) {
            opt.learningRate(cont.rate(s, peak));
            int fromA = ContinualPretrainer.replayCount(BATCH, replay, r);
            step(lm, opt, a, newData, fromA, r, penalty == null ? null : penalty::penalty);
            if (s % 10 == 0) {
                gap.record(s, lm.loss(heldA));
            }
        }
        return new Run(method, lm.loss(heldA), lm.loss(heldB), gap.depth(), gap.start() + gap.depth() - gap.last(),
                gap.length(),
                pretrainSteps + continualSteps);
    }
}
