package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Preference fine-tuning of a {@link Llama} on pairs of a preferred and a rejected answer to
 * the same prompt, as TRL's {@code DPOTrainer}: the model learns to raise the chosen
 * answer's likelihood relative to a frozen reference (usually the model before tuning) more
 * than the rejected one's. Works with full fine-tuning or with {@link Llama#addLora LoRA}.
 *
 * <pre>
 * List&lt;PreferenceTraining.Pair&gt; pairs = ...;   // PreferenceTraining.Pair.of(tokenizer, prompt, good, bad)
 * float[][] ref = PreferenceTraining.reference(model, pairs, 4);   // before any update
 * for (int from = 0; from &lt; pairs.size(); from += 4) {
 *     List&lt;PreferenceTraining.Pair&gt; batch = pairs.subList(from, Math.min(pairs.size(), from + 4));
 *     PreferenceTraining.Step s = PreferenceTraining.loss(model, batch, Arrays.copyOfRange(ref, from, from + batch.size()),
 *             PreferenceTraining.Loss.DPO, 0.1f);
 *     s.loss().backward();
 *     optimizer.step(); optimizer.zeroGrad();
 * }
 * </pre>
 */
public final class PreferenceTraining {

    private PreferenceTraining() {
    }

    /**
     * The objective, of {@code h = (log pi(chosen) - log ref(chosen)) - (log pi(rejected) -
     * log ref(rejected))}.
     */
    public enum Loss {
        /** DPO, Rafailov et al. 2023: {@code -log sigmoid(beta h)}. */
        DPO,
        /** IPO, Azar et al. 2023: {@code (h - 1 / (2 beta))^2}, which does not overfit to deterministic preferences. */
        IPO,
        /** SLiC's hinge, Zhao et al. 2023: {@code max(0, 1 - beta h)}. */
        HINGE
    }

    /**
     * A prompt with a preferred and a rejected answer: each the whole text, with only the
     * answer's tokens marked for training (see {@link InstructionData}).
     *
     * @param chosen the preferred example
     * @param rejected the rejected example
     */
    public record Pair(InstructionData.Example chosen, InstructionData.Example rejected) {

        /**
         * Tokenizes a pair the way {@link InstructionData#completion} does.
         *
         * @param tokenizer the tokenizer
         * @param prompt the prompt, already in the chat template if the model needs one
         * @param chosen the preferred answer
         * @param rejected the rejected answer
         * @return the pair
         */
        public static Pair of(PretrainedTokenizer tokenizer, String prompt, String chosen, String rejected) {
            return new Pair(InstructionData.completion(tokenizer, prompt, chosen),
                    InstructionData.completion(tokenizer, prompt, rejected));
        }
    }

    /**
     * One batch's loss and the implicit rewards {@code beta (log pi - log ref)} it saw.
     *
     * @param loss the mean loss over the pairs, to call {@code backward} on
     * @param chosenRewards per pair, the chosen answer's reward
     * @param rejectedRewards per pair, the rejected answer's reward
     */
    public record Step(Var loss, float[] chosenRewards, float[] rejectedRewards) {

        /** @return the fraction of pairs whose chosen answer has the higher reward */
        public double accuracy() {
            int n = 0;
            for (int i = 0; i < chosenRewards.length; i++) {
                n += chosenRewards[i] > rejectedRewards[i] ? 1 : 0;
            }
            return chosenRewards.length == 0 ? 0 : (double) n / chosenRewards.length;
        }

        /** @return the mean of chosen minus rejected reward */
        public double margin() {
            double m = 0;
            for (int i = 0; i < chosenRewards.length; i++) {
                m += chosenRewards[i] - rejectedRewards[i];
            }
            return chosenRewards.length == 0 ? 0 : m / chosenRewards.length;
        }
    }

    /**
     * The reference log-likelihoods, to compute once before training changes the model.
     *
     * @param model the model as it is now
     * @param pairs the pairs
     * @param batch pairs per forward pass
     * @return per pair, {@code {log ref(chosen), log ref(rejected)}}
     */
    public static float[][] reference(Llama model, List<Pair> pairs, int batch) {
        float[][] out = new float[pairs.size()][];
        for (int from = 0; from < pairs.size(); from += batch) {
            List<Pair> part = pairs.subList(from, Math.min(pairs.size(), from + batch));
            float[] lp = Var.noGrad(() -> model.sequenceLogProbs(rows(part)).value().toArray());
            for (int i = 0; i < part.size(); i++) {
                out[from + i] = new float[] {lp[i], lp[part.size() + i]};
            }
        }
        return out;
    }

    /**
     * The loss of a batch of pairs.
     *
     * @param model the model being tuned
     * @param pairs the pairs
     * @param reference the pairs' rows of {@link #reference}
     * @param loss the objective
     * @param beta how far the model may move from the reference: smaller lets it move further; 0.1 is usual
     * @return the loss and rewards
     */
    public static Step loss(Llama model, List<Pair> pairs, float[][] reference, Loss loss, float beta) {
        int n = pairs.size();
        if (reference.length != n) {
            throw new IllegalArgumentException(n + " pairs but " + reference.length + " reference rows");
        }
        Var lp = model.sequenceLogProbs(rows(pairs));
        Var chosen = lp.slice(0, 0, n);
        Var rejected = lp.slice(0, n, 2 * n);
        float[] refChosen = new float[n];
        float[] refRejected = new float[n];
        float[] refDiff = new float[n];
        for (int i = 0; i < n; i++) {
            refChosen[i] = reference[i][0];
            refRejected[i] = reference[i][1];
            refDiff[i] = refChosen[i] - refRejected[i];
        }
        Tensor ref = Tensor.of(Shape.of(n), refDiff).to(lp.value().device()).to(lp.value().dtype());
        Var h = chosen.sub(rejected).sub(ref);
        Var l = switch (loss) {
            case DPO -> h.mul(-beta).softplus();
            case IPO -> h.sub(1 / (2 * beta)).square();
            case HINGE -> h.mul(-beta).add(1f).relu();
        };
        float[] c = chosen.value().toArray();
        float[] r = rejected.value().toArray();
        float[] cr = new float[n];
        float[] rr = new float[n];
        for (int i = 0; i < n; i++) {
            cr[i] = beta * (c[i] - refChosen[i]);
            rr[i] = beta * (r[i] - refRejected[i]);
        }
        return new Step(l.mean(), cr, rr);
    }

    /** The chosen examples, then the rejected ones, padded into one batch. */
    private static InstructionData.Batch rows(List<Pair> pairs) {
        List<InstructionData.Example> all = new ArrayList<>();
        int longest = 2;
        for (Pair p : pairs) {
            all.add(p.chosen);
            longest = Math.max(longest, p.chosen.tokens().length);
        }
        for (Pair p : pairs) {
            all.add(p.rejected);
            longest = Math.max(longest, p.rejected.tokens().length);
        }
        return InstructionData.pad(all, all.size(), longest).getFirst();
    }
}
