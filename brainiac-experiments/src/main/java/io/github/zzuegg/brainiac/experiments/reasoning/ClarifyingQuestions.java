package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.MultiturnRewardTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ClarificationDialogue.Intent;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * CollabLLM on a toy: does a model learn to ask for a missing operand? A {@link WindowLm}
 * is fine-tuned like an instruction-tuned model that can ask but rarely does: on fully
 * specified sums ({@code a+b=} then the answer), on finished clarification transcripts
 * ({@code a+?=?b=} then the answer), and on underspecified openings that it mostly answers
 * with a guess (a question for one in five of them). It is then
 * trained with GRPO in {@link ClarificationDialogue} conversations where half the requests
 * hide {@code b}, either with the single-turn reward (a response is worth its outcome only if
 * it ends the conversation) or with CollabLLM's multiturn-aware reward
 * ({@link MultiturnRewardTrainer}: the mean outcome of simulated futures minus 0.1 per
 * assistant turn). Candidates are sampled at temperature 2 during training, because the
 * supervised model rarely asks. Task success and assistant turns are measured on every
 * request, greedily.
 */
public final class ClarifyingQuestions {

    /** How the assistant was trained. */
    public enum Training {
        /** Supervised only. */
        SFT,
        /** Then GRPO with single-turn rewards. */
        SINGLE_TURN_GRPO,
        /** Then GRPO with CollabLLM's multiturn-aware rewards. */
        COLLABLLM
    }

    /**
     * @param training how the model was trained
     * @param success task success over all requests (half underspecified)
     * @param specifiedSuccess success on the fully specified requests
     * @param underspecifiedSuccess success on the underspecified ones
     * @param turns mean assistant turns per conversation
     * @param askRate the share of underspecified requests whose first response is a question
     * @param needlessAskRate the share of fully specified requests answered with a question
     */
    public record Result(Training training, double success, double specifiedSuccess, double underspecifiedSuccess,
            double turns, double askRate, double needlessAskRate) {
    }

    private ClarifyingQuestions() {
    }

    /**
     * The supervised starting point.
     *
     * @param dialogue the dialogue
     * @param steps full-batch supervised steps
     * @param askShare the share of underspecified openings whose target is a question (the
     *     rest get a guessed answer)
     * @param seed the seed
     * @return the model
     */
    public static WindowLm sft(ClarificationDialogue dialogue, int steps, double askShare, long seed) {
        WindowLm lm = WindowLm.of(dialogue.vocabSize(), 8, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 3);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (Intent i : dialogue.all(true)) {
            ps.add(dialogue.opening(i));
            cs.add(dialogue.answer(i));
        }
        List<Intent> hidden = dialogue.all(false);
        for (Intent i : hidden) {
            int[] asked = concat(dialogue.opening(i), dialogue.question());
            ps.add(concat(asked, dialogue.reveal(i)));
            cs.add(dialogue.answer(i));
            ps.add(dialogue.opening(i));
            cs.add(r.nextDouble() < askShare ? dialogue.question()
                    : dialogue.answer(new Intent(i.a(), r.nextInt(dialogue.maxOperand() + 1), true)));
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * Fine-tunes a copy of the supervised model with GRPO in collaborative conversations.
     *
     * @param dialogue the dialogue
     * @param sft the supervised model (not changed)
     * @param mode single-turn or multiturn-aware rewards
     * @param steps training steps (16 conversations each)
     * @param seed the seed
     * @return the trained model
     */
    public static WindowLm train(ClarificationDialogue dialogue, WindowLm sft, MultiturnRewardTrainer.Mode mode,
            int steps, long seed) {
        WindowLm lm = sft.copy();
        GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(Optimizer.adam(3e-3f)).innerEpochs(2).klCoefficient(0.01f)
                .completions(dialogue.maxResponseTokens(), dialogue.stops()[0]);
        ClarificationDialogue user = dialogue.withMode(ClarificationDialogue.Mode.COLLABORATIVE);
        MultiturnRewardTrainer<Intent> trainer = MultiturnRewardTrainer.of(lm, grpo, user)
                .sampling(8, dialogue.maxResponseTokens(), dialogue.stops(), 2.0)
                .reward(mode, 4, 3, 0.1f);
        SplittableRandom random = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            List<Intent> intents = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                intents.add(user.sample(random, 0.5));
            }
            trainer.step(intents, random);
        }
        return lm;
    }

    /**
     * Greedy conversations on every request.
     *
     * @param dialogue the dialogue (its mode is used)
     * @param lm the assistant
     * @param training the label
     * @return the measurements
     */
    public static Result evaluate(ClarificationDialogue dialogue, WindowLm lm, Training training) {
        double[] success = new double[2];
        double turns = 0;
        double[] asks = new double[2];
        int n = 0;
        for (int spec = 0; spec < 2; spec++) {
            List<Intent> intents = dialogue.all(spec == 1);
            for (Intent i : intents) {
                MultiturnRewardTrainer.Conversation c = MultiturnRewardTrainer.converse(lm, dialogue, i,
                        dialogue.maxResponseTokens(), dialogue.stops(), 3, 0, new SplittableRandom(0));
                success[spec] += c.outcome() / intents.size();
                turns += c.turns();
                int first = dialogue.opening(i).length;
                asks[spec] += c.context().length > first && c.context()[first] == dialogue.question()[0]
                        ? 1.0 / intents.size() : 0;
                n++;
            }
        }
        return new Result(training, (success[0] + success[1]) / 2, success[1], success[0], turns / n, asks[0],
                asks[1]);
    }
}
