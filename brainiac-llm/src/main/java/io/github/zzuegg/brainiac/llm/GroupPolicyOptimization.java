package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reinforcement learning with verifiable rewards on a {@link Llama}, as GRPO (Shao et al.
 * 2024, DeepSeekMath) does it: each prompt is answered several times by sampling, a reward
 * function scores the answers (a unit test, an exact-match check, a format check), and each
 * answer's advantage is its reward relative to its group's. The loss raises the likelihood of
 * answers that beat their group and lowers the rest. No value model and no reward model.
 *
 * <p>One {@link #step} is one on-policy update: the samples come from the model being
 * trained, so the probability ratio of PPO's clipped objective is 1 and the gradient is
 * {@code -advantage * grad log p(answer)}, averaged per token of each answer.
 *
 * <pre>
 * GroupPolicyOptimization grpo = new GroupPolicyOptimization(generator, 8, Generator.Options.sampled(256,
 *         Sampling.temperature(1f), seed), (prompt, answer) -&gt; answer.contains("#### 42") ? 1 : 0);
 * for (List&lt;String&gt; prompts : batches) {
 *     GroupPolicyOptimization.Step s = grpo.step(model, prompts);
 *     if (!s.loss().requiresGrad()) continue;   // all answers scored alike
 *     s.loss().backward();
 *     optimizer.step(); optimizer.zeroGrad();
 * }
 * </pre>
 */
public final class GroupPolicyOptimization {

    /** Scores an answer to a prompt; higher is better. */
    @FunctionalInterface
    public interface Reward {

        /**
         * @param prompt the prompt
         * @param answer the generated answer
         * @return the reward
         */
        double score(String prompt, String answer);
    }

    /**
     * One update's loss and what it saw.
     *
     * @param loss the loss to call {@code backward} on; a constant zero, which does not
     *     {@linkplain Var#requiresGrad() require gradients}, when every group's rewards were equal
     * @param rewards per prompt, the rewards of its answers
     * @param answers per prompt, the answers
     */
    public record Step(Var loss, double[][] rewards, List<List<String>> answers) {

        /** @return the mean reward over all answers */
        public double meanReward() {
            return Arrays.stream(rewards).flatMapToDouble(Arrays::stream).average().orElse(0);
        }
    }

    private final Generator generator;
    private final int group;
    private final Generator.Options options;
    private final Reward reward;
    private long draws;

    /**
     * @param generator the model being trained, with its tokenizer
     * @param group answers sampled per prompt, at least 2
     * @param options how answers are sampled: a temperature near 1 keeps the group diverse
     * @param reward the verifier
     */
    public GroupPolicyOptimization(Generator generator, int group, Generator.Options options, Reward reward) {
        if (group < 2) {
            throw new IllegalArgumentException("a group needs at least 2 answers, got " + group);
        }
        this.generator = generator;
        this.group = group;
        this.options = options;
        this.reward = reward;
    }

    /**
     * Samples, scores and builds the loss for a batch of prompts.
     *
     * @param model the model the generator runs, which the loss differentiates
     * @param prompts the prompts, already in the chat template if the model needs one
     * @return the loss and the rewards
     */
    public Step step(Llama model, List<String> prompts) {
        List<String> repeated = new ArrayList<>();
        for (String p : prompts) {
            for (int g = 0; g < group; g++) {
                repeated.add(p);
            }
        }
        // fresh draws each step
        Generator.Options o = new Generator.Options(options.maxNewTokens(), options.sampling(),
                options.seed() + draws++ * 1_000_003L, options.stopTokens(), options.stopStrings(), options.constraint());
        List<Generator.Output> outs = generator.generate(repeated, o, null);
        double[][] rewards = new double[prompts.size()][group];
        List<List<String>> answers = new ArrayList<>();
        List<InstructionData.Example> examples = new ArrayList<>();
        List<Float> weights = new ArrayList<>();
        for (int p = 0; p < prompts.size(); p++) {
            List<String> texts = new ArrayList<>();
            for (int g = 0; g < group; g++) {
                Generator.Output out = outs.get(p * group + g);
                texts.add(out.text());
                rewards[p][g] = reward.score(prompts.get(p), out.text());
            }
            answers.add(texts);
            double mean = Arrays.stream(rewards[p]).average().orElse(0);
            double var = Arrays.stream(rewards[p]).map(r -> (r - mean) * (r - mean)).sum() / (group - 1);
            double std = Math.sqrt(var);
            if (std < 1e-8) {
                continue;   // all equal: nothing to learn from this group
            }
            int[] prompt = generator.encode(prompts.get(p));
            for (int g = 0; g < group; g++) {
                int[] answer = outs.get(p * group + g).tokens();
                if (answer.length == 0) {
                    continue;
                }
                int[] tokens = Arrays.copyOf(prompt, prompt.length + answer.length);
                System.arraycopy(answer, 0, tokens, prompt.length, answer.length);
                boolean[] train = new boolean[tokens.length];
                Arrays.fill(train, prompt.length, tokens.length, true);
                examples.add(new InstructionData.Example(tokens, train));
                // the advantage, spread over the answer's tokens
                weights.add((float) ((rewards[p][g] - mean) / std / answer.length));
            }
        }
        if (examples.isEmpty()) {
            return new Step(Var.of(Tensor.scalar(0f)), rewards, answers);
        }
        int longest = examples.stream().mapToInt(e -> e.tokens().length).max().orElse(2);
        InstructionData.Batch batch = InstructionData.pad(examples, examples.size(), Math.max(2, longest)).getFirst();
        Var logp = model.sequenceLogProbs(batch);
        float[] w = new float[weights.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = -weights.get(i) / w.length;
        }
        Tensor coef = Tensor.of(Shape.of(w.length), w).to(logp.value().device()).to(logp.value().dtype());
        return new Step(logp.mul(coef).sum(), rewards, answers);
    }
}
