package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Self-Rewarding Language Models (<a href="https://arxiv.org/abs/2401.10020">2401.10020</a>):
 * one model is both the actor and its own judge ("LLM-as-a-judge"). Each iteration it
 * samples several answers per prompt, scores them itself, and trains with DPO on
 * (best, worst) pairs. Since judging is often easier than answering, the judge lifts the
 * actor.
 *
 * <p>The judge is the same network read with a judging prompt: {@code prompt ++ answer ++
 * [judgeToken]}, and the score is the probability of {@code yesToken} against
 * {@code noToken} as the next token. Seed judging ability with {@link #judgeExample}
 * pairs in ordinary supervised training.
 */
public final class SelfRewardingTrainer {

    private final WindowLm model;
    private final int judgeToken;
    private final int yesToken;
    private final int noToken;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int candidates = 6;
    private int stepsPerIteration = 20;
    private int maxNewTokens = 4;
    private int endToken;
    private float beta = 0.1f;
    private float nll;
    private double minGap = 0.2;
    private double minChosen = 0.5;

    private SelfRewardingTrainer(WindowLm model, int judgeToken, int yesToken, int noToken) {
        this.model = Objects.requireNonNull(model);
        this.judgeToken = judgeToken;
        this.yesToken = yesToken;
        this.noToken = noToken;
    }

    /**
     * @param model the model that answers and judges
     * @param judgeToken the token that asks "is this right?"
     * @param yesToken the token for a good answer
     * @param noToken the token for a bad answer
     * @return the trainer
     */
    public static SelfRewardingTrainer of(WindowLm model, int judgeToken, int yesToken, int noToken) {
        return new SelfRewardingTrainer(model, judgeToken, yesToken, noToken);
    }

    /**
     * @param o the optimizer
     * @return this trainer
     */
    public SelfRewardingTrainer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param n answers sampled per prompt
     * @return this trainer
     */
    public SelfRewardingTrainer candidates(int n) {
        this.candidates = n;
        return this;
    }

    /**
     * @param n DPO steps per iteration
     * @return this trainer
     */
    public SelfRewardingTrainer stepsPerIteration(int n) {
        this.stepsPerIteration = n;
        return this;
    }

    /**
     * @param maxTokens the most tokens per answer
     * @param end the end token
     * @return this trainer
     */
    public SelfRewardingTrainer generation(int maxTokens, int end) {
        this.maxNewTokens = maxTokens;
        this.endToken = end;
        return this;
    }

    /**
     * @param weight the weight of a likelihood term on the chosen answers, see
     *     {@link DpoTrainer#chosenNll(float)}
     * @return this trainer
     */
    public SelfRewardingTrainer chosenNll(float weight) {
        this.nll = weight;
        return this;
    }

    /**
     * @param beta the DPO inverse temperature
     * @return this trainer
     */
    public SelfRewardingTrainer beta(float beta) {
        this.beta = beta;
        return this;
    }

    /**
     * @param gap the smallest score difference between best and worst that makes a pair
     * @return this trainer
     */
    public SelfRewardingTrainer minScoreGap(double gap) {
        this.minGap = gap;
        return this;
    }

    /**
     * @param score the lowest self-judged score a chosen answer needs; pairs whose best
     *     answer the judge itself rejects are skipped, so that confident mistakes are not
     *     reinforced
     * @return this trainer
     */
    public SelfRewardingTrainer minChosenScore(double score) {
        this.minChosen = score;
        return this;
    }

    /**
     * @param prompt a prompt
     * @param answer an answer
     * @return the judging prompt
     */
    public int[] judgePrompt(int[] prompt, int[] answer) {
        int[] p = Arrays.copyOf(prompt, prompt.length + answer.length + 1);
        System.arraycopy(answer, 0, p, prompt.length, answer.length);
        p[p.length - 1] = judgeToken;
        return p;
    }

    /**
     * A supervised example that teaches judging.
     *
     * @param prompt a prompt
     * @param answer an answer
     * @param good whether the answer is good
     * @return {judging prompt, verdict completion}
     */
    public int[][] judgeExample(int[] prompt, int[] answer, boolean good) {
        return new int[][] {judgePrompt(prompt, answer), {good ? yesToken : noToken, endToken}};
    }

    /**
     * @param prompt a prompt
     * @param answer an answer
     * @return the model's own probability that the answer is good
     */
    public double judge(int[] prompt, int[] answer) {
        float[] logits = model.logits(judgePrompt(prompt, answer));
        double y = logits[yesToken];
        double n = logits[noToken];
        return 1 / (1 + Math.exp(n - y));
    }

    /**
     * One self-rewarding iteration: generate, self-judge, DPO on (best, worst).
     *
     * @param prompts the prompts
     * @param random the random source
     * @return how many preference pairs were made
     */
    public int iterate(List<int[]> prompts, SplittableRandom random) {
        List<DpoTrainer.Preference> pairs = new ArrayList<>();
        for (int[] prompt : prompts) {
            int[][] group = new int[candidates][];
            Arrays.fill(group, prompt);
            int[][] answers = model.sample(group, maxNewTokens, endToken, 1.0, random);
            int best = -1;
            int worst = -1;
            double hi = -1;
            double lo = 2;
            for (int i = 0; i < answers.length; i++) {
                if (answers[i].length == 0) {
                    continue;
                }
                double s = judge(prompt, answers[i]);
                if (s > hi) {
                    hi = s;
                    best = i;
                }
                if (s < lo) {
                    lo = s;
                    worst = i;
                }
            }
            if (best >= 0 && hi >= minChosen && hi - lo >= minGap && !Arrays.equals(answers[best], answers[worst])) {
                pairs.add(new DpoTrainer.Preference(prompt, answers[best], answers[worst]));
            }
        }
        if (pairs.isEmpty()) {
            return 0;
        }
        DpoTrainer dpo = DpoTrainer.of(model).beta(beta).optimizer(optimizer).chosenNll(nll);
        for (int s = 0; s < stepsPerIteration; s++) {
            dpo.step(pairs);
        }
        return pairs.size();
    }
}
