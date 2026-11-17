package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Quiet-STaR (Zelikman et al. 2024, <a href="https://arxiv.org/abs/2403.09629">2403.09629</a>):
 * a language model learns to "think before it speaks" at every position, with no rationale
 * data. Before predicting each token, it samples a short hidden thought delimited by
 * start-of-thought and end-of-thought tokens; the next token is predicted from the text plus
 * the thought, and the thought is then discarded (later positions do not see it).
 *
 * <p>The thoughts are trained with REINFORCE: the reward of a thought is how much it raises
 * the log-likelihood of the true next token compared with the other thoughts sampled at the
 * same position ({@code r_j = log p(x | thought_j) - mean_k log p(x | thought_k)}), so
 * thoughts that help prediction become likely. The predictor itself is trained with the
 * ordinary next-token loss given the sampled thoughts. (The paper also learns a mixing head
 * between the with- and without-thought predictions and uses teacher-forced look-ahead over
 * several tokens; this block keeps the core: per-position thoughts and the REINFORCE reward.)
 *
 * <pre>{@code
 * QuietStar qs = QuietStar.of(lm, START, END).thoughts(1, 4);
 * qs.step(prompts, targets, optimizer, random);
 * int[] answer = qs.generate(prompt, 2, eos);
 * }</pre>
 */
public final class QuietStar {

    /**
     * What one step did.
     *
     * @param nll the mean next-token negative log-likelihood given the thoughts
     * @param rewardSpread the mean absolute REINFORCE reward (how much thoughts mattered)
     */
    public record Stats(double nll, double rewardSpread) {
    }

    private final WindowLm lm;
    private final int start;
    private final int end;
    private int thoughtLength = 1;
    private int samples = 4;
    private float reinforce = 1;
    private double temperature = 1;

    private QuietStar(WindowLm lm, int start, int end) {
        this.lm = Objects.requireNonNull(lm);
        this.start = start;
        this.end = end;
    }

    /**
     * @param lm the model; its vocabulary must contain the two delimiter tokens
     * @param startThought the start-of-thought token
     * @param endThought the end-of-thought token
     * @return the trainer
     */
    public static QuietStar of(WindowLm lm, int startThought, int endThought) {
        return new QuietStar(lm, startThought, endThought);
    }

    /**
     * @param length the most tokens per thought
     * @param perPosition thoughts sampled per position during training
     * @return this trainer
     */
    public QuietStar thoughts(int length, int perPosition) {
        if (perPosition < 2) {
            throw new IllegalArgumentException("the REINFORCE baseline needs at least two thoughts per position");
        }
        this.thoughtLength = length;
        this.samples = perPosition;
        return this;
    }

    /**
     * @param weight the weight of the REINFORCE term
     * @return this trainer
     */
    public QuietStar reinforceWeight(float weight) {
        this.reinforce = weight;
        return this;
    }

    /**
     * @param t the thought sampling temperature during training
     * @return this trainer
     */
    public QuietStar temperature(double t) {
        this.temperature = t;
        return this;
    }

    private static int[] concat(int[] a, int... b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private int[] close(int[] thought) {
        return thought.length > 0 && thought[thought.length - 1] == end ? thought : concat(thought, end);
    }

    /**
     * One training step: thoughts at every target position, next-token loss given them and
     * REINFORCE on the thoughts.
     *
     * @param prompts the prompts
     * @param targets the tokens to predict after each prompt
     * @param optimizer the optimizer
     * @param random the random source
     * @return what happened
     */
    public Stats step(List<int[]> prompts, List<int[]> targets, Optimizer optimizer, SplittableRandom random) {
        List<int[]> opened = new ArrayList<>();
        List<Integer> next = new ArrayList<>();
        for (int s = 0; s < prompts.size(); s++) {
            int[] p = prompts.get(s);
            int[] y = targets.get(s);
            for (int t = 0; t < y.length; t++) {
                int[] text = Arrays.copyOf(p, p.length + t);
                System.arraycopy(y, 0, text, p.length, t);
                int[] withStart = concat(text, start);
                for (int k = 0; k < samples; k++) {
                    opened.add(withStart);
                    next.add(y[t]);
                }
            }
        }
        int[][] thoughts = lm.sampleUntil(opened.toArray(int[][]::new), thoughtLength, new int[] {end}, temperature,
                random);
        List<int[]> thoughtPrompts = new ArrayList<>();
        List<int[]> thoughtTokens = new ArrayList<>();
        List<int[]> full = new ArrayList<>();
        int[] targetIds = new int[opened.size()];
        for (int i = 0; i < opened.size(); i++) {
            int[] th = close(thoughts[i]);
            thoughtPrompts.add(opened.get(i));
            thoughtTokens.add(th);
            full.add(concat(opened.get(i), th));
            targetIds[i] = next.get(i);
        }
        Var nextLogp = LlmTensors.pick(LlmTensors.logSoftmax(lm.logits(full)), targetIds);
        Var thoughtLogp = Var.of(WindowLm.sequenceSum(thoughtTokens, false))
                .matmul(lm.logProbs(thoughtPrompts, thoughtTokens));
        float[] lp = LlmTensors.values(nextLogp);
        float[] reward = new float[lp.length];
        double spread = 0;
        for (int g = 0; g < lp.length; g += samples) {
            double mean = 0;
            for (int k = 0; k < samples; k++) {
                mean += lp[g + k];
            }
            mean /= samples;
            for (int k = 0; k < samples; k++) {
                reward[g + k] = (float) (lp[g + k] - mean);
                spread += Math.abs(reward[g + k]);
            }
        }
        Tensor r = LlmTensors.column(reward);
        Var nll = nextLogp.mean().neg();
        double nllValue = nll.value().item();
        Var loss = nll.sub(thoughtLogp.mul(r).mean().mul(reinforce));
        loss.backward();
        optimizer.step(lm.parameters());
        return new Stats(nllValue, spread / lp.length);
    }

    /**
     * The thought the model has at the end of a text, decoded greedily.
     *
     * @param text the text so far
     * @return the thought, end-of-thought token included
     */
    public int[] think(int[] text) {
        return close(lm.sampleUntil(new int[][] {concat(text, start)}, thoughtLength, new int[] {end}, 0,
                new SplittableRandom(0))[0]);
    }

    /**
     * Greedy generation that thinks before every token; thoughts are not part of the output
     * and later positions do not see them.
     *
     * @param prompt the prompt
     * @param maxNew the most tokens to add
     * @param eos the end token
     * @return the generated tokens
     */
    public int[] generate(int[] prompt, int maxNew, int eos) {
        int[] text = prompt;
        for (int i = 0; i < maxNew; i++) {
            int[] full = concat(concat(text, start), think(text));
            int tok = LlmTensors.argmax(lm.logits(full));
            text = concat(text, tok);
            if (tok == eos) {
                break;
            }
        }
        return Arrays.copyOfRange(text, prompt.length, text.length);
    }
}
