package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * GRPO for masked diffusion language models ({@link MaskedDiffusion}). A diffusion LM has no
 * left-to-right factorization, so the per-token log-probabilities that the policy gradient
 * needs must be estimated. Two estimators are built in ({@link Estimator}):
 * <ul>
 *   <li>{@code DIFFU_GRPO} (d1, Zhao et al., <a href="https://arxiv.org/abs/2504.12216">2504.12216</a>):
 *       one forward pass with the <em>whole</em> completion masked and a random 15% of the
 *       prompt tokens masked too; the log-probability of each completion token is read off
 *       that single pass (a mean-field estimate). Rollouts use the diffusion sampler.</li>
 *   <li>{@code JUST_GRPO} ("just do GRPO", ICML 2026 outstanding paper): during RL the
 *       completion is generated left to right, one masked position at a time, so its exact
 *       likelihood is the product of the per-position probabilities with the previous tokens
 *       revealed and the later ones masked; that is an ordinary autoregressive policy and
 *       plain GRPO applies. Parallel decoding is kept for inference.</li>
 * </ul>
 * Both use group-normalized advantages, token-level averaging and PPO clipping.
 */
public final class DiffusionGrpo {

    /** How the per-token log-probabilities are estimated. */
    public enum Estimator {
        /** d1's one-step estimate with prompt masking; rollouts from the diffusion sampler. */
        DIFFU_GRPO,
        /** Exact likelihood of left-to-right rollouts. */
        JUST_GRPO
    }

    /** Scores a whole generated sequence. */
    @FunctionalInterface
    public interface Reward {
        /**
         * @param sequence the prompt positions and the generated ones
         * @return the reward
         */
        float score(int[] sequence);
    }

    /**
     * What one step did.
     *
     * @param meanReward the mean reward of the rollouts
     * @param clipFraction the share of clipped tokens in the last epoch
     */
    public record Stats(double meanReward, double clipFraction) {
    }

    private final MaskedDiffusion diffusion;
    private final int vocabulary;
    private final int mask;
    private final Estimator estimator;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private int groupSize = 8;
    private int prompts = 16;
    private int innerEpochs = 2;
    private float clip = 0.2f;
    private double promptMask = 0.15;

    private DiffusionGrpo(MaskedDiffusion diffusion, int vocabulary, int mask, Estimator estimator) {
        this.diffusion = Objects.requireNonNull(diffusion);
        this.vocabulary = vocabulary;
        this.mask = mask;
        this.estimator = Objects.requireNonNull(estimator);
    }

    /**
     * @param diffusion the model to train
     * @param vocabulary its vocabulary size, mask included
     * @param mask the mask token
     * @param estimator the likelihood estimator
     * @return the trainer
     */
    public static DiffusionGrpo of(MaskedDiffusion diffusion, int vocabulary, int mask, Estimator estimator) {
        return new DiffusionGrpo(diffusion, vocabulary, mask, estimator);
    }

    /**
     * @param o the optimizer
     * @param group rollouts per prompt
     * @param promptsPerStep prompts per step
     * @param epochs gradient steps per batch
     * @return this trainer
     */
    public DiffusionGrpo settings(Optimizer o, int group, int promptsPerStep, int epochs) {
        this.optimizer = Objects.requireNonNull(o);
        this.groupSize = group;
        this.prompts = promptsPerStep;
        this.innerEpochs = epochs;
        return this;
    }

    private float[] logits(int[][] rows, boolean training, List<Var> out) {
        int len = rows[0].length;
        float[] flat = new float[rows.length * len];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < len; j++) {
                flat[i * len + j] = rows[i][j];
            }
        }
        Var logits = diffusion.model().forward(Var.of(Tensor.of(Shape.of(rows.length, len), flat)), training);
        if (out != null) {
            out.add(logits);
        }
        return logits.value().toArray();
    }

    /**
     * Generates the masked positions left to right, sampling each with the later ones still
     * masked: the JustGRPO rollout policy, whose likelihood is exact.
     *
     * @param prompt fixed tokens, -1 at the positions to generate
     * @param count how many sequences
     * @param sampling how each token is chosen
     * @param random the random source
     * @return the sequences
     */
    public int[][] leftToRight(int[] prompt, int count, Sampling sampling, SplittableRandom random) {
        int len = prompt.length;
        int[][] x = new int[count][];
        for (int i = 0; i < count; i++) {
            x[i] = new int[len];
            for (int j = 0; j < len; j++) {
                x[i][j] = prompt[j] >= 0 ? prompt[j] : mask;
            }
        }
        for (int j = 0; j < len; j++) {
            if (prompt[j] >= 0) {
                continue;
            }
            float[] l = logits(x, false, null);
            for (int i = 0; i < count; i++) {
                float[] row = Arrays.copyOfRange(l, (i * len + j) * vocabulary, (i * len + j + 1) * vocabulary);
                row[mask] = Float.NEGATIVE_INFINITY;
                x[i][j] = sampling.sample(row, new int[0], random);
            }
        }
        return x;
    }

    /**
     * The estimator's inputs: one row per forward input, and for every generated token the
     * row and position whose logits score it.
     */
    private record Plan(int[][] rows, int[] tokenRow, int[] tokenPos, int[] tokenId) {
    }

    private Plan plan(List<int[]> prompts, List<int[]> sequences, SplittableRandom random) {
        List<int[]> rows = new ArrayList<>();
        List<int[]> refs = new ArrayList<>();   // {row, pos, token}
        for (int s = 0; s < sequences.size(); s++) {
            int[] p = prompts.get(s);
            int[] seq = sequences.get(s);
            int len = seq.length;
            if (estimator == Estimator.DIFFU_GRPO) {
                int[] row = new int[len];
                for (int j = 0; j < len; j++) {
                    row[j] = p[j] < 0 || random.nextDouble() < promptMask ? mask : seq[j];
                }
                int r = rows.size();
                rows.add(row);
                for (int j = 0; j < len; j++) {
                    if (p[j] < 0) {
                        refs.add(new int[] {r, j, seq[j]});
                    }
                }
            } else {
                int[] row = new int[len];
                for (int j = 0; j < len; j++) {
                    row[j] = p[j] >= 0 ? seq[j] : mask;
                }
                for (int j = 0; j < len; j++) {
                    if (p[j] >= 0) {
                        continue;
                    }
                    refs.add(new int[] {rows.size(), j, seq[j]});
                    rows.add(row.clone());
                    row[j] = seq[j];   // revealed for the later positions
                }
            }
        }
        int[] tr = new int[refs.size()];
        int[] tp = new int[refs.size()];
        int[] ti = new int[refs.size()];
        for (int k = 0; k < tr.length; k++) {
            tr[k] = refs.get(k)[0];
            tp[k] = refs.get(k)[1];
            ti[k] = refs.get(k)[2];
        }
        return new Plan(rows.toArray(int[][]::new), tr, tp, ti);
    }

    /** Per-token log-probabilities {@code [tokens, 1]} under the current model, with gradient. */
    private Var logProbs(Plan plan) {
        List<Var> out = new ArrayList<>();
        logits(plan.rows(), true, out);
        int len = plan.rows()[0].length;
        int n = plan.rows().length * len;
        Var logp = Activation.LOG_SOFTMAX.forward(out.get(0).reshape(n, vocabulary));
        float[] pick = new float[n * vocabulary];
        float[] select = new float[plan.tokenRow().length * n];
        for (int k = 0; k < plan.tokenRow().length; k++) {
            int at = plan.tokenRow()[k] * len + plan.tokenPos()[k];
            pick[at * vocabulary + plan.tokenId()[k]] = 1;
            select[k * n + at] = 1;
        }
        Var perPosition = logp.mul(Tensor.of(Shape.of(n, vocabulary), pick)).sum(1, true);
        return Var.of(Tensor.of(Shape.of(plan.tokenRow().length, n), select)).matmul(perPosition);
    }

    /**
     * One GRPO step.
     *
     * @param promptSource draws a prompt: fixed tokens, -1 at the positions to generate
     * @param reward scores generated sequences
     * @param random the random source
     * @return what happened
     */
    public Stats step(Function<SplittableRandom, int[]> promptSource, Reward reward, SplittableRandom random) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> seqs = new ArrayList<>();
        List<Float> adv = new ArrayList<>();
        double rewardSum = 0;
        for (int p = 0; p < prompts; p++) {
            int[] prompt = promptSource.apply(random);
            int masked = 0;
            for (int t : prompt) {
                masked += t < 0 ? 1 : 0;
            }
            int[][] outs = estimator == Estimator.JUST_GRPO ? leftToRight(prompt, groupSize, Sampling.temperature(1), random)
                    : diffusion.sample(groupSize, prompt.length, masked, prompt, Sampling.temperature(1), random, null);
            float[] r = new float[groupSize];
            for (int g = 0; g < groupSize; g++) {
                r[g] = reward.score(outs[g]);
                rewardSum += r[g];
            }
            float[] a = GrpoTrainer.advantages(r, true);
            for (int g = 0; g < groupSize; g++) {
                ps.add(prompt);
                seqs.add(outs[g]);
                adv.add(a[g]);
            }
        }
        Plan plan = plan(ps, seqs, random);
        int tokens = plan.tokenRow().length;
        float[] tokenAdv = new float[tokens];
        int perSeq = tokens / seqs.size();
        for (int k = 0; k < tokens; k++) {
            tokenAdv[k] = adv.get(k / perSeq);
        }
        Tensor advT = LlmTensors.column(tokenAdv);
        Tensor old = logProbs(plan).value();
        double clipped = 0;
        for (int e = 0; e < innerEpochs; e++) {
            Var ratio = logProbs(plan).sub(old).exp();
            Var objective = ratio.mul(advT).minimum(ratio.clamp(1 - clip, 1 + clip).mul(advT)).mean();
            if (e == innerEpochs - 1) {
                int c = 0;
                for (float x : LlmTensors.values(ratio)) {
                    c += x < 1 - clip || x > 1 + clip ? 1 : 0;
                }
                clipped = (double) c / tokens;
            }
            objective.neg().backward();
            optimizer.step(diffusion.model().parameters());
        }
        return new Stats(rewardSum / (prompts * groupSize), clipped);
    }
}
