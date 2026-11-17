package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * The classic RLHF recipe: token-level PPO with a learned critic (Ouyang et al. 2022,
 * InstructGPT, <a href="https://arxiv.org/abs/2203.02155">2203.02155</a>; Stiennon et al.
 * 2020, <a href="https://arxiv.org/abs/2009.01325">2009.01325</a>; the PPO details follow
 * Huang et al., "The N+ implementation details of RLHF with PPO",
 * <a href="https://arxiv.org/abs/2403.17031">2403.17031</a>).
 *
 * <p>Each generated token is an action. The per-token reward is the KL penalty
 * {@code -beta (log pi_old - log ref)} and the last token also gets the task reward. A
 * value head (an MLP over the same features as the policy) estimates the return of each
 * prefix; generalized advantage estimation (GAE, Schulman et al.
 * <a href="https://arxiv.org/abs/1506.02438">1506.02438</a>) turns rewards and values into
 * advantages, which are whitened over the batch. The policy then takes a few clipped
 * surrogate steps and the critic regresses on the returns.
 *
 * <p>It is the baseline the critic-free recipes in {@link GrpoTrainer} (GRPO, RLOO, ReMax,
 * REINFORCE++, GSPO) were invented to replace: it needs a second network and its
 * optimizer, but gets a per-token baseline instead of a per-group one.
 *
 * <pre>{@code
 * PpoLmTrainer ppo = PpoLmTrainer.of(lm).completions(task.maxAnswerTokens(), task.eos());
 * for (int i = 0; i < 300; i++) {
 *     ppo.step(task::samplePrompt, task::reward, random);
 * }
 * }</pre>
 */
public final class PpoLmTrainer {

    /**
     * What one step did.
     *
     * @param meanReward the mean task reward of the sampled completions
     * @param meanKl the mean per-token KL estimate {@code log pi_old - log ref}
     * @param valueLoss the critic's mean squared error before its update
     * @param clipFraction the fraction of tokens whose ratio was clipped in the last epoch
     */
    public record Stats(double meanReward, double meanKl, double valueLoss, double clipFraction) {
    }

    private final WindowLm policy;
    private final Model value;
    private WindowLm reference;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private Optimizer valueOptimizer = Optimizer.adam(3e-3f);
    private int batch = 128;
    private int innerEpochs = 2;
    private int maxNewTokens = 4;
    private int endToken = -1;
    private float clip = 0.2f;
    private float kl = 0.05f;
    private double gamma = 1.0;
    private double lambda = 0.95;
    private double temperature = 1.0;

    private PpoLmTrainer(WindowLm policy, int hidden, long seed) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
        this.value = Model.sequential()
                .input(policy.encode(new int[0]).length)
                .dense(hidden, Activation.TANH)
                .dense(1)
                .seed(seed)
                .build();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the KL reference
     * @return the trainer, with a 64-unit value head
     */
    public static PpoLmTrainer of(WindowLm policy) {
        return new PpoLmTrainer(policy, 64, 7);
    }

    /**
     * @param policy the model to train
     * @param valueHidden the width of the value head
     * @param seed the value head's initialization seed
     * @return the trainer
     */
    public static PpoLmTrainer of(WindowLm policy, int valueHidden, long seed) {
        return new PpoLmTrainer(policy, valueHidden, seed);
    }

    /** @return the critic: prefix features to the expected return */
    public Model valueHead() {
        return value;
    }

    /**
     * @param o the policy optimizer
     * @param v the critic optimizer
     * @return this trainer
     */
    public PpoLmTrainer optimizers(Optimizer o, Optimizer v) {
        this.optimizer = Objects.requireNonNull(o);
        this.valueOptimizer = Objects.requireNonNull(v);
        return this;
    }

    /**
     * @param n completions sampled per step (one per prompt)
     * @return this trainer
     */
    public PpoLmTrainer batch(int n) {
        this.batch = n;
        return this;
    }

    /**
     * @param n PPO epochs over each batch
     * @return this trainer
     */
    public PpoLmTrainer innerEpochs(int n) {
        this.innerEpochs = n;
        return this;
    }

    /**
     * @param n the most tokens per completion
     * @param end the end token
     * @return this trainer
     */
    public PpoLmTrainer completions(int n, int end) {
        this.maxNewTokens = n;
        this.endToken = end;
        return this;
    }

    /**
     * @param epsilon the ratio clip range
     * @return this trainer
     */
    public PpoLmTrainer clip(float epsilon) {
        this.clip = epsilon;
        return this;
    }

    /**
     * @param beta the per-token KL penalty in the reward
     * @return this trainer
     */
    public PpoLmTrainer klCoefficient(float beta) {
        this.kl = beta;
        return this;
    }

    /**
     * @param discount the discount {@code gamma}, usually 1 for language
     * @param gae the GAE {@code lambda}
     * @return this trainer
     */
    public PpoLmTrainer gae(double discount, double gae) {
        this.gamma = discount;
        this.lambda = gae;
        return this;
    }

    /**
     * @param t the sampling temperature
     * @return this trainer
     */
    public PpoLmTrainer temperature(double t) {
        this.temperature = t;
        return this;
    }

    /**
     * Generalized advantage estimation over one episode.
     *
     * @param rewards per-step rewards
     * @param values per-step value estimates; the value after the last step is 0
     * @param discount {@code gamma}
     * @param lambda the GAE {@code lambda}
     * @return the advantages; returns are {@code advantages + values}
     */
    public static double[] gae(double[] rewards, double[] values, double discount, double lambda) {
        int n = rewards.length;
        double[] a = new double[n];
        double next = 0;
        double running = 0;
        for (int t = n - 1; t >= 0; t--) {
            double delta = rewards[t] + discount * next - values[t];
            running = delta + discount * lambda * running;
            a[t] = running;
            next = values[t];
        }
        return a;
    }

    private Var values(List<int[]> prefixes) {
        float[][] rows = new float[prefixes.size()][];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = policy.encode(prefixes.get(i));
        }
        return value.forward(Var.of(LlmTensors.rows(rows)), true);
    }

    /**
     * Samples one completion per prompt, scores it and runs the PPO epochs.
     *
     * @param promptSource draws a prompt
     * @param reward the task reward of a whole completion
     * @param random the random source
     * @return what happened
     */
    public Stats step(Function<SplittableRandom, int[]> promptSource, GrpoTrainer.Reward reward,
            SplittableRandom random) {
        if (endToken < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
        int[][] prompts = new int[batch][];
        for (int i = 0; i < batch; i++) {
            prompts[i] = promptSource.apply(random);
        }
        int[][] outs = policy.sample(prompts, maxNewTokens, endToken, temperature, random);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        double rewardSum = 0;
        for (int i = 0; i < batch; i++) {
            float r = reward.score(prompts[i], outs[i]);
            rewardSum += r;
            if (outs[i].length > 0) {
                ps.add(prompts[i]);
                cs.add(outs[i]);
                scores.add(r);
            }
        }
        if (cs.isEmpty()) {
            return new Stats(rewardSum / batch, 0, 0, 0);
        }
        List<int[]> prefixes = new ArrayList<>();
        for (int s = 0; s < cs.size(); s++) {
            int[] p = ps.get(s);
            int[] c = cs.get(s);
            for (int t = 0; t < c.length; t++) {
                int[] prefix = Arrays.copyOf(p, p.length + t);
                System.arraycopy(c, 0, prefix, p.length, t);
                prefixes.add(prefix);
            }
        }
        Tensor old = policy.logProbs(ps, cs).value();
        float[] oldV = old.toArray();
        float[] refV = reference.logProbs(ps, cs).value().toArray();
        float[] v = LlmTensors.values(values(prefixes));
        int tokens = oldV.length;
        float[] adv = new float[tokens];
        float[] ret = new float[tokens];
        double klSum = 0;
        int k = 0;
        for (int s = 0; s < cs.size(); s++) {
            int len = cs.get(s).length;
            double[] r = new double[len];
            double[] vs = new double[len];
            for (int t = 0; t < len; t++) {
                double d = oldV[k + t] - refV[k + t];
                klSum += d;
                r[t] = -kl * d;
                vs[t] = v[k + t];
            }
            r[len - 1] += scores.get(s);
            double[] a = gae(r, vs, gamma, lambda);
            for (int t = 0; t < len; t++) {
                adv[k + t] = (float) a[t];
                ret[k + t] = (float) (a[t] + vs[t]);
            }
            k += len;
        }
        whiten(adv);
        Tensor advT = LlmTensors.column(adv);
        Tensor retT = LlmTensors.column(ret);
        double valueLoss = 0;
        double clipped = 0;
        for (int e = 0; e < innerEpochs; e++) {
            Var logp = policy.logProbs(ps, cs);
            Var ratio = logp.sub(old).exp();
            Var surrogate = ratio.mul(advT).minimum(ratio.clamp(1 - clip, 1 + clip).mul(advT));
            Var loss = surrogate.mean().neg();
            loss.backward();
            optimizer.step(policy.parameters());

            Var critic = values(prefixes).sub(retT).square().mean();
            if (e == 0) {
                valueLoss = critic.value().item();
            }
            critic.backward();
            valueOptimizer.step(value.parameters());
            if (e == innerEpochs - 1) {
                int c = 0;
                for (float x : LlmTensors.values(ratio)) {
                    if (x < 1 - clip || x > 1 + clip) {
                        c++;
                    }
                }
                clipped = (double) c / tokens;
            }
        }
        return new Stats(rewardSum / batch, klSum / tokens, valueLoss, clipped);
    }

    private static void whiten(float[] a) {
        double mean = LlmTensors.mean(a);
        double var = 0;
        for (float x : a) {
            var += (x - mean) * (x - mean);
        }
        double std = Math.sqrt(var / Math.max(1, a.length - 1));
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) ((a[i] - mean) / (std + 1e-6));
        }
    }
}
