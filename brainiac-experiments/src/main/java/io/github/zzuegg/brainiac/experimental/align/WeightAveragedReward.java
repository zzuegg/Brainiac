package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Weight-averaged reward models (WARM, Ramé et al., ICML 2024,
 * <a href="https://arxiv.org/abs/2401.12187">arXiv:2401.12187</a>): several reward models are
 * fine-tuned from <em>one shared initialization</em> on different orders and subsets of the
 * preference data, then their weights are averaged into a single model. Fine-tunes of one
 * pretrained model stay linearly mode-connected, so the average is itself a good reward model,
 * and averaging weights acts like averaging predictions: it keeps the members' shared,
 * invariant features and cancels their individual memorization of noisy labels. The result
 * has most of an ensemble's robustness against reward hacking at the inference cost of one
 * model.
 *
 * <p>{@link #average} works on any architecture given parallel parameter lists;
 * {@link #fit} trains WARM members of a {@link RewardModel} and returns the averaged model
 * together with the members (for the prediction-ensemble comparison).
 */
public final class WeightAveragedReward {

    /**
     * Trained WARM members and their weight average.
     *
     * @param averaged the weight-averaged model
     * @param members the individual fine-tunes
     */
    public record Result(RewardModel averaged, List<RewardModel> members) {
    }

    private WeightAveragedReward() {
    }

    /**
     * Writes the element-wise mean of the members' parameters into {@code into}.
     *
     * @param members the members' parameters, all in the same order and shapes
     * @param into the parameters to overwrite
     */
    public static void average(List<List<Var>> members, List<Var> into) {
        for (int i = 0; i < into.size(); i++) {
            Tensor sum = null;
            for (List<Var> m : members) {
                Tensor v = m.get(i).value();
                sum = sum == null ? v : sum.add(v);
            }
            if (sum != null) {
                into.get(i).assign(sum.mul(1f / members.size()));
            }
        }
    }

    /**
     * Linear interpolation {@code (1 - t) a + t b} of two parameter lists, written into
     * {@code into}: the probe of linear mode connectivity.
     *
     * @param a the first end point
     * @param b the second end point
     * @param t the position, 0 at {@code a}
     * @param into the parameters to overwrite
     */
    public static void interpolate(List<Var> a, List<Var> b, float t, List<Var> into) {
        for (int i = 0; i < into.size(); i++) {
            into.get(i).assign(a.get(i).value().mul(1 - t).add(b.get(i).value().mul(t)));
        }
    }

    /**
     * Fine-tunes {@code members} reward models from one shared backbone, each on its own
     * bootstrap sample and order of the pairs, and averages their weights.
     *
     * @param sharedInit the shared initialization (copied, not changed)
     * @param chosen the preferred sequences
     * @param rejected the other sequences, the same length
     * @param members how many fine-tunes
     * @param steps optimizer steps per member
     * @param batch pairs per step
     * @param learningRate the Adam learning rate
     * @param seed the seed of the data orders
     * @return the averaged model and the members
     */
    public static Result fit(HookedDecoder sharedInit, int[][] chosen, int[][] rejected, int members, int steps,
            int batch, float learningRate, long seed) {
        List<RewardModel> ms = new ArrayList<>();
        SplittableRandom random = new SplittableRandom(seed);
        for (int m = 0; m < members; m++) {
            RewardModel rm = new RewardModel(sharedInit.copy());
            train(rm, chosen, rejected, steps, batch, learningRate, random.split());
            ms.add(rm);
        }
        RewardModel avg = new RewardModel(sharedInit.copy());
        List<List<Var>> params = new ArrayList<>();
        for (RewardModel rm : ms) {
            params.add(rm.parameters());
        }
        average(params, avg.parameters());
        return new Result(avg, List.copyOf(ms));
    }

    /**
     * Trains a reward model with the Bradley-Terry loss on random minibatches.
     *
     * @param rm the model
     * @param chosen the preferred sequences
     * @param rejected the other sequences
     * @param steps optimizer steps
     * @param batch pairs per step
     * @param learningRate the Adam learning rate
     * @param random the minibatch order
     */
    public static void train(RewardModel rm, int[][] chosen, int[][] rejected, int steps, int batch,
            float learningRate, SplittableRandom random) {
        Optimizer opt = Optimizer.adam(learningRate);
        int n = chosen.length;
        int b = Math.min(batch, n);
        for (int s = 0; s < steps; s++) {
            int[][] c = new int[b][];
            int[][] r = new int[b][];
            for (int i = 0; i < b; i++) {
                int j = random.nextInt(n);
                c[i] = chosen[j];
                r[i] = rejected[j];
            }
            rm.zeroGrad();
            rm.loss(c, r).backward();
            opt.step(rm.parameters());
        }
    }
}
