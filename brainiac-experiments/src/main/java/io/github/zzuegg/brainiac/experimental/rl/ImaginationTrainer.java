package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learns a discrete-action policy entirely in imagination, as Dreamer agents do from
 * offline data: rollouts of a learned {@link Simulator} start from logged observations, the
 * actor is trained by REINFORCE on normalized {@code lambda}-return advantages (plus an
 * entropy bonus) and the critic regresses the returns. No real environment step is taken.
 */
public final class ImaginationTrainer implements Agent {

    /** A learned model that can be stepped in batches. */
    @FunctionalInterface
    public interface Simulator {

        /**
         * @param obs observations {@code [n, obs]}
         * @param acts the actions
         * @return the next observations {@code [n, obs]} and rewards {@code [n, 1]}
         */
        Tensor[] step(Tensor obs, int[] acts);
    }

    private final int obsDims;
    private final int actions;
    private final Model actor;
    private final Model critic;
    private final Optimizer actorOpt = Optimizer.adam(3e-3f);
    private final Optimizer criticOpt = Optimizer.adam(3e-3f);
    private int horizon = 15;
    private int batch = 64;
    private float gamma = 0.95f;
    private float lambda = 0.95f;
    private float entropy = 0.01f;
    private final SplittableRandom random;

    private ImaginationTrainer(int obsDims, int actions, int hidden, long seed) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.actor = Nets.mlp(obsDims, actions, Activation.TANH, seed, hidden, hidden);
        this.critic = Nets.mlp(obsDims, 1, Activation.TANH, seed + 1, hidden, hidden);
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param obsDims values per observation
     * @param actions discrete actions
     * @param hidden hidden width
     * @param seed the seed
     * @return the trainer, itself the trained agent
     */
    public static ImaginationTrainer of(int obsDims, int actions, int hidden, long seed) {
        return new ImaginationTrainer(obsDims, actions, hidden, seed);
    }

    /** @param steps imagined steps per rollout @return this trainer */
    public ImaginationTrainer horizon(int steps) {
        this.horizon = steps;
        return this;
    }

    /**
     * Trains the actor and critic in imagination.
     *
     * @param model the simulator
     * @param starts observations to start rollouts from
     * @param iterations batches of rollouts
     * @return this trainer
     */
    public ImaginationTrainer train(Simulator model, List<float[]> starts, int iterations) {
        for (int it = 0; it < iterations; it++) {
            float[] s = new float[batch * obsDims];
            for (int i = 0; i < batch; i++) {
                System.arraycopy(starts.get(random.nextInt(starts.size())), 0, s, i * obsDims, obsDims);
            }
            Tensor obs = Tensor.of(Shape.of(batch, obsDims), s);
            List<Tensor> states = new ArrayList<>();
            List<int[]> acts = new ArrayList<>();
            float[][] rewards = new float[horizon][];
            for (int t = 0; t < horizon; t++) {
                states.add(obs);
                float[] logits = actor.forward(Var.of(obs), false).value().to(DType.FLOAT32).toArray();
                int[] a = new int[batch];
                for (int i = 0; i < batch; i++) {
                    a[i] = sample(logits, i);
                }
                acts.add(a);
                Tensor[] next = model.step(obs, a);
                rewards[t] = next[1].to(DType.FLOAT32).toArray();
                obs = next[0];
            }
            states.add(obs);
            float[][] v = new float[horizon + 1][];
            for (int t = 0; t <= horizon; t++) {
                v[t] = critic.forward(Var.of(states.get(t)), false).value().to(DType.FLOAT32).toArray();
            }
            float[][] ret = new float[horizon][batch];
            float[] g = v[horizon].clone();
            for (int t = horizon - 1; t >= 0; t--) {
                for (int i = 0; i < batch; i++) {
                    g[i] = rewards[t][i] + gamma * ((1 - lambda) * v[t + 1][i] + lambda * g[i]);
                    ret[t][i] = g[i];
                }
            }
            double sum = 0;
            double sq = 0;
            for (int t = 0; t < horizon; t++) {
                for (int i = 0; i < batch; i++) {
                    double a = ret[t][i] - v[t][i];
                    sum += a;
                    sq += a * a;
                }
            }
            int n = horizon * batch;
            double m = sum / n;
            double sd = Math.sqrt(Math.max(1e-8, sq / n - m * m));
            Tensor allObs = Tensor.concat(0, states.subList(0, horizon).toArray(new Tensor[0]));
            float[] adv = new float[n];
            float[] target = new float[n];
            float[] act = new float[n];
            for (int t = 0; t < horizon; t++) {
                for (int i = 0; i < batch; i++) {
                    adv[t * batch + i] = (float) ((ret[t][i] - v[t][i] - m) / sd);
                    target[t * batch + i] = ret[t][i];
                    act[t * batch + i] = acts.get(t)[i];
                }
            }
            Var logp = RlTensors.logSoftmax(actor.forward(Var.of(allObs), true));
            Tensor onehot = RlTensors.oneHot(act, actions, allObs.device());
            Var chosen = logp.mul(onehot).sum(1, true);
            Var ent = logp.exp().mul(logp).sum(1, true).neg();
            Var actorLoss = chosen.mul(Tensor.of(Shape.of(n, 1), adv)).add(ent.mul(entropy)).mean().neg();
            actorLoss.backward();
            actorOpt.step(actor.parameters());
            Var criticLoss = critic.forward(Var.of(allObs), true).sub(Tensor.of(Shape.of(n, 1), target)).square()
                    .mean();
            criticLoss.backward();
            criticOpt.step(critic.parameters());
        }
        return this;
    }

    private int sample(float[] logits, int row) {
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < actions; k++) {
            max = Math.max(max, logits[row * actions + k]);
        }
        double z = 0;
        for (int k = 0; k < actions; k++) {
            z += Math.exp(logits[row * actions + k] - max);
        }
        double u = random.nextDouble() * z;
        for (int k = 0; k < actions; k++) {
            u -= Math.exp(logits[row * actions + k] - max);
            if (u <= 0) {
                return k;
            }
        }
        return actions - 1;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] l = RlTensors.row(actor, observation);
        return new float[] {explore ? sample(l, 0) : RlTensors.argmax(l)};
    }
}
