package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Flow Q-learning (FQL, Park et al., <a href="https://arxiv.org/abs/2502.02538">2502.02538</a>)
 * with Q-chunking (Li et al., <a href="https://arxiv.org/abs/2507.07969">2507.07969</a>):
 * offline RL for continuous actions with an expressive behaviour policy.
 *
 * <ul>
 *   <li>A flow-matching behaviour-cloning policy ({@link FlowActionPolicy}) models the data's
 *       actions, multimodality included.</li>
 *   <li>A <em>one-step</em> policy {@code mu(s, z)} maps an observation and a noise vector
 *       straight to an action. It maximizes the critic while a distillation term keeps it
 *       close to what the multi-step flow produces from the <em>same</em> noise:
 *       {@code -Q(s, mu(s, z)) / mean|Q| + alpha ||mu(s, z) - flow(s, z)||^2}. Unlike
 *       backpropagating Q through the whole flow, this is cheap and stable, and unlike
 *       advantage weighting it can move beyond the best data action.</li>
 *   <li>Two critics with target networks; the TD target uses the one-step policy at the
 *       next state.</li>
 *   <li>Q-chunking: with {@code chunk > 1} every "action" is the next {@code chunk} actions,
 *       the critic is {@code Q(s_t, a_t..a_{t+h-1})} with the h-step return
 *       {@code sum_i gamma^i r_{t+i} + gamma^h Q(s_{t+h}, mu(s_{t+h}))} as target (unbiased,
 *       because the whole chunk is the policy's own), and the agent executes chunks open
 *       loop. Value propagates h times faster and the chunks inherit the data's temporal
 *       coherence.</li>
 * </ul>
 *
 * <p>Actions are assumed to lie in {@code [-1, 1]}.
 */
public final class FlowQLearning implements Agent {
    private final int observationSize;
    private final int actionSize;
    private final int chunk;
    private final FlowActionPolicy flow;
    private final Model actor;
    private final Model q1;
    private final Model q2;
    private final Model q1Target;
    private final Model q2Target;
    private final Optimizer actorOpt = Optimizer.adam(1e-3f);
    private final Optimizer criticOpt = Optimizer.adam(1e-3f);
    private float alpha = 3f;
    private float gamma = 0.97f;
    private float tau = 0.01f;
    private int batchSize = 64;
    private final SplittableRandom random;
    private float[][] queue;
    private int next;

    private FlowQLearning(int observationSize, int actionSize, int chunk, int width, long seed) {
        this.observationSize = observationSize;
        this.actionSize = actionSize;
        this.chunk = chunk;
        int a = chunk * actionSize;
        this.flow = FlowActionPolicy.of(observationSize, actionSize, chunk, width, 2, seed);
        this.actor = Nets.mlp(observationSize + a, a, Activation.RELU, seed + 1, width, width);
        this.q1 = Nets.mlp(observationSize + a, 1, Activation.RELU, seed + 2, width, width);
        this.q2 = Nets.mlp(observationSize + a, 1, Activation.RELU, seed + 3, width, width);
        this.q1Target = Nets.mlp(observationSize + a, 1, Activation.RELU, seed + 2, width, width);
        this.q2Target = Nets.mlp(observationSize + a, 1, Activation.RELU, seed + 3, width, width);
        this.random = new SplittableRandom(seed + 4);
    }

    /**
     * Creates the learner.
     *
     * @param observationSize the observation size
     * @param actionSize the action size
     * @param chunk the action-chunk length h; 1 is plain FQL
     * @param width the hidden width of every network
     * @param seed the seed
     * @return the learner
     */
    public static FlowQLearning of(int observationSize, int actionSize, int chunk, int width, long seed) {
        return new FlowQLearning(observationSize, actionSize, chunk, width, seed);
    }

    /**
     * Sets the distillation weight alpha; larger stays closer to the data.
     *
     * @param alpha the weight
     * @return this learner
     */
    public FlowQLearning alpha(float alpha) {
        this.alpha = alpha;
        return this;
    }

    /**
     * Sets the per-step discount.
     *
     * @param discount gamma
     * @return this learner
     */
    public FlowQLearning discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /**
     * Sets the target-network blending rate.
     *
     * @param tau the share of the online weights blended in per step
     * @return this learner
     */
    public FlowQLearning tau(float tau) {
        this.tau = tau;
        return this;
    }

    /** One chunked transition: observation, actions, discounted reward sum, next observation, bootstrap factor. */
    private record ChunkStep(float[] obs, float[] actions, float reward, float[] next, float bootstrap) {
    }

    private List<ChunkStep> chunkSteps(OfflineDataset data) {
        List<ChunkStep> out = new ArrayList<>();
        for (List<OfflineDataset.Step> t : data.trajectories()) {
            for (int i = 0; i < t.size(); i++) {
                float[] acts = new float[chunk * actionSize];
                double ret = 0;
                double g = 1;
                boolean ended = false;
                int k = 0;
                OfflineDataset.Step last = t.get(i);
                for (; k < chunk; k++) {
                    OfflineDataset.Step s = t.get(Math.min(i + k, t.size() - 1));
                    System.arraycopy(s.action(), 0, acts, k * actionSize, actionSize);
                    if (ended || i + k >= t.size()) {
                        continue;   // past the end: pad with the last action, no more reward
                    }
                    ret += g * s.reward();
                    g *= gamma;
                    last = s;
                    ended = s.terminated();
                }
                boolean truncated = !ended && i + chunk > t.size();
                if (truncated) {
                    continue;   // the chunk runs past a time-limit cut: no valid bootstrap
                }
                out.add(new ChunkStep(t.get(i).observation(), acts, (float) ret, last.nextObservation(),
                        ended ? 0 : (float) g));
            }
        }
        return out;
    }

    /**
     * Trains on a fixed data set.
     *
     * @param data the offline data
     * @param steps gradient steps (each updates the flow, the critics and the one-step policy)
     * @return this learner
     */
    public FlowQLearning train(OfflineDataset data, int steps) {
        ActionChunking.Pairs pairs = ActionChunking.pairs(data, chunk);
        List<ChunkStep> steps2 = chunkSteps(data);
        int a = chunk * actionSize;
        List<Var> critics = new ArrayList<>(q1.parameters());
        critics.addAll(q2.parameters());
        for (int s = 0; s < steps; s++) {
            flow.train(pairs, 1, batchSize);
            float[] o = new float[batchSize * observationSize];
            float[] act = new float[batchSize * a];
            float[] r = new float[batchSize];
            float[] n = new float[batchSize * observationSize];
            float[] boot = new float[batchSize];
            for (int b = 0; b < batchSize; b++) {
                ChunkStep c = steps2.get(random.nextInt(steps2.size()));
                System.arraycopy(c.obs(), 0, o, b * observationSize, observationSize);
                System.arraycopy(c.actions(), 0, act, b * a, a);
                System.arraycopy(c.next(), 0, n, b * observationSize, observationSize);
                r[b] = c.reward();
                boot[b] = c.bootstrap();
            }
            Tensor obs = Tensor.of(Shape.of(batchSize, observationSize), o);
            Tensor actions = Tensor.of(Shape.of(batchSize, a), act);
            Tensor nextObs = Tensor.of(Shape.of(batchSize, observationSize), n);

            Tensor nextA = clip(oneStep(Var.of(nextObs), noise()).value());
            Tensor nextIn = Tensor.concat(1, nextObs, nextA);
            Tensor minNext = q1Target.forward(Var.of(nextIn), false).value()
                    .minimum(q2Target.forward(Var.of(nextIn), false).value());
            Tensor y = Tensor.of(Shape.of(batchSize, 1), r).add(Tensor.of(Shape.of(batchSize, 1), boot).mul(minNext));
            Var in = Var.of(Tensor.concat(1, obs, actions));
            Var criticLoss = q1.forward(in, true).sub(y).square().mean().add(q2.forward(in, true).sub(y).square().mean());
            criticLoss.backward();
            RlTensors.clip(critics, 10f);
            criticOpt.step(critics);

            Tensor z = noise();
            Tensor target = flow.sample(obs, z);
            Var mu = oneStep(Var.of(obs), z);
            Var q = q1.forward(Var.concat(1, Var.of(obs), mu), true);
            float scale = (float) (1 / Math.max(1e-3, q.value().abs().mean().item()));
            Var distill = mu.sub(Var.of(clip(target))).square().sum(1, true).mean();
            Var actorLoss = q.mean().mul(-scale).add(distill.mul(alpha));
            actorLoss.backward();
            RlTensors.clip(actor.parameters(), 10f);
            actorOpt.step(actor.parameters());
            RlTensors.zeroGrad(critics);
            RlTensors.blend(q1.parameters(), q1Target.parameters(), tau);
            RlTensors.blend(q2.parameters(), q2Target.parameters(), tau);
        }
        return this;
    }

    private static Tensor clip(Tensor t) {
        return t.maximum(Tensor.full(Shape.of(), -1f)).minimum(Tensor.full(Shape.of(), 1f));
    }

    private Tensor noise() {
        return Tensor.normal(Shape.of(batchSize, chunk * actionSize), 0, 1, random);
    }

    private Var oneStep(Var obs, Tensor z) {
        return actor.forward(Var.concat(1, obs, Var.of(z)), true);
    }

    /**
     * The one-step policy's chunk for an observation.
     *
     * @param observation the observation
     * @return {@code [chunk][actionSize]}
     */
    public float[][] chunkFor(float[] observation) {
        Tensor z = Tensor.normal(Shape.of(1, chunk * actionSize), 0, 1, random);
        Tensor o = Tensor.of(Shape.of(1, observationSize), observation.clone());
        return ActionChunking.unflatten(clip(actor.forward(Var.concat(1, Var.of(o), Var.of(z)), false).value())
                .toArray(), actionSize);
    }

    /**
     * The smaller of the two critics' values.
     *
     * @param observation the observation
     * @param actions a chunk, flattened ({@code chunk * actionSize})
     * @return {@code min(Q1, Q2)}
     */
    public double value(float[] observation, float[] actions) {
        float[] in = new float[observationSize + actions.length];
        System.arraycopy(observation, 0, in, 0, observationSize);
        System.arraycopy(actions, 0, in, observationSize, actions.length);
        return Math.min(q1.predict(in)[0], q2.predict(in)[0]);
    }

    /** @return the flow behaviour-cloning policy trained alongside */
    public FlowActionPolicy behaviourPolicy() {
        return flow;
    }

    @Override
    public void startEpisode() {
        queue = null;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (queue == null || next >= queue.length) {
            queue = chunkFor(observation);
            next = 0;
        }
        return queue[next++].clone();
    }
}
