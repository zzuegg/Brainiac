package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * TD3+BC (Fujimoto and Gu, "A Minimalist Approach to Offline Reinforcement Learning", NeurIPS
 * 2021, <a href="https://arxiv.org/abs/2106.06860">arXiv:2106.06860</a>): {@link Td3} trained on a
 * fixed {@link OfflineDataset} with one behaviour-cloning term added to the actor's loss,
 *
 * <pre>{@code
 * loss = -lambda * Q(s, pi(s)) / mean|Q(s, pi(s))| + (pi(s) - a)^2,
 * }</pre>
 *
 * and states normalised by the data's mean and standard deviation. Dividing by the batch's mean
 * absolute Q makes {@code lambda} (2.5 in the paper) independent of the reward scale. With this
 * two-line change TD3 matches much more complex offline methods, which makes it the standard
 * sanity baseline of offline RL. Actions are stored and predicted in {@code [-1, 1]} and scaled
 * to the box given at construction.
 */
public final class Td3Bc implements Agent {

    private final int obsDims;
    private final int actDims;
    private final float[] low;
    private final float[] high;
    private final Model actor;
    private final Model actorTarget;
    private final Model q1;
    private final Model q2;
    private final Model q1Target;
    private final Model q2Target;
    private final Optimizer actorOpt;
    private final Optimizer criticOpt;
    private float alpha = 2.5f;
    private float gamma = 0.99f;
    private float tau = 0.005f;
    private int batchSize = 128;
    private float policyNoise = 0.2f;
    private float noiseClip = 0.5f;
    private boolean normalise = true;
    private SplittableRandom random = new SplittableRandom(0);
    private float[] mean;
    private float[] std;

    private Td3Bc(int obsDims, int actDims, float[] low, float[] high, RlNetworkFactory nets, float lr) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.low = low.clone();
        this.high = high.clone();
        this.actor = nets.create(obsDims, actDims, 21);
        this.actorTarget = nets.create(obsDims, actDims, 21);
        this.q1 = nets.create(obsDims + actDims, 1, 22);
        this.q2 = nets.create(obsDims + actDims, 1, 23);
        this.q1Target = nets.create(obsDims + actDims, 1, 22);
        this.q2Target = nets.create(obsDims + actDims, 1, 23);
        this.actorOpt = Optimizer.adam(lr);
        this.criticOpt = Optimizer.adam(lr);
    }

    /**
     * @param observationDims values per observation
     * @param low the action box's lower bounds
     * @param high the action box's upper bounds
     * @param networks builds the actor and critics
     * @return the learner, with learning rate 3e-4
     */
    public static Td3Bc of(int observationDims, float[] low, float[] high, RlNetworkFactory networks) {
        return of(observationDims, low, high, networks, 3e-4f);
    }

    /**
     * @param observationDims values per observation
     * @param low the action box's lower bounds
     * @param high the action box's upper bounds
     * @param networks builds the actor and critics
     * @param learningRate the learning rate of actor and critics
     * @return the learner
     */
    public static Td3Bc of(int observationDims, float[] low, float[] high, RlNetworkFactory networks,
            float learningRate) {
        Objects.requireNonNull(networks, "networks");
        if (low.length != high.length) {
            throw new IllegalArgumentException("low and high differ in length");
        }
        return new Td3Bc(observationDims, low.length, low, high, networks, learningRate);
    }

    /** @param weight lambda, the weight of the Q term (2.5 in the paper; 0 is pure BC) @return this learner */
    public Td3Bc alpha(float weight) {
        this.alpha = weight;
        return this;
    }

    /** @param discount gamma @return this learner */
    public Td3Bc discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param on whether states are normalised by the data's statistics @return this learner */
    public Td3Bc normaliseStates(boolean on) {
        this.normalise = on;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public Td3Bc batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Td3Bc seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    private float[] norm(float[] o) {
        if (mean == null) {
            return o;
        }
        float[] x = new float[o.length];
        for (int i = 0; i < o.length; i++) {
            x[i] = (o[i] - mean[i]) / std[i];
        }
        return x;
    }

    private float[] unit(float[] a) {
        float[] u = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            u[i] = Math.max(-1, Math.min(1, 2 * (a[i] - low[i]) / (high[i] - low[i]) - 1));
        }
        return u;
    }

    /**
     * Trains on logged data.
     *
     * @param data the dataset (continuous actions)
     * @param steps gradient steps
     * @return this learner
     */
    public Td3Bc train(OfflineDataset data, int steps) {
        List<OfflineDataset.Step> all = data.steps();
        if (normalise) {
            mean = new float[obsDims];
            std = new float[obsDims];
            for (OfflineDataset.Step s : all) {
                for (int i = 0; i < obsDims; i++) {
                    mean[i] += s.observation()[i] / all.size();
                }
            }
            for (OfflineDataset.Step s : all) {
                for (int i = 0; i < obsDims; i++) {
                    float d = s.observation()[i] - mean[i];
                    std[i] += d * d / all.size();
                }
            }
            for (int i = 0; i < obsDims; i++) {
                std[i] = (float) Math.sqrt(std[i]) + 1e-3f;
            }
        }
        ReplayBuffer buffer = new ReplayBuffer(all.size(), obsDims, actDims);
        for (OfflineDataset.Step s : all) {
            buffer.add(new Transition(norm(s.observation()), unit(s.action()), s.reward(), norm(s.nextObservation()),
                    s.terminated(), gamma));
        }
        RlTensors.copy(actor.parameters(), actorTarget.parameters());
        RlTensors.copy(q1.parameters(), q1Target.parameters());
        RlTensors.copy(q2.parameters(), q2Target.parameters());
        Device device = actor.device();
        for (int step = 0; step < steps; step++) {
            ReplayBuffer.Batch b = buffer.sample(batchSize, random, device);
            Tensor next = Var.noGrad(() -> actorTarget.forward(Var.of(b.nextObservations()), false)).value().tanh();
            Tensor noise = Tensor.normal(next.shape(), 0, policyNoise, random).maximum(Tensor.scalar(-noiseClip))
                    .minimum(Tensor.scalar(noiseClip));
            Tensor nextA = next.add(noise).maximum(Tensor.scalar(-1)).minimum(Tensor.scalar(1));
            Tensor nextIn = Tensor.concat(1, b.nextObservations(), nextA);
            Tensor qNext = Var.noGrad(() -> q1Target.forward(Var.of(nextIn), false)).value()
                    .minimum(Var.noGrad(() -> q2Target.forward(Var.of(nextIn), false)).value());
            Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(qNext));
            Var in = Var.of(Tensor.concat(1, b.observations(), b.actions()));
            Var loss = q1.forward(in, true).sub(y).square().mean().add(q2.forward(in, true).sub(y).square().mean());
            loss.backward();
            List<Var> critics = new ArrayList<>(q1.parameters());
            critics.addAll(q2.parameters());
            criticOpt.step(critics);
            if (step % 2 == 0) {
                Var pi = actor.forward(Var.of(b.observations()), true).tanh();
                Var q = q1.forward(Var.concat(1, Var.of(b.observations()), pi), false);
                float lambda = alpha / (float) Math.max(1e-6, q.value().abs().mean().item());
                Var actorLoss = q.mean().mul(-lambda).add(pi.sub(b.actions()).square().mean());
                actorLoss.backward();
                actorOpt.step(actor.parameters());
                RlTensors.zeroGrad(critics);
                RlTensors.blend(actor.parameters(), actorTarget.parameters(), tau);
                RlTensors.blend(q1.parameters(), q1Target.parameters(), tau);
                RlTensors.blend(q2.parameters(), q2Target.parameters(), tau);
            }
        }
        return this;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] x = norm(observation);
        float[] out = Var.noGrad(() -> actor.forward(Var.of(Tensor.of(Shape.of(1, obsDims), x)), false)).value()
                .toArray();
        float[] a = new float[actDims];
        for (int i = 0; i < actDims; i++) {
            a[i] = low[i] + ((float) Math.tanh(out[i]) + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    @Override
    public String toString() {
        return "TD3+BC(alpha " + alpha + ")";
    }
}
