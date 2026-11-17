package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.VectorConditionedDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A pi0-style flow-matching action expert (Black et al., <a href="https://arxiv.org/abs/2410.24164">2410.24164</a>):
 * a velocity network over an action chunk, conditioned on the observation, trained with
 * core {@link FlowMatching} and sampled with a few Euler steps from noise to actions.
 *
 * <p>It also supports <em>real-time chunking</em> (RTC, Black et al.,
 * <a href="https://arxiv.org/abs/2506.07339">2506.07339</a>) through
 * {@link #chunk(float[], float[][], int, SplittableRandom)}: when the next chunk is computed
 * while the current one keeps running, its first actions are already decided. RTC treats
 * the new chunk as inpainting: at every denoising step it estimates the final chunk
 * {@code A = x - t v}, and adds to the velocity the gradient (through the network, by
 * autograd) of a weighted squared error between {@code A} and the committed actions, with
 * weight 1 on the frozen prefix, an exponentially decaying weight over the rest of the
 * overlap and 0 after it, the guidance step capped at {@link #guidanceLimit}. The chunks
 * then join smoothly instead of jumping between modes. No retraining is needed.
 *
 * <p>The time convention is core flow matching's: {@code t = 0} is data, {@code t = 1} noise.
 */
public final class FlowActionPolicy implements ChunkPolicy, Agent {
    private final int observationSize;
    private final int actionSize;
    private final int horizon;
    private final VectorConditionedDenoiser net;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private final SplittableRandom random;
    private int samplingSteps = 10;
    private float guidanceLimit = 5f;
    private int execute;
    private Agent executor;

    private FlowActionPolicy(int observationSize, int actionSize, int horizon, int width, int depth, long seed) {
        this.observationSize = observationSize;
        this.actionSize = actionSize;
        this.horizon = horizon;
        this.net = VectorConditionedDenoiser.of(horizon * actionSize, observationSize, width, depth, seed);
        this.random = new SplittableRandom(seed + 1);
        this.execute = Math.max(1, horizon / 2);
    }

    /**
     * Creates a policy.
     *
     * @param observationSize the observation size
     * @param actionSize the action size (actions live in [-1, 1])
     * @param horizon the chunk length H
     * @param width the network width
     * @param depth the network's residual layers
     * @param seed the seed
     * @return the policy
     */
    public static FlowActionPolicy of(int observationSize, int actionSize, int horizon, int width, int depth,
            long seed) {
        return new FlowActionPolicy(observationSize, actionSize, horizon, width, depth, seed);
    }

    /**
     * Sets the number of Euler steps.
     *
     * @param steps at least 1
     * @return this policy
     */
    public FlowActionPolicy samplingSteps(int steps) {
        this.samplingSteps = steps;
        return this;
    }

    /**
     * Sets the cap on the RTC guidance weight (beta in the paper).
     *
     * @param limit the cap
     * @return this policy
     */
    public FlowActionPolicy guidanceLimit(float limit) {
        this.guidanceLimit = limit;
        return this;
    }

    /**
     * Sets how many actions of each chunk run before replanning, when used directly as an
     * {@link Agent}.
     *
     * @param steps 1 to the horizon
     * @return this policy
     */
    public FlowActionPolicy execute(int steps) {
        this.execute = steps;
        this.executor = null;
        return this;
    }

    /**
     * Trains on (observation, chunk) pairs with the core flow-matching loss.
     *
     * @param pairs the demonstrations, cut into chunks of this policy's horizon
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the last loss
     */
    public float train(ActionChunking.Pairs pairs, int steps, int batch) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Tensor[] b = DiffusionPolicy.batch(pairs, batch, random);
            Var loss = FlowMatching.loss(net.given(b[0]), b[1], random);
            last = loss.value().item();
            loss.backward();
            RlTensors.clip(net.parameters(), 1f);
            optimizer.step(net.parameters());
        }
        return last;
    }

    /**
     * Samples chunks for a batch of observations from given noise, the multi-step flow
     * output that Flow Q-learning distils into a one-step policy.
     *
     * @param observations {@code [batch, observationSize]}
     * @param noise {@code [batch, horizon * actionSize]}
     * @return {@code [batch, horizon * actionSize]}, not clipped
     */
    public Tensor sample(Tensor observations, Tensor noise) {
        return FlowMatching.sample(net.given(observations), noise, samplingSteps);
    }

    @Override
    public float[][] chunk(float[] observation, SplittableRandom random) {
        return chunk(observation, null, 0, random);
    }

    @Override
    public float[][] chunk(float[] observation, float[][] previous, int frozen, SplittableRandom random) {
        int size = horizon * actionSize;
        Tensor cond = VectorConditionedDenoiser.repeat(observation, 1);
        Tensor x = Tensor.normal(Shape.of(1, size), 0, 1, random);
        float[] target = new float[size];
        float[] weight = new float[size];
        boolean guided = previous != null && previous.length > 0;
        if (guided) {
            int overlap = Math.min(previous.length, horizon);
            for (int i = 0; i < overlap; i++) {
                double w;
                if (i < frozen) {
                    w = 1;
                } else {
                    double c = (double) (overlap - i) / (overlap - frozen + 1);
                    w = c * (Math.exp(c) - 1) / (Math.E - 1);
                }
                for (int k = 0; k < actionSize; k++) {
                    target[i * actionSize + k] = previous[i][k];
                    weight[i * actionSize + k] = (float) w;
                }
            }
        }
        Tensor y = Tensor.of(Shape.of(1, size), target);
        Tensor w = Tensor.of(Shape.of(1, size), weight);
        float dt = 1f / samplingSteps;
        for (int s = 0; s < samplingSteps; s++) {
            float t = 1 - s * dt;
            Tensor time = Tensor.of(Shape.of(1), new float[] {t});
            Tensor v;
            if (guided && t < 1) {
                Var xv = Var.param(x);
                Var out = net.apply(xv, time, cond);
                Var estimate = xv.sub(out.mul(t));   // the predicted final chunk
                Var err = estimate.sub(Var.of(y)).square().mul(Var.of(w)).sum().mul(0.5f);
                Tensor grad = Var.gradients(err, List.of(xv), false).getFirst().value();
                double r2 = (double) t * t / ((1 - t) * (1 - t) + t * t);
                float k = (float) Math.min(guidanceLimit, t / ((1 - t) * r2));
                v = out.value().add(grad.mul(k));
            } else {   // unguided, or the first step from pure noise where the estimate says nothing
                v = net.apply(Var.of(x), time, cond).value();
            }
            x = x.sub(v.mul(dt));
        }
        float[][] out = ActionChunking.unflatten(x.toArray(), actionSize);
        for (int i = 0; i < out.length; i++) {
            out[i] = ActionChunking.clip(out[i]);
        }
        return out;
    }

    @Override
    public int horizon() {
        return horizon;
    }

    @Override
    public int actionSize() {
        return actionSize;
    }

    /** @return the observation size */
    public int observationSize() {
        return observationSize;
    }

    /** @return the velocity network */
    public VectorConditionedDenoiser network() {
        return net;
    }

    @Override
    public void startEpisode() {
        executor().startEpisode();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return executor().act(observation, explore);
    }

    private Agent executor() {
        if (executor == null) {
            executor = ActionChunking.recedingHorizon(this, execute, random.nextLong());
        }
        return executor;
    }
}
