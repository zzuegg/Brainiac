package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.VectorConditionedDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Diffusion Policy (Chi et al., <a href="https://arxiv.org/abs/2303.04137">2303.04137</a>):
 * behaviour cloning where the policy is a conditional denoiser over a whole <em>action
 * chunk</em>. Given the observation it turns Gaussian noise into the next {@code H} actions
 * with a few deterministic DDIM steps (core {@link Diffusion}), and runs with receding
 * horizon: the first {@code execute} actions open loop, then a new chunk.
 *
 * <p>Why it matters: demonstrations are often multimodal (go left or right round an
 * obstacle). Regressing the mean action (MSE behaviour cloning) averages the modes into an
 * action no demonstrator took; a Gaussian head can only spread around that mean. A denoiser
 * models the whole action distribution and samples one mode, and chunking keeps it committed
 * to that mode across steps.
 *
 * <pre>{@code
 * DiffusionPolicy pi = DiffusionPolicy.of(2, 2, 8, 128, 2, 1).execute(4);
 * pi.train(ActionChunking.pairs(demos, 8), 1500, 64);
 * double success = env.successRate(pi, 50, 1);
 * }</pre>
 */
public final class DiffusionPolicy implements ChunkPolicy, Agent {
    private static final int STEPS = 50;
    private final int observationSize;
    private final int actionSize;
    private final int horizon;
    private final VectorConditionedDenoiser net;
    private Diffusion diffusion;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private final SplittableRandom random;
    private int samplingSteps = 10;
    private int execute;
    private Agent executor;

    private DiffusionPolicy(int observationSize, int actionSize, int horizon, int width, int depth, long seed) {
        this.observationSize = observationSize;
        this.actionSize = actionSize;
        this.horizon = horizon;
        this.net = VectorConditionedDenoiser.of(horizon * actionSize, observationSize, width, depth, seed)
                .timeScale(1f / STEPS);
        this.diffusion = new Diffusion(NoiseSchedule.cosine(STEPS), Diffusion.Prediction.SAMPLE).clipSample(1f);
        this.random = new SplittableRandom(seed + 1);
        this.execute = Math.max(1, horizon / 2);
    }

    /**
     * Creates a policy.
     *
     * @param observationSize the observation size
     * @param actionSize the action size (actions live in [-1, 1])
     * @param horizon the chunk length H
     * @param width the denoiser width
     * @param depth the denoiser's residual layers
     * @param seed the seed
     * @return the policy
     */
    public static DiffusionPolicy of(int observationSize, int actionSize, int horizon, int width, int depth,
            long seed) {
        return new DiffusionPolicy(observationSize, actionSize, horizon, width, depth, seed);
    }

    /**
     * Sets the training target. The default is the clean chunk (x-prediction): at the small
     * training budgets of the toy tasks it beats the paper's epsilon prediction by a wide
     * margin (0.9 against 0.6 success on {@link MultimodalReach} after 1500 steps).
     *
     * @param prediction the target
     * @return this policy
     */
    public DiffusionPolicy prediction(Diffusion.Prediction prediction) {
        this.diffusion = new Diffusion(NoiseSchedule.cosine(STEPS), prediction).clipSample(1f);
        return this;
    }

    /**
     * Sets how many DDIM steps sampling takes.
     *
     * @param steps 1 to 50
     * @return this policy
     */
    public DiffusionPolicy samplingSteps(int steps) {
        this.samplingSteps = steps;
        return this;
    }

    /**
     * Sets how many actions of each chunk run before replanning.
     *
     * @param steps 1 to the horizon
     * @return this policy
     */
    public DiffusionPolicy execute(int steps) {
        this.execute = steps;
        this.executor = null;
        return this;
    }

    /**
     * Trains on (observation, chunk) pairs with the epsilon-prediction loss.
     *
     * @param pairs the demonstrations, cut into chunks of this policy's horizon
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the last loss
     */
    public float train(ActionChunking.Pairs pairs, int steps, int batch) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(pairs, batch, random);
            Var loss = diffusion.loss(net.given(b[0]), b[1], random);
            last = loss.value().item();
            loss.backward();
            RlTensors.clip(net.parameters(), 1f);
            optimizer.step(net.parameters());
        }
        return last;
    }

    static Tensor[] batch(ActionChunking.Pairs pairs, int batch, SplittableRandom random) {
        int o = pairs.observations()[0].length;
        int c = pairs.chunks()[0].length;
        float[] obs = new float[batch * o];
        float[] chunks = new float[batch * c];
        for (int i = 0; i < batch; i++) {
            int k = random.nextInt(pairs.size());
            System.arraycopy(pairs.observations()[k], 0, obs, i * o, o);
            System.arraycopy(pairs.chunks()[k], 0, chunks, i * c, c);
        }
        return new Tensor[] {Tensor.of(Shape.of(batch, o), obs), Tensor.of(Shape.of(batch, c), chunks)};
    }

    @Override
    public float[][] chunk(float[] observation, SplittableRandom random) {
        Tensor noise = Tensor.normal(Shape.of(1, horizon * actionSize), 0, 1, random);
        Tensor x = diffusion.sample(net.given(VectorConditionedDenoiser.repeat(observation, 1)), noise, samplingSteps,
                0, random);
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

    /** @return the denoiser */
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
