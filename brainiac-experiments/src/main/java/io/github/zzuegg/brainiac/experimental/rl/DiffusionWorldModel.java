package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.VectorConditionedDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A diffusion world model in the spirit of DIAMOND (Alonso et al. 2024, "Diffusion for World
 * Modeling: Visual Details Matter in Atari", <a href="https://arxiv.org/abs/2405.12399">arXiv:2405.12399</a>):
 * the next observation is <em>sampled</em> by a conditional denoiser given the current
 * observation and the action, in a handful of denoising steps, instead of regressed to a
 * mean; rewards and terminations come from a separate model. DIAMOND denoises frames with a
 * U-Net and EDM preconditioning; here the observation change (scaled) is denoised by a
 * {@link VectorConditionedDenoiser} with clean-sample prediction (core {@link Diffusion}),
 * and the reward and termination model is an {@link MlpWorldModel} whose own next-state
 * guess is ignored.
 */
public final class DiffusionWorldModel implements DiscreteWorldModel {
    private static final int STEPS = 50;

    private final int observations;
    private final int actions;
    private final float scale;
    private final VectorConditionedDenoiser net;
    private final Diffusion diffusion;
    private final MlpWorldModel rewards;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private final SplittableRandom random;
    private int samplingSteps = 4;

    private DiffusionWorldModel(int observations, int actions, int hidden, float scale, long seed) {
        this.observations = observations;
        this.actions = actions;
        this.scale = scale;
        this.net = VectorConditionedDenoiser.of(observations, observations + actions, hidden, 3, seed)
                .timeScale(1f / STEPS);
        this.diffusion = new Diffusion(NoiseSchedule.cosine(STEPS), Diffusion.Prediction.SAMPLE).clipSample(3f);
        this.rewards = MlpWorldModel.of(observations, actions, hidden, seed + 1);
        this.random = new SplittableRandom(seed + 2);
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param hidden hidden width
     * @param scale the factor the observation change is multiplied by before denoising, so
     *     that it is of unit size
     * @param seed the seed
     * @return an untrained model
     */
    public static DiffusionWorldModel of(int observations, int actions, int hidden, float scale, long seed) {
        return new DiffusionWorldModel(observations, actions, hidden, scale, seed);
    }

    /**
     * @param steps denoising steps per imagined transition (DIAMOND uses 3)
     * @return this model
     */
    public DiffusionWorldModel samplingSteps(int steps) {
        this.samplingSteps = steps;
        return this;
    }

    private Tensor condition(Tensor obs, int[] acts) {
        int n = acts.length;
        float[] o = obs.toArray();
        float[] c = new float[n * (observations + actions)];
        for (int i = 0; i < n; i++) {
            System.arraycopy(o, i * observations, c, i * (observations + actions), observations);
            c[i * (observations + actions) + observations + acts[i]] = 1;
        }
        return Tensor.of(Shape.of(n, observations + actions), c);
    }

    @Override
    public float train(List<Transition> batch) {
        int n = batch.size();
        float[] obs = new float[n * observations];
        float[] delta = new float[n * observations];
        int[] acts = new int[n];
        for (int i = 0; i < n; i++) {
            Transition t = batch.get(i);
            acts[i] = Math.round(t.action()[0]);
            for (int k = 0; k < observations; k++) {
                obs[i * observations + k] = t.observation()[k];
                delta[i * observations + k] = (t.nextObservation()[k] - t.observation()[k]) * scale;
            }
        }
        Tensor cond = condition(Tensor.of(Shape.of(n, observations), obs), acts);
        Var loss = diffusion.loss(net.given(cond), Tensor.of(Shape.of(n, observations), delta), random);
        float v = loss.value().item();
        loss.backward();
        optimizer.step(net.parameters());
        return v + rewards.train(batch);
    }

    @Override
    public MlpWorldModel.Prediction predict(Tensor obs, int[] acts) {
        int n = acts.length;
        Tensor noise = Tensor.normal(Shape.of(n, observations), 0, 1, random);
        Tensor delta = diffusion.sample(net.given(condition(obs, acts)), noise, samplingSteps, 0, random);
        MlpWorldModel.Prediction r = rewards.predict(obs, acts);
        return new MlpWorldModel.Prediction(obs.add(delta.div(scale)), r.reward(), r.continuation());
    }
}
