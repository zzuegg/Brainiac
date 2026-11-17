package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A JEPA-style latent world model (Joint-Embedding Predictive Architecture, LeCun's
 * proposal; I-JEPA <a href="https://arxiv.org/abs/2301.08243">2301.08243</a>, action-conditioned
 * as in V-JEPA 2 <a href="https://arxiv.org/abs/2506.09985">2506.09985</a>): instead of
 * predicting the next observation, predict the next observation's <em>embedding</em>. An
 * online encoder embeds {@code s}, a predictor maps {@code (z, a)} to a guess of the target
 * encoder's embedding of {@code s'}, and the target encoder is an exponential moving average
 * of the online one (no gradient). Predicting in latent space lets the model ignore
 * unpredictable detail. To keep the embedding from collapsing to a constant, a VICReg-style
 * variance term (<a href="https://arxiv.org/abs/2105.04906">2105.04906</a>) keeps every
 * latent dimension's standard deviation near 1.
 */
public final class JepaWorldModel {

    private final int actions;
    private final Model encoder;
    private final Model target;
    private final Model predictor;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float ema = 0.99f;
    private float varianceWeight = 1f;

    private JepaWorldModel(int observations, int actions, int latent, int hidden, long seed) {
        this.actions = actions;
        this.encoder = Model.sequential().input(observations).dense(hidden, Activation.TANH).dense(latent).seed(seed)
                .build();
        this.target = Model.sequential().input(observations).dense(hidden, Activation.TANH).dense(latent).seed(seed)
                .build();
        this.predictor = AgentKit.mlp(latent + actions, latent, seed + 1, hidden);
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param latent embedding size
     * @param hidden hidden width
     * @param seed the seed
     * @return an untrained model
     */
    public static JepaWorldModel of(int observations, int actions, int latent, int hidden, long seed) {
        return new JepaWorldModel(observations, actions, latent, hidden, seed);
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public JepaWorldModel optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param decay the EMA decay of the target encoder
     * @return this model
     */
    public JepaWorldModel targetDecay(float decay) {
        this.ema = decay;
        return this;
    }

    private Var predictLatent(Var z, int[] acts) {
        float[] oneHot = new float[acts.length * actions];
        for (int i = 0; i < acts.length; i++) {
            oneHot[i * actions + acts[i]] = 1;
        }
        return predictor.forward(Var.concat(1, z, Var.of(Tensor.of(Shape.of(acts.length, actions), oneHot))), true);
    }

    /**
     * @param observations a batch of observations
     * @return their online embeddings
     */
    public Tensor embed(Tensor observations) {
        return encoder.forward(Var.of(observations), false).value();
    }

    /**
     * One training step.
     *
     * @param batch transitions
     * @return {prediction loss, mean latent standard deviation}
     */
    public float[] train(List<Transition> batch) {
        List<float[]> s = new ArrayList<>();
        List<float[]> next = new ArrayList<>();
        int[] acts = new int[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            s.add(batch.get(i).observation());
            next.add(batch.get(i).nextObservation());
            acts[i] = Math.round(batch.get(i).action()[0]);
        }
        Var z = encoder.forward(Var.of(AgentKit.rows(s)), true);
        Tensor zt = target.forward(Var.of(AgentKit.rows(next)), false).value();
        Var prediction = predictLatent(z, acts).sub(zt).square().mean();
        Var centered = z.sub(z.mean(0, true));
        Var std = centered.square().mean(0, true).add(1e-4f).sqrt();
        Var variance = std.neg().add(1f).relu().mean();
        Var loss = prediction.add(variance.mul(varianceWeight));
        float[] out = {prediction.value().item(), std.mean().value().item()};
        loss.backward();
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.addAll(predictor.parameters());
        optimizer.step(params);
        List<Var> online = encoder.parameters();
        List<Var> tgt = target.parameters();
        for (int i = 0; i < online.size(); i++) {
            tgt.get(i).assign(tgt.get(i).value().mul(ema).add(online.get(i).value().mul(1 - ema)));
        }
        return out;
    }

    /**
     * The prediction error on transitions, against the target encoder.
     *
     * @param batch transitions
     * @param useActionModel {@code false} measures the naive baseline that predicts no change
     *     ({@code z' = z}) instead
     * @return the mean squared error
     */
    public float error(List<Transition> batch, boolean useActionModel) {
        List<float[]> s = new ArrayList<>();
        List<float[]> next = new ArrayList<>();
        int[] acts = new int[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            s.add(batch.get(i).observation());
            next.add(batch.get(i).nextObservation());
            acts[i] = Math.round(batch.get(i).action()[0]);
        }
        Var z = encoder.forward(Var.of(AgentKit.rows(s)), false);
        Tensor zt = target.forward(Var.of(AgentKit.rows(next)), false).value();
        Var guess = useActionModel ? predictLatent(z, acts) : z;
        return guess.value().sub(zt).square().mean().item();
    }
}
