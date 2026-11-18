package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.function.UnaryOperator;

/**
 * DINO-WM: a world model in a frozen pretrained feature space, planned with the cross-entropy
 * method (Zhou et al., ICML 2025, <a href="https://arxiv.org/abs/2411.04983">arXiv:2411.04983</a>).
 * Instead of learning an encoder together with the dynamics (which then needs reconstruction or
 * anti-collapse losses), the observation encoder is a frozen self-supervised network; only a
 * predictor is trained, on offline trajectories, to map the current features and the action to
 * the next frame's features. At test time a {@link CemPlanner} searches for action sequences
 * whose predicted features approach the features of a goal image, so new goals need no
 * retraining.
 *
 * <p>The paper's predictor is a causal ViT over DINOv2 patch features of several past frames;
 * here it is a residual MLP over the encoder's feature vector of the current frame
 * ({@code f' = f + MLP([f, a])}), which is the one-frame-history case. The encoder is any frozen
 * function from images to features, so different self-supervised encoders can be compared as the
 * feature space.
 */
public final class FeatureSpacePlanner {

    private final UnaryOperator<Tensor> encoder;
    private final int features;
    private final int actionDims;
    private final SimpleMlp predictor;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);

    private FeatureSpacePlanner(UnaryOperator<Tensor> encoder, int features, int actionDims, int hidden, long seed) {
        this.encoder = encoder;
        this.features = features;
        this.actionDims = actionDims;
        this.predictor = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), features + actionDims, hidden, hidden, features);
    }

    /**
     * Creates a planner around a frozen encoder.
     *
     * @param encoder images {@code [n, pixels]} to features {@code [n, features]}, never trained here
     * @param features the feature size
     * @param actionDims the action size
     * @param hidden the predictor width
     * @param seed the seed
     * @return the planner
     */
    public static FeatureSpacePlanner of(UnaryOperator<Tensor> encoder, int features, int actionDims, int hidden,
            long seed) {
        return new FeatureSpacePlanner(encoder, features, actionDims, hidden, seed);
    }

    private static Tensor rows(float[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows[i], 0, f, i * d, d);
        }
        return Tensor.of(Shape.of(n, d), f);
    }

    /**
     * Encodes one image.
     *
     * @param image the pixels
     * @return its features
     */
    public float[] encode(float[] image) {
        return Var.noGrad(() -> Var.of(encoder.apply(Tensor.of(Shape.of(1, image.length), image)))).value()
                .toArray();
    }

    private Var predict(Var f, Tensor actions) {
        return f.add(predictor.forward(Var.concat(1, f, Var.of(actions))));
    }

    /**
     * One predictor step on offline transitions (observations are images).
     *
     * @param batch the transitions
     * @return the feature prediction error
     */
    public float train(List<Transition> batch) {
        int n = batch.size();
        float[][] o = new float[n][];
        float[][] o2 = new float[n][];
        float[][] a = new float[n][];
        for (int i = 0; i < n; i++) {
            o[i] = batch.get(i).observation();
            o2[i] = batch.get(i).nextObservation();
            a[i] = batch.get(i).action();
        }
        Tensor f = Var.noGrad(() -> Var.of(encoder.apply(rows(o)))).value();
        Tensor f2 = Var.noGrad(() -> Var.of(encoder.apply(rows(o2)))).value();
        Var loss = predict(Var.of(f), rows(a)).sub(f2).square().mean();
        float v = loss.value().item();
        loss.backward();
        optimizer.step(predictor.parameters());
        return v;
    }

    /** @return the learned dynamics in feature space, for a {@link CemPlanner} */
    public CemPlanner.Dynamics dynamics() {
        return (obs, actions) -> Var.noGrad(() -> predict(Var.of(obs), actions)).value();
    }

    /**
     * A CEM planner towards a goal image: the reward of a feature vector is minus its squared
     * distance to the goal's features (summed over the horizon).
     *
     * @param goalImage the goal
     * @param actions the action box
     * @param horizon the planning horizon
     * @param samples action sequences per CEM round
     * @param elites the sequences kept per round
     * @param rounds CEM rounds
     * @param seed the seed
     * @return the planner; feed it {@link #encode encoded} observations
     */
    public CemPlanner planner(float[] goalImage, Space.Box actions, int horizon, int samples, int elites, int rounds,
            long seed) {
        float[] goal = encode(goalImage);
        CemPlanner.Reward reward = (f, a) -> {
            double d = 0;
            for (int i = 0; i < features; i++) {
                double e = f[i] - goal[i];
                d += e * e;
            }
            return -d;
        };
        return CemPlanner.of(dynamics(), reward, actions).horizon(horizon).search(samples, elites, rounds).seed(seed);
    }

    /** @return the action size */
    public int actionDims() {
        return actionDims;
    }
}
