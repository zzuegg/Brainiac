package io.github.zzuegg.brainiac.experiments.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.deeprl.PixelObservations;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.experimental.rl.DiffusionWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.DiscreteWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.DreamerAgent;
import io.github.zzuegg.brainiac.experimental.rl.KeyDoorGrid;
import io.github.zzuegg.brainiac.experimental.rl.MlpWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.vision.FrameMetrics;
import io.github.zzuegg.brainiac.experimental.vision.SimVp;
import io.github.zzuegg.brainiac.experimental.world.FeatureSpacePlanner;
import io.github.zzuegg.brainiac.experimental.world.SlotDynamics;
import io.github.zzuegg.brainiac.experimental.world.TokenWorldModel;
import io.github.zzuegg.brainiac.experimental.world.VideoMetrics;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.MountainCar;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 items 9.5.5-9.5.8: world models compared at equal real-environment experience.
 *
 * <ul>
 *   <li>{@link #tokenWorldModel} (9.5.5): IRIS's {@link TokenWorldModel} against the Markov
 *       {@link MlpWorldModel} and {@link DiffusionWorldModel} (DIAMOND) on {@link KeyDoorGrid},
 *       where the key is only visible for the first two steps: 10-step imagined rollouts under
 *       the real actions (token and cell accuracy, reward error), and the return of
 *       random-shooting MPC in each model, against {@code DreamerAgent} trained on the same
 *       number of real steps and a random policy.
 *   <li>{@link #dreamerTricks} (9.5.6): {@code DreamerAgent} with and without DreamerV3's
 *       symlog targets, two-hot critic, percentile return normalisation and unimix, on
 *       cart-pole with rewards scaled by 0.01, 1 and 100, one fixed configuration for all.
 *   <li>{@link #goalReaching} (9.5.7): {@link FeatureSpacePlanner} (DINO-WM) on a point-mass
 *       reacher rendered to 16x16 images, with frozen MAE-style, SimCLR-style and random
 *       encoders, a pixel-space world model with the same CEM planner, and goal-conditioned
 *       behaviour cloning; unseen goal images.
 *   <li>{@link #slotVideo} (9.5.8): {@link SlotDynamics} (SlotFormer) against its per-slot MLP
 *       ablation and SimVP rolled out autoregressively, on bouncing tiles, tested with more
 *       objects than in training.
 *   <li>{@link #pixels} (11.6.17): {@code DreamerAgent} and DQN on cart-pole from states and
 *       from 32x32 (or smaller) renders with the previous frame as a second channel
 *       ({@link PixelObservations#cartPole}), and on the pixel mountain car, at equal steps.
 * </ul>
 */
public final class WorldModelShootout {

    private WorldModelShootout() {
    }

    // ---------------------------------------------------------------- 9.5.5 token world model

    /**
     * A 9.5.5 row.
     *
     * @param model the world model or agent
     * @param tokenAccuracy imagined frame-token accuracy over 10 steps ({@code NaN} if not tokenized)
     * @param cellAccuracy imagined frame-cell accuracy over 10 steps
     * @param rewardError mean absolute reward error over 10 imagined steps
     * @param meanReturn the mean real return when acting with the model
     */
    public record TokenRow(String model, double tokenAccuracy, double cellAccuracy, double rewardError,
            double meanReturn) {
    }

    private static List<List<Transition>> randomEpisodes(Environment env, int episodes, SplittableRandom r) {
        List<List<Transition>> out = new ArrayList<>();
        for (int e = 0; e < episodes; e++) {
            List<Transition> ep = new ArrayList<>();
            float[] obs = env.reset(r.split());
            boolean done = false;
            while (!done) {
                float[] a = {r.nextInt(4)};
                Environment.Step s = env.step(a);
                ep.add(new Transition(obs, a, s.reward(), s.observation(), s.terminated(), 0.95f));
                obs = s.observation();
                done = s.done();
            }
            out.add(ep);
        }
        return out;
    }

    private static TokenWorldModel.Segment segment(List<Transition> ep, int start, int n) {
        int len = Math.min(n, ep.size() - start);
        float[][] obs = new float[len + 1][];
        int[] acts = new int[len];
        float[] rew = new float[len];
        boolean[] done = new boolean[len];
        for (int t = 0; t < len; t++) {
            Transition tr = ep.get(start + t);
            obs[t] = tr.observation();
            acts[t] = Math.round(tr.action()[0]);
            rew[t] = tr.reward();
            done[t] = tr.terminated();
            obs[t + 1] = tr.nextObservation();
        }
        return new TokenWorldModel.Segment(obs, acts, rew, done);
    }

    private static double cellAccuracy(float[] predicted, float[] truth) {
        int hit = 0;
        for (int i = 0; i < truth.length; i++) {
            hit += (predicted[i] > 0.5f) == (truth[i] > 0.5f) ? 1 : 0;
        }
        return hit / (double) truth.length;
    }

    /**
     * 9.5.5: the token world model against Markov world models and Dreamer.
     *
     * @param episodes random episodes of experience
     * @param tokenizerSteps frame-autoencoder steps
     * @param dynamicsSteps world-model steps (every model)
     * @param evalEpisodes episodes for the rollout and return scores
     * @param seed the seed
     * @return one row per model
     */
    public static List<TokenRow> tokenWorldModel(int episodes, int tokenizerSteps, int dynamicsSteps, int evalEpisodes,
            long seed) {
        KeyDoorGrid env = new KeyDoorGrid(4, 20, 2);
        int obsSize = env.observationSize();
        SplittableRandom r = new SplittableRandom(seed);
        List<List<Transition>> data = randomEpisodes(env, episodes, r);
        List<Transition> flat = new ArrayList<>();
        List<float[]> frames = new ArrayList<>();
        data.forEach(flat::addAll);
        for (Transition t : flat) {
            frames.add(t.observation());
        }
        int context = 8;
        TokenWorldModel iris = TokenWorldModel.of(obsSize, 4, 3, 32, 32, 2, 2, context, seed);
        for (int s = 0; s < tokenizerSteps; s++) {
            List<float[]> b = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                b.add(frames.get(r.nextInt(frames.size())));
            }
            iris.trainTokenizer(b);
            if (s > 0 && s % 50 == 0) {
                iris.restartDeadCodes(b);
            }
        }
        for (int s = 0; s < dynamicsSteps; s++) {
            List<TokenWorldModel.Segment> b = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                List<Transition> ep = data.get(r.nextInt(data.size()));
                b.add(segment(ep, r.nextInt(ep.size()), context));
            }
            iris.trainDynamics(b);
        }
        MlpWorldModel mlp = MlpWorldModel.of(obsSize, 4, 64, seed);
        DiffusionWorldModel diamond = DiffusionWorldModel.of(obsSize, 4, 64, 1f, seed);
        for (int s = 0; s < dynamicsSteps; s++) {
            List<Transition> b = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                b.add(flat.get(r.nextInt(flat.size())));
            }
            mlp.train(b);
            diamond.train(b);
        }
        List<List<Transition>> test = new ArrayList<>();
        SplittableRandom tr = new SplittableRandom(seed + 99);
        while (test.size() < evalEpisodes) {
            for (List<Transition> ep : randomEpisodes(env, 1, tr)) {
                if (ep.size() >= 10) {
                    test.add(ep);
                }
            }
        }
        double[] irisScore = new double[3];
        double[] mlpScore = new double[3];
        double[] diamondScore = new double[3];
        for (List<Transition> ep : test) {
            int[][] plan = new int[1][10];
            for (int t = 0; t < 10; t++) {
                plan[0][t] = Math.round(ep.get(t).action()[0]);
            }
            int[] hist = iris.history(List.of(ep.get(0).observation()), List.of());
            TokenWorldModel.Imagination im = iris.imagine(hist, plan, true, r);
            List<float[]> realNext = new ArrayList<>();
            for (int t = 0; t < 10; t++) {
                realNext.add(ep.get(t).nextObservation());
            }
            int[][] realTokens = iris.tokenize(realNext);
            float[][] decoded = iris.decode(im.frames()[0]);
            for (int t = 0; t < 10; t++) {
                int same = 0;
                for (int k = 0; k < realTokens[t].length; k++) {
                    same += realTokens[t][k] == im.frames()[0][t][k] ? 1 : 0;
                }
                irisScore[0] += same / (double) realTokens[t].length / (10.0 * test.size());
                irisScore[1] += cellAccuracy(decoded[t], realNext.get(t)) / (10.0 * test.size());
                irisScore[2] += Math.abs(im.rewards()[0][t] - ep.get(t).reward()) / (10.0 * test.size());
            }
            markovRollout(mlp, ep, mlpScore, test.size());
            markovRollout(diamond, ep, diamondScore, test.size());
        }
        List<TokenRow> rows = new ArrayList<>();
        rows.add(new TokenRow("IRIS token world model", irisScore[0], irisScore[1], irisScore[2],
                irisReturn(iris, env, evalEpisodes, seed)));
        rows.add(new TokenRow("MLP world model", Double.NaN, mlpScore[1], mlpScore[2],
                markovReturn(mlp, env, evalEpisodes, seed)));
        rows.add(new TokenRow("DIAMOND diffusion world model", Double.NaN, diamondScore[1], diamondScore[2],
                markovReturn(diamond, env, evalEpisodes, seed)));
        DreamerAgent dreamer = DreamerAgent.of(obsSize, 4, 64, seed).schedule(Math.max(50, flat.size() / 4), 60, 30)
                .horizon(10);
        dreamer.fit(env, flat.size(), null);
        rows.add(new TokenRow("DreamerAgent (same real steps)", Double.NaN, Double.NaN, Double.NaN,
                Rl.evaluate(dreamer, env, evalEpisodes, seed)));
        double random = 0;
        for (List<Transition> ep : randomEpisodes(env, evalEpisodes, new SplittableRandom(seed + 5))) {
            for (Transition t : ep) {
                random += t.reward() / evalEpisodes;
            }
        }
        rows.add(new TokenRow("random policy", Double.NaN, Double.NaN, Double.NaN, random));
        return rows;
    }

    private static void markovRollout(DiscreteWorldModel model, List<Transition> ep, double[] score, int episodes) {
        Tensor obs = Tensor.of(Shape.of(1, ep.get(0).observation().length), ep.get(0).observation());
        for (int t = 0; t < 10; t++) {
            MlpWorldModel.Prediction p = model.predict(obs, new int[] {Math.round(ep.get(t).action()[0])});
            score[1] += cellAccuracy(p.next().toArray(), ep.get(t).nextObservation()) / (10.0 * episodes);
            score[2] += Math.abs(p.reward().toArray()[0] - ep.get(t).reward()) / (10.0 * episodes);
            obs = p.next();
        }
    }

    private static double irisReturn(TokenWorldModel model, KeyDoorGrid env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 7);
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            List<float[]> seen = new ArrayList<>();
            List<Integer> acts = new ArrayList<>();
            seen.add(env.reset(r.split()));
            boolean done = false;
            while (!done) {
                int from = Math.max(0, seen.size() - model.contextSteps() + 6);
                int[] hist = model.history(seen.subList(from, seen.size()), acts.subList(from, acts.size()));
                int a = model.plan(hist, 32, 6, 0.95f, r);
                Environment.Step s = env.step(new float[] {a});
                total += s.reward() / episodes;
                seen.add(s.observation());
                acts.add(a);
                done = s.done();
            }
        }
        return total;
    }

    private static double markovReturn(DiscreteWorldModel model, KeyDoorGrid env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 7);
        double total = 0;
        int n = 32;
        int h = 6;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(r.split());
            boolean done = false;
            while (!done) {
                int[][] plans = new int[n][h];
                float[] start = new float[n * obs.length];
                for (int i = 0; i < n; i++) {
                    System.arraycopy(obs, 0, start, i * obs.length, obs.length);
                    for (int t = 0; t < h; t++) {
                        plans[i][t] = r.nextInt(4);
                    }
                }
                Tensor x = Tensor.of(Shape.of(n, obs.length), start);
                double[] value = new double[n];
                double[] alive = new double[n];
                java.util.Arrays.fill(alive, 1);
                double discount = 1;
                for (int t = 0; t < h; t++) {
                    int[] a = new int[n];
                    for (int i = 0; i < n; i++) {
                        a[i] = plans[i][t];
                    }
                    MlpWorldModel.Prediction p = model.predict(x, a);
                    float[] rew = p.reward().toArray();
                    float[] cont = p.continuation().toArray();
                    for (int i = 0; i < n; i++) {
                        value[i] += discount * alive[i] * rew[i];
                        alive[i] *= cont[i];
                    }
                    discount *= 0.95;
                    x = p.next();
                }
                int best = 0;
                for (int i = 1; i < n; i++) {
                    if (value[i] > value[best]) {
                        best = i;
                    }
                }
                Environment.Step s = env.step(new float[] {plans[best][0]});
                total += s.reward() / episodes;
                obs = s.observation();
                done = s.done();
            }
        }
        return total;
    }

    // ---------------------------------------------------------------- 9.5.6 DreamerV3 tricks

    /** Cart-pole with every reward multiplied by a constant. */
    private static final class ScaledRewards implements Environment {
        private final Environment inner;
        private final float scale;

        ScaledRewards(Environment inner, float scale) {
            this.inner = inner;
            this.scale = scale;
        }

        @Override
        public Space observationSpace() {
            return inner.observationSpace();
        }

        @Override
        public Space actionSpace() {
            return inner.actionSpace();
        }

        @Override
        public float[] reset(SplittableRandom random) {
            return inner.reset(random);
        }

        @Override
        public Step step(float[] action) {
            Step s = inner.step(action);
            return new Step(s.observation(), s.reward() * scale, s.terminated(), s.truncated());
        }
    }

    /**
     * A 9.5.6 row.
     *
     * @param config the configuration
     * @param scale the reward scale
     * @param meanReturn the greedy return after training, in unscaled units (cart-pole steps)
     */
    public record TrickRow(String config, float scale, double meanReturn) {
    }

    /** The configurations of 9.5.6. */
    public enum Tricks {
        /** The repo's simplified Dreamer. */
        DEFAULT,
        /** Symlog, two-hot critic, percentile normalisation and unimix. */
        ALL,
        /** All but symlog targets. */
        NO_SYMLOG,
        /** All but the two-hot critic. */
        NO_TWO_HOT,
        /** All but percentile normalisation. */
        NO_PERCENTILE,
        /** All but unimix. */
        NO_UNIMIX
    }

    private static DreamerAgent configure(Tricks t, long seed) {
        DreamerAgent a = DreamerAgent.of(4, 2, 64, seed).schedule(250, 80, 40).horizon(15);
        if (t == Tricks.DEFAULT) {
            return a;
        }
        a.symlog(t != Tricks.NO_SYMLOG).percentileNormalisation(t != Tricks.NO_PERCENTILE)
                .unimix(t == Tricks.NO_UNIMIX ? 0 : 0.01f);
        if (t != Tricks.NO_TWO_HOT) {
            a.twoHotCritic(41, -20, 20, 64, seed + 2);
        }
        return a;
    }

    /**
     * 9.5.6: Dreamer configurations across reward scales, one fixed configuration for all scales.
     *
     * @param configs the configurations
     * @param scales the reward scales
     * @param steps real environment steps per run
     * @param seed the seed
     * @return one row per configuration and scale
     */
    public static List<TrickRow> dreamerTricks(Tricks[] configs, float[] scales, long steps, long seed) {
        List<TrickRow> rows = new ArrayList<>();
        for (Tricks t : configs) {
            for (float scale : scales) {
                Environment env = new ScaledRewards(new CartPole(200), scale);
                DreamerAgent agent = configure(t, seed);
                agent.fit(env, steps, null);
                rows.add(new TrickRow(t.toString().toLowerCase(java.util.Locale.ROOT), scale,
                        Rl.evaluate(agent, env, 5, seed) / scale));
            }
        }
        return rows;
    }

    // ---------------------------------------------------------------- 9.5.7 feature-space planning

    private static final int REACH_SIDE = 16;
    private static final float REACH_SPEED = 0.12f;

    private static float[] render(float x, float y) {
        float[] img = new float[REACH_SIDE * REACH_SIDE];
        for (int j = 0; j < REACH_SIDE; j++) {
            for (int i = 0; i < REACH_SIDE; i++) {
                double dx = i - x * (REACH_SIDE - 1);
                double dy = j - y * (REACH_SIDE - 1);
                img[j * REACH_SIDE + i] = (float) Math.exp(-(dx * dx + dy * dy) / (2 * 1.2 * 1.2));
            }
        }
        return img;
    }

    private static float[] move(float[] pos, float[] a) {
        return new float[] {Math.max(0, Math.min(1, pos[0] + REACH_SPEED * Math.max(-1, Math.min(1, a[0])))),
            Math.max(0, Math.min(1, pos[1] + REACH_SPEED * Math.max(-1, Math.min(1, a[1]))))};
    }

    /**
     * A 9.5.7 row.
     *
     * @param method the planner and feature space
     * @param success the share of episodes ending within 0.1 of the goal
     * @param finalDistance the mean final distance to the goal
     */
    public record ReachRow(String method, double success, double finalDistance) {
    }

    private static Tensor rows(List<float[]> rows) {
        int n = rows.size();
        int d = rows.get(0).length;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows.get(i), 0, f, i * d, d);
        }
        return Tensor.of(Shape.of(n, d), f);
    }

    private static float[] shifted(float[] img, int dx, int dy, SplittableRandom r) {
        float[] out = new float[img.length];
        for (int j = 0; j < REACH_SIDE; j++) {
            for (int i = 0; i < REACH_SIDE; i++) {
                int si = Math.floorMod(i - dx, REACH_SIDE);
                int sj = Math.floorMod(j - dy, REACH_SIDE);
                out[j * REACH_SIDE + i] = img[sj * REACH_SIDE + si] + (float) (0.05 * r.nextGaussian());
            }
        }
        return out;
    }

    /** An MAE-style encoder: reconstruct the whole frame from a frame with half its pixels hidden. */
    private static SimpleMlp maeEncoder(List<float[]> frames, int features, int steps, long seed) {
        SimpleMlp enc = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), REACH_SIDE * REACH_SIDE, 64, features);
        SimpleMlp dec = SimpleMlp.of(seed + 1, v -> v.mul(v.sigmoid()), features, 64, REACH_SIDE * REACH_SIDE);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            List<float[]> clean = new ArrayList<>();
            List<float[]> masked = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                float[] f = frames.get(r.nextInt(frames.size()));
                float[] m = f.clone();
                for (int p = 0; p < m.length; p++) {
                    if (r.nextBoolean()) {
                        m[p] = 0;
                    }
                }
                clean.add(f);
                masked.add(m);
            }
            Var loss = dec.forward(enc.forward(Var.of(rows(masked)))).sub(rows(clean)).square().mean();
            loss.backward();
            List<Var> p = new ArrayList<>(enc.parameters());
            p.addAll(dec.parameters());
            opt.step(p);
        }
        return enc;
    }

    /** A SimCLR-style encoder: two shifted, noisy views of a frame agree under InfoNCE. */
    private static SimpleMlp simclrEncoder(List<float[]> frames, int features, int steps, long seed) {
        SimpleMlp enc = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), REACH_SIDE * REACH_SIDE, 64, features);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            List<float[]> a = new ArrayList<>();
            List<float[]> b = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                float[] f = frames.get(r.nextInt(frames.size()));
                a.add(shifted(f, r.nextInt(5) - 2, r.nextInt(5) - 2, r));
                b.add(shifted(f, r.nextInt(5) - 2, r.nextInt(5) - 2, r));
            }
            Var loss = Functional.infoNce(enc.forward(Var.of(rows(a))), enc.forward(Var.of(rows(b))), 0.2f);
            loss.backward();
            opt.step(enc.parameters());
        }
        return enc;
    }

    private static java.util.function.UnaryOperator<Tensor> frozen(SimpleMlp enc) {
        return x -> Var.noGrad(() -> enc.forward(Var.of(x))).value();
    }

    /**
     * 9.5.7: goal reaching from images with frozen feature spaces and CEM, against a pixel-space
     * model and goal-conditioned behaviour cloning, all from the same offline random-play data.
     *
     * @param transitions offline random transitions
     * @param pretrainSteps encoder pretraining steps
     * @param predictorSteps predictor (and behaviour-cloning) steps
     * @param episodes test episodes with unseen goals
     * @param seed the seed
     * @return one row per method
     */
    public static List<ReachRow> goalReaching(int transitions, int pretrainSteps, int predictorSteps, int episodes,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Transition> data = new ArrayList<>();
        List<List<OfflineDataset.Step>> trajectories = new ArrayList<>();
        List<float[]> frames = new ArrayList<>();
        while (data.size() < transitions) {
            float[] pos = {r.nextFloat(), r.nextFloat()};
            List<float[]> imgs = new ArrayList<>();
            List<float[]> acts = new ArrayList<>();
            imgs.add(render(pos[0], pos[1]));
            for (int t = 0; t < 10; t++) {
                float[] a = {2 * r.nextFloat() - 1, 2 * r.nextFloat() - 1};
                pos = move(pos, a);
                acts.add(a);
                imgs.add(render(pos[0], pos[1]));
                data.add(new Transition(imgs.get(t), a, 0, imgs.get(t + 1), false, 1f));
                frames.add(imgs.get(t));
            }
            List<OfflineDataset.Step> traj = new ArrayList<>();
            for (int t = 0; t < 10; t++) {
                int g = t + 1 + r.nextInt(10 - t);
                float[] obs = new float[2 * imgs.get(t).length];
                System.arraycopy(imgs.get(t), 0, obs, 0, imgs.get(t).length);
                System.arraycopy(imgs.get(g), 0, obs, imgs.get(t).length, imgs.get(t).length);
                traj.add(new OfflineDataset.Step(obs, acts.get(t), 0, obs, false));
            }
            trajectories.add(traj);
        }
        int features = 16;
        List<ReachRow> rows = new ArrayList<>();
        java.util.Map<String, java.util.function.UnaryOperator<Tensor>> encoders = new java.util.LinkedHashMap<>();
        encoders.put("DINO-WM, frozen MAE-style features", frozen(maeEncoder(frames, features, pretrainSteps, seed)));
        encoders.put("DINO-WM, frozen SimCLR-style features", frozen(simclrEncoder(frames, features, pretrainSteps,
                seed)));
        encoders.put("DINO-WM, frozen random features", frozen(SimpleMlp.of(seed + 3, v -> v.mul(v.sigmoid()),
                REACH_SIDE * REACH_SIDE, 64, features)));
        encoders.put("pixel-space world model + CEM", x -> x);
        Space.Box box = Space.box(2, -1, 1);
        for (var e : encoders.entrySet()) {
            int f = e.getKey().startsWith("pixel") ? REACH_SIDE * REACH_SIDE : features;
            FeatureSpacePlanner planner = FeatureSpacePlanner.of(e.getValue(), f, 2, 64, seed);
            for (int s = 0; s < predictorSteps; s++) {
                List<Transition> b = new ArrayList<>();
                for (int i = 0; i < 64; i++) {
                    b.add(data.get(r.nextInt(data.size())));
                }
                planner.train(b);
            }
            rows.add(reach(e.getKey(), episodes, seed, (img, goal, pos) -> {
                CemPlanner cem = planner.planner(goal, box, 5, 64, 8, 3, seed);
                return obs -> cem.act(planner.encode(obs), false);
            }));
        }
        Model policy = Model.sequential().input(2 * REACH_SIDE * REACH_SIDE).dense(64, Activation.RELU)
                .dense(64, Activation.RELU).dense(2, Activation.TANH).seed(seed).build();
        BehaviorCloning bc = BehaviorCloning.continuous(policy).seed(seed).train(OfflineDataset.of(trajectories),
                predictorSteps);
        rows.add(reach("goal-conditioned behaviour cloning", episodes, seed, (img, goal, pos) -> obs -> {
            float[] in = new float[2 * obs.length];
            System.arraycopy(obs, 0, in, 0, obs.length);
            System.arraycopy(goal, 0, in, obs.length, obs.length);
            return bc.act(in, false);
        }));
        return rows;
    }

    /** Makes a per-episode policy from the start image, the goal image and the start position. */
    private interface EpisodePolicy {
        java.util.function.Function<float[], float[]> start(float[] image, float[] goal, float[] position);
    }

    private static ReachRow reach(String name, int episodes, long seed, EpisodePolicy policy) {
        SplittableRandom r = new SplittableRandom(seed + 1234);
        int success = 0;
        double dist = 0;
        for (int e = 0; e < episodes; e++) {
            float[] pos = {r.nextFloat(), r.nextFloat()};
            float[] goal = {r.nextFloat(), r.nextFloat()};
            float[] goalImg = render(goal[0], goal[1]);
            float[] img = render(pos[0], pos[1]);
            java.util.function.Function<float[], float[]> act = policy.start(img, goalImg, pos);
            for (int t = 0; t < 12; t++) {
                pos = move(pos, act.apply(img));
                img = render(pos[0], pos[1]);
            }
            double d = Math.hypot(pos[0] - goal[0], pos[1] - goal[1]);
            dist += d / episodes;
            success += d < 0.1 ? 1 : 0;
        }
        return new ReachRow(name, success / (double) episodes, dist);
    }

    // ---------------------------------------------------------------- 9.5.8 slot dynamics

    /**
     * A 9.5.8 row.
     *
     * @param model the model
     * @param psnr10 PSNR of the 10th predicted frame
     * @param psnr20 PSNR of the 20th predicted frame
     * @param position10 object-position error at frame 10 (pixels)
     * @param position20 object-position error at frame 20
     * @param fgAri FG-ARI of the predicted masks at frame 10 ({@code NaN} for pixel models)
     */
    public record SlotRow(String model, double psnr10, double psnr20, double position10, double position20,
            double fgAri) {
    }

    private static final int SLOT_SIDE = 16;
    private static final int SLOT_CONTEXT = 4;

    private static ShapeScenes.Video[] videos(int n, int length, int minObjects, int maxObjects, long seed) {
        ShapeScenes gen = new ShapeScenes(SLOT_SIDE, 0f);
        SplittableRandom r = new SplittableRandom(seed);
        ShapeScenes.Video[] v = new ShapeScenes.Video[n];
        for (int i = 0; i < n; i++) {
            v[i] = gen.video(length, minObjects + r.nextInt(maxObjects - minObjects + 1), 4, 0.6f, 1.2f, false, r);
        }
        return v;
    }

    private static Tensor tensor(ShapeScenes.Video[] v, int from, int to) {
        int p = SLOT_SIDE * SLOT_SIDE;
        int t = to - from;
        float[] out = new float[v.length * t * p];
        for (int i = 0; i < v.length; i++) {
            for (int f = 0; f < t; f++) {
                System.arraycopy(v[i].frames()[from + f], 0, out, (i * t + f) * p, p);
            }
        }
        return Tensor.of(Shape.of(v.length, t, SLOT_SIDE, SLOT_SIDE), out);
    }

    private static double positionError(float[] frame, ShapeScenes.Video v, int t) {
        double[][] blobs = VideoMetrics.blobs(frame, SLOT_SIDE, 0.4f, 3);
        double err = 0;
        for (int o = 0; o < v.objects(); o++) {
            float[] b = java.util.Arrays.copyOfRange(v.boxes()[t], 4 * o, 4 * o + 4);
            double cx = (b[0] + b[2]) / 2 - 0.5;
            double cy = (b[1] + b[3]) / 2 - 0.5;
            double best = SLOT_SIDE;
            for (double[] q : blobs) {
                best = Math.min(best, Math.hypot(q[0] - cx, q[1] - cy));
            }
            err += best / v.objects();
        }
        return err;
    }

    /** Adjusted Rand index between two labelings of the same pixels. */
    static double ari(int[] a, int[] b) {
        int ka = 0;
        int kb = 0;
        for (int i = 0; i < a.length; i++) {
            ka = Math.max(ka, a[i] + 1);
            kb = Math.max(kb, b[i] + 1);
        }
        long[][] m = new long[ka][kb];
        for (int i = 0; i < a.length; i++) {
            m[a[i]][b[i]]++;
        }
        double sumIj = 0;
        double[] rowSum = new double[ka];
        double[] colSum = new double[kb];
        for (int i = 0; i < ka; i++) {
            for (int j = 0; j < kb; j++) {
                sumIj += m[i][j] * (m[i][j] - 1) / 2.0;
                rowSum[i] += m[i][j];
                colSum[j] += m[i][j];
            }
        }
        double sa = 0;
        double sb = 0;
        for (double x : rowSum) {
            sa += x * (x - 1) / 2;
        }
        for (double x : colSum) {
            sb += x * (x - 1) / 2;
        }
        double n = a.length;
        double expected = sa * sb / (n * (n - 1) / 2);
        double max = (sa + sb) / 2;
        return max == expected ? 1 : (sumIj - expected) / (max - expected);
    }

    private static double fgAri(float[] masks, int slots, ShapeScenes.Video v, int t) {
        int p = SLOT_SIDE * SLOT_SIDE;
        List<Integer> truth = new ArrayList<>();
        List<Integer> pred = new ArrayList<>();
        for (int y = 0; y < SLOT_SIDE; y++) {
            for (int x = 0; x < SLOT_SIDE; x++) {
                int label = -1;
                for (int o = 0; o < v.objects(); o++) {
                    float[] b = java.util.Arrays.copyOfRange(v.boxes()[t], 4 * o, 4 * o + 4);
                    if (x + 0.5 >= b[0] && x + 0.5 < b[2] && y + 0.5 >= b[1] && y + 0.5 < b[3]) {
                        label = o;
                    }
                }
                if (label < 0) {
                    continue;
                }
                int i = y * SLOT_SIDE + x;
                int best = 0;
                for (int k = 1; k < slots; k++) {
                    if (masks[k * p + i] > masks[best * p + i]) {
                        best = k;
                    }
                }
                truth.add(label);
                pred.add(best);
            }
        }
        if (truth.size() < 2) {
            return 1;
        }
        return ari(truth.stream().mapToInt(Integer::intValue).toArray(),
                pred.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * 9.5.8: SlotFormer against its no-interaction ablation and SimVP, trained on 2-3 objects
     * and tested on 4, 20 frames ahead of 4 context frames.
     *
     * @param autoencoderSteps slot autoencoder steps
     * @param dynamicsSteps dynamics steps (and SimVP steps)
     * @param test test videos
     * @param seed the seed
     * @return one row per model
     */
    public static List<SlotRow> slotVideo(int autoencoderSteps, int dynamicsSteps, int test, long seed) {
        int horizon = 20;
        int rollout = 4;
        SplittableRandom r = new SplittableRandom(seed);
        ShapeScenes.Video[] testVideos = videos(test, SLOT_CONTEXT + horizon, 4, 4, seed + 99);
        Tensor context = tensor(testVideos, 0, SLOT_CONTEXT);
        List<SlotRow> rows = new ArrayList<>();
        SlotDynamics base = null;
        for (SlotDynamics.Dynamics kind : SlotDynamics.Dynamics.values()) {
            SlotDynamics m = SlotDynamics.of(SLOT_SIDE, 5, 32, SLOT_CONTEXT, kind, seed);
            for (int s = 0; s < autoencoderSteps; s++) {
                m.trainAutoencoder(tensor(videos(8, 3, 2, 3, r.nextLong()), 0, 3));
            }
            for (int s = 0; s < dynamicsSteps; s++) {
                m.trainDynamics(tensor(videos(8, SLOT_CONTEXT + rollout, 2, 3, r.nextLong()), 0,
                        SLOT_CONTEXT + rollout), rollout);
            }
            base = m;
            Tensor[] out = m.rollout(context, horizon);
            rows.add(slotRow(kind == SlotDynamics.Dynamics.TRANSFORMER ? "SlotFormer" : "slots + per-slot MLP",
                    out[0].toArray(), out[1].toArray(), m.slots(), testVideos, horizon));
        }
        SimVp simvp = new SimVp(SLOT_CONTEXT, rollout, 8, 2, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < dynamicsSteps; s++) {
            Tensor v = tensor(videos(8, SLOT_CONTEXT + rollout, 2, 3, r.nextLong()), 0, SLOT_CONTEXT + rollout);
            simvp.forward(Var.of(v.slice(1, 0, SLOT_CONTEXT))).sub(v.slice(1, SLOT_CONTEXT, SLOT_CONTEXT + rollout))
                    .square().mean().backward();
            opt.step(simvp.parameters());
        }
        Tensor window = context;
        List<Tensor> outs = new ArrayList<>();
        int have = 0;
        while (have < horizon) {
            Tensor in = window;
            Tensor next = Var.noGrad(() -> simvp.forward(Var.of(in))).value();
            outs.add(next);
            have += rollout;
            window = Tensor.concat(1, window, next).slice(1, rollout, SLOT_CONTEXT + rollout);
        }
        Tensor pred = Tensor.concat(1, outs.toArray(new Tensor[0])).slice(1, 0, horizon);
        rows.add(slotRow("SimVP (autoregressive)", pred.toArray(), null, base == null ? 0 : base.slots(), testVideos,
                horizon));
        return rows;
    }

    private static SlotRow slotRow(String name, float[] frames, float[] masks, int slots, ShapeScenes.Video[] v,
            int horizon) {
        int p = SLOT_SIDE * SLOT_SIDE;
        double[] psnr = new double[2];
        double[] pos = new double[2];
        double ari = 0;
        int[] at = {9, 19};
        for (int i = 0; i < v.length; i++) {
            for (int k = 0; k < 2; k++) {
                int f = at[k];
                float[] frame = java.util.Arrays.copyOfRange(frames, (i * horizon + f) * p, (i * horizon + f + 1) * p);
                int t = SLOT_CONTEXT + f;
                psnr[k] += FrameMetrics.psnr(frame, v[i].frames()[t]) / v.length;
                pos[k] += positionError(frame, v[i], t) / v.length;
            }
            if (masks != null) {
                int f = at[0];
                float[] m = java.util.Arrays.copyOfRange(masks, (i * horizon + f) * slots * p,
                        (i * horizon + f + 1) * slots * p);
                ari += fgAri(m, slots, v[i], SLOT_CONTEXT + f) / v.length;
            }
        }
        return new SlotRow(name, psnr[0], psnr[1], pos[0], pos[1], masks == null ? Double.NaN : ari);
    }

    // ---------------------------------------------------------------- 11.6.17 control from pixels

    /**
     * An 11.6.17 row.
     *
     * @param task the task and its input
     * @param agent the learner
     * @param meanReturn the mean greedy return over seeds
     */
    public record PixelRow(String task, String agent, double meanReturn) {
    }

    private static Environment pixelTask(String task, boolean pixels, int size) {
        return switch (task) {
            case "cartpole" -> pixels ? PixelObservations.cartPole(new CartPole(200), size) : new CartPole(200);
            case "mountaincar" -> pixels ? PixelObservations.mountainCar(new MountainCar(200), size)
                    : new MountainCar(200);
            default -> throw new IllegalArgumentException(task);
        };
    }

    private static int observations(Environment env) {
        return ((Space.Box) env.observationSpace()).size();
    }

    /**
     * 11.6.17: the cost of learning from pixels, at equal environment steps.
     *
     * @param tasks "cartpole" and/or "mountaincar"
     * @param steps environment steps per learner
     * @param seeds seeds
     * @param size the render side in pixels (32 in the item; smaller for quick runs)
     * @return one row per task, input and agent
     */
    public static List<PixelRow> pixels(String[] tasks, long steps, int seeds, int size) {
        List<PixelRow> rows = new ArrayList<>();
        for (String task : tasks) {
            int actions = ((Space.Discrete) pixelTask(task, false, size).actionSpace()).n();
            for (boolean pix : new boolean[] {false, true}) {
                String name = task + (pix ? " (pixels 2x" + size + "x" + size + ")" : " (states)");
                int obs = observations(pixelTask(task, pix, size));
                double dreamer = 0;
                double dqn = 0;
                for (int s = 0; s < seeds; s++) {
                    Environment env = pixelTask(task, pix, size);
                    DreamerAgent d = DreamerAgent.of(obs, actions, 64, s).schedule(250, 80, 40).horizon(15);
                    d.fit(env, steps, null);
                    dreamer += Rl.evaluate(d, pixelTask(task, pix, size), 3, 100 + s);
                    long seed = s;
                    Learner q = Dqn.of(() -> Nets.mlp(obs, actions, Activation.RELU, seed, 64), actions)
                            .bufferSize((int) Math.min(steps, 20_000))
                            .learningStarts((int) Math.min(1_000, steps / 5))
                            .epsilon(Schedule.linear(1, 0.05, Math.max(1, steps / 2))).seed(s);
                    q.fit(pixelTask(task, pix, size), steps, null);
                    dqn += Rl.evaluate(q, pixelTask(task, pix, size), 3, 100 + s);
                }
                rows.add(new PixelRow(name, "Dreamer", dreamer / seeds));
                rows.add(new PixelRow(name, "DQN", dqn / seeds));
            }
        }
        return rows;
    }
}
