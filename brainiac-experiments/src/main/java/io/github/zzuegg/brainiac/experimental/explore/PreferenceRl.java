package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Reinforcement learning from segment comparisons: the agent never sees the reward, only a
 * teacher's answers to "which of these two trajectory segments was better?". An ensemble of
 * reward models {@code r(s, a)} is fitted to the answers with the Bradley-Terry loss of
 * {@link PreferenceLosses#bradleyTerry} on summed segment rewards, and an off-policy TD3
 * ({@link StepwiseTd3}) learns from the predicted reward.
 *
 * <ul>
 *   <li>{@link Method#CHRISTIANO} (Christiano et al., NeurIPS 2017,
 *       <a href="https://arxiv.org/abs/1706.03741">arXiv:1706.03741</a>): rewards are predicted
 *       when transitions are stored and never revised;</li>
 *   <li>{@link Method#PEBBLE} (Lee et al., ICML 2021,
 *       <a href="https://arxiv.org/abs/2106.05091">arXiv:2106.05091</a>): unsupervised
 *       pre-training on a k-nearest-neighbour state-entropy reward first, so the first queried
 *       segments are diverse, and the whole replay buffer relabelled whenever the reward model
 *       changes;</li>
 *   <li>{@link Method#SURF} (Park et al., ICLR 2022,
 *       <a href="https://arxiv.org/abs/2203.10050">arXiv:2203.10050</a>): PEBBLE plus confident
 *       pseudo-labels for unlabelled pairs and temporal-crop augmentation of segments;</li>
 *   <li>{@link Method#TRUE_REWARD}: the same TD3 on the environment's reward, the ceiling.</li>
 * </ul>
 *
 * <p>Queries are picked uniformly, by ensemble disagreement (the spread of the predicted
 * preference probabilities) or by entropy (probabilities closest to 1/2), see
 * {@link Query}. The scripted teacher prefers the segment with the larger true return and
 * flips its answer with probability {@code labelNoise}.
 */
public final class PreferenceRl implements Learner {

    /** The algorithm. */
    public enum Method {
        /** reward model fixed at insertion */
        CHRISTIANO,
        /** state-entropy pre-training and buffer relabelling */
        PEBBLE,
        /** PEBBLE with pseudo-labels and temporal crops */
        SURF,
        /** learn from the true reward */
        TRUE_REWARD
    }

    /** How queried pairs are chosen among candidates. */
    public enum Query {
        /** at random */
        UNIFORM,
        /** largest ensemble standard deviation of the preference probability */
        DISAGREEMENT,
        /** mean probability closest to 1/2 */
        ENTROPY
    }

    private record Step(float[] obs, float[] action, float trueReward, int episode) {
    }

    private record Pair(int a, int b, int length, float label) {
    }

    private final Method method;
    private final int obsDims;
    private final int actDims;
    private final StepwiseTd3 td3;
    private final List<Model> ensemble = new ArrayList<>();
    private final List<Optimizer> optimizers = new ArrayList<>();
    private final List<Step> history = new ArrayList<>();
    private final List<Pair> labelled = new ArrayList<>();
    private final SplittableRandom random;
    private Query query = Query.DISAGREEMENT;
    private int segment = 10;
    private int budget = 200;
    private int perRound = 20;
    private int queryEvery = 400;
    private int pretrain = 300;
    private double labelNoise;
    private int asked;

    private PreferenceRl(Method method, int obsDims, int actDims, RlNetworkFactory nets, long seed) {
        this.method = method;
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.random = new SplittableRandom(seed);
        this.td3 = StepwiseTd3.of(obsDims, actDims, nets, 1e-3f, seed).discount(0.97f);
        for (int k = 0; k < 3; k++) {
            ensemble.add(Nets.mlp(obsDims + actDims, 1, Activation.TANH, seed + 10 + k, 32));
            optimizers.add(Optimizer.adam(3e-3f));
        }
    }

    /**
     * @param method the algorithm
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param networks builds the TD3 actor and critics
     * @param seed the seed
     * @return the learner
     */
    public static PreferenceRl of(Method method, int observationDims, int actionDims, RlNetworkFactory networks,
            long seed) {
        return new PreferenceRl(Objects.requireNonNull(method), observationDims, actionDims, networks, seed);
    }

    /** @param q the query selection @return this learner */
    public PreferenceRl query(Query q) {
        this.query = Objects.requireNonNull(q);
        return this;
    }

    /**
     * @param total the most queries to the teacher
     * @param round queries per round
     * @param every environment steps between rounds
     * @return this learner
     */
    public PreferenceRl feedback(int total, int round, int every) {
        this.budget = total;
        this.perRound = round;
        this.queryEvery = every;
        return this;
    }

    /** @param p the teacher's label-flip probability @return this learner */
    public PreferenceRl labelNoise(double p) {
        this.labelNoise = p;
        return this;
    }

    /** @param length the segment length @return this learner */
    public PreferenceRl segmentLength(int length) {
        this.segment = length;
        return this;
    }

    /** @param steps PEBBLE's unsupervised pre-training steps @return this learner */
    public PreferenceRl pretrainSteps(int steps) {
        this.pretrain = steps;
        return this;
    }

    /** @return queries asked so far */
    public int queries() {
        return asked;
    }

    /** @return gradient updates of the TD3 */
    public long updates() {
        return td3.updates();
    }

    private float[] input(float[] obs, float[] action) {
        float[] x = new float[obsDims + actDims];
        System.arraycopy(obs, 0, x, 0, obsDims);
        System.arraycopy(action, 0, x, obsDims, actDims);
        return x;
    }

    /** Mean predicted reward of rows {@code [n, obs + act]}. */
    private float[] predict(float[] rows, int n) {
        float[] out = new float[n];
        Tensor x = Tensor.of(Shape.of(n, obsDims + actDims), rows);
        for (Model m : ensemble) {
            float[] r = m.forward(Var.of(x), false).value().toArray();
            for (int i = 0; i < n; i++) {
                out[i] += r[i] / ensemble.size();
            }
        }
        return out;
    }

    private float rewardHat(float[] obs, float[] action) {
        return predict(input(obs, action), 1)[0];
    }

    private boolean validStart(int i, int length) {
        return i + length <= history.size() && history.get(i).episode() == history.get(i + length - 1).episode();
    }

    private int randomStart(int length) {
        for (int tries = 0; tries < 100; tries++) {
            int i = random.nextInt(Math.max(1, history.size() - length));
            if (validStart(i, length)) {
                return i;
            }
        }
        return -1;
    }

    private double trueReturn(int start, int length) {
        double s = 0;
        for (int i = start; i < start + length; i++) {
            s += history.get(i).trueReward();
        }
        return s;
    }

    /** Per-member preference probabilities that segment a beats segment b. */
    private double[] probabilities(int a, int b, int length) {
        float[] rows = new float[2 * length * (obsDims + actDims)];
        int w = obsDims + actDims;
        for (int i = 0; i < length; i++) {
            System.arraycopy(input(history.get(a + i).obs(), history.get(a + i).action()), 0, rows, i * w, w);
            System.arraycopy(input(history.get(b + i).obs(), history.get(b + i).action()), 0, rows,
                    (length + i) * w, w);
        }
        Tensor x = Tensor.of(Shape.of(2 * length, w), rows);
        double[] p = new double[ensemble.size()];
        for (int k = 0; k < ensemble.size(); k++) {
            float[] r = ensemble.get(k).forward(Var.of(x), false).value().toArray();
            double ra = 0;
            double rb = 0;
            for (int i = 0; i < length; i++) {
                ra += r[i];
                rb += r[length + i];
            }
            p[k] = 1 / (1 + Math.exp(-(ra - rb)));
        }
        return p;
    }

    private void askTeacher() {
        int n = Math.min(perRound, budget - asked);
        int candidates = query == Query.UNIFORM ? n : 5 * n;
        List<int[]> pool = new ArrayList<>();
        List<Double> score = new ArrayList<>();
        for (int c = 0; c < candidates; c++) {
            int a = randomStart(segment);
            int b = randomStart(segment);
            if (a < 0 || b < 0) {
                continue;
            }
            pool.add(new int[] {a, b});
            if (query == Query.UNIFORM) {
                score.add(random.nextDouble());
            } else {
                double[] p = probabilities(a, b, segment);
                double mean = java.util.Arrays.stream(p).average().orElse(0.5);
                double sd = Math.sqrt(java.util.Arrays.stream(p).map(v -> (v - mean) * (v - mean)).average().orElse(0));
                score.add(query == Query.DISAGREEMENT ? sd : -Math.abs(mean - 0.5));
            }
        }
        Integer[] order = new Integer[pool.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(score.get(y), score.get(x)));
        for (int i = 0; i < Math.min(n, order.length); i++) {
            int[] q = pool.get(order[i]);
            boolean aBetter = trueReturn(q[0], segment) >= trueReturn(q[1], segment);
            if (random.nextDouble() < labelNoise) {
                aBetter = !aBetter;
            }
            labelled.add(new Pair(q[0], q[1], segment, aBetter ? 1 : 0));
            asked++;
        }
    }

    private void trainRewardModel() {
        List<Pair> data = new ArrayList<>(labelled);
        if (method == Method.SURF) {
            for (int c = 0; c < 2 * labelled.size(); c++) {
                int a = randomStart(segment);
                int b = randomStart(segment);
                if (a < 0 || b < 0) {
                    continue;
                }
                double p = java.util.Arrays.stream(probabilities(a, b, segment)).average().orElse(0.5);
                if (p > 0.95 || p < 0.05) {
                    data.add(new Pair(a, b, segment, p > 0.5 ? 1 : 0));
                }
            }
        }
        int w = obsDims + actDims;
        for (int epoch = 0; epoch < 60; epoch++) {
            for (int k = 0; k < ensemble.size(); k++) {
                int batch = Math.min(32, data.size());
                int len = segment;
                int crop = method == Method.SURF ? segment - 1 - random.nextInt(Math.max(1, segment / 3)) : segment;
                float[] win = new float[batch * crop * w];
                float[] lose = new float[batch * crop * w];
                for (int i = 0; i < batch; i++) {
                    Pair p = data.get(random.nextInt(data.size()));
                    int offA = method == Method.SURF ? random.nextInt(len - crop + 1) : 0;
                    int offB = method == Method.SURF ? random.nextInt(len - crop + 1) : 0;
                    int ws = p.label() > 0.5f ? p.a() + offA : p.b() + offB;
                    int ls = p.label() > 0.5f ? p.b() + offB : p.a() + offA;
                    for (int j = 0; j < crop; j++) {
                        Step x = history.get(ws + j);
                        Step y = history.get(ls + j);
                        System.arraycopy(input(x.obs(), x.action()), 0, win, (i * crop + j) * w, w);
                        System.arraycopy(input(y.obs(), y.action()), 0, lose, (i * crop + j) * w, w);
                    }
                }
                Model m = ensemble.get(k);
                Var rw = m.forward(Var.of(Tensor.of(Shape.of(batch * crop, w), win)), true).reshape(batch, crop)
                        .sum(1, false);
                Var rl = m.forward(Var.of(Tensor.of(Shape.of(batch * crop, w), lose)), true).reshape(batch, crop)
                        .sum(1, false);
                Var loss = PreferenceLosses.bradleyTerry(rw, rl);
                loss.backward();
                optimizers.get(k).step(m.parameters());
            }
        }
    }

    private void relabel() {
        List<StepwiseTd3.Entry> buf = td3.buffer();
        int n = buf.size();
        int w = obsDims + actDims;
        float[] rows = new float[n * w];
        for (int i = 0; i < n; i++) {
            System.arraycopy(input(buf.get(i).observation(), buf.get(i).action()), 0, rows, i * w, w);
        }
        float[] r = predict(rows, n);
        int[] idx = {0};
        td3.relabel(e -> r[idx[0]++]);
    }

    private float entropyReward(float[] next) {
        int n = td3.buffer().size();
        if (n < 10) {
            return 0;
        }
        double[] d = new double[Math.min(64, n)];
        for (int i = 0; i < d.length; i++) {
            float[] o = td3.buffer().get(random.nextInt(n)).next();
            double s = 0;
            for (int j = 0; j < obsDims; j++) {
                s += (o[j] - next[j]) * (o[j] - next[j]);
            }
            d[i] = Math.sqrt(s);
        }
        java.util.Arrays.sort(d);
        return (float) Math.log(1 + d[Math.min(4, d.length - 1)]);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return td3.act(observation, explore);
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        Space.Box box = (Space.Box) env.actionSpace();
        Tracker t = new Tracker(listener);
        float[] obs = env.reset(random.split());
        int episode = 0;
        long taken = 0;
        boolean pebble = method == Method.PEBBLE || method == Method.SURF;
        int warm = pebble ? pretrain : 200;
        while (t.steps() < steps) {
            float[] unit;
            if (taken < 200) {
                unit = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    unit[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                unit = td3.unitAction(obs, true);
            }
            Environment.Step s = env.step(StepwiseTd3.scale(box, unit));
            taken++;
            history.add(new Step(obs, unit, s.reward(), episode));
            float r;
            if (method == Method.TRUE_REWARD) {
                r = s.reward();
            } else if (pebble && taken <= pretrain) {
                r = entropyReward(s.observation());
            } else {
                r = labelled.isEmpty() ? 0 : rewardHat(obs, unit);
            }
            td3.remember(obs, unit, r, s.observation(), s.terminated());
            if (t.step(s)) {
                obs = env.reset(random.split());
                episode++;
            } else {
                obs = s.observation();
            }
            if (method != Method.TRUE_REWARD && taken >= warm && asked < budget
                    && (taken - warm) % queryEvery == 0) {
                askTeacher();
                trainRewardModel();
                if (pebble || labelled.size() == asked && taken == warm) {
                    relabel();
                }
            }
            if (taken > 200) {
                td3.update();
            }
        }
        return t.result();
    }

    @Override
    public String toString() {
        return method + "(" + query + ")";
    }
}
