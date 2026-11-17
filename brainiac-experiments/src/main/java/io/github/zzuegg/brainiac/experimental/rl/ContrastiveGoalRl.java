package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
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
 * Contrastive goal-conditioned RL (Eysenbach et al., <a href="https://arxiv.org/abs/2206.07568">2206.07568</a>),
 * in the scaled-up form of "1000 Layer Networks for Self-Supervised RL" (Wang et al.,
 * <a href="https://arxiv.org/abs/2503.14858">2503.14858</a>, NeurIPS 2025 best paper).
 * There is no reward at all: a critic {@code f(s, a, g) = phi(s, a) . psi(g)} is trained
 * with InfoNCE to tell which goal is a <em>future state of the same trajectory</em> (offset
 * drawn from a geometric distribution with the discount) among the other goals of the
 * batch. That score is, up to a constant, the log of the discounted occupancy of {@code g}
 * after {@code (s, a)}, i.e. a Q-function for reaching {@code g}; the policy picks the
 * action with the highest score for the commanded goal.
 *
 * <p>The paper's finding is that this objective, unlike TD-based RL, keeps improving as the
 * encoders get much deeper, given residual blocks with LayerNorm and Swish
 * ({@link ResidualRlNetworks#deepResidual}). Discrete actions: {@code phi} outputs one
 * embedding per action. Works with any {@link GoalEnvironment} whose observation ends with
 * the goal.
 */
public final class ContrastiveGoalRl implements Learner {
    private final int stateSize;
    private final int goalSize;
    private final int actions;
    private final int representation;
    private final Model phi;
    private final Model psi;
    private final Optimizer optimizer = Optimizer.adam(1e-3f);
    private float gamma = 0.9f;
    private int batchSize = 64;
    private int updatesPerEpisode = 4;
    private double epsilon = 0.2;
    private final SplittableRandom random;
    private final List<List<float[]>> states = new ArrayList<>();   // per episode: s_0..s_T (achieved goals)
    private final List<List<float[]>> observations = new ArrayList<>();
    private final List<List<Integer>> taken = new ArrayList<>();
    private double lastLoss = Double.NaN;
    private GoalEnvironment env;

    private ContrastiveGoalRl(int stateSize, int goalSize, int actions, RlNetworkFactory networks, int representation,
            long seed) {
        this.stateSize = stateSize;
        this.goalSize = goalSize;
        this.actions = actions;
        this.representation = representation;
        this.phi = networks.create(stateSize, actions * representation, seed);
        this.psi = networks.create(goalSize, representation, seed + 1);
        this.random = new SplittableRandom(seed + 2);
    }

    /**
     * Creates the learner.
     *
     * @param stateSize the observation size without the goal
     * @param goalSize the goal size
     * @param actions the number of discrete actions
     * @param networks builds the two encoders
     * @param representation the embedding size
     * @param seed the seed
     * @return the learner
     */
    public static ContrastiveGoalRl of(int stateSize, int goalSize, int actions, RlNetworkFactory networks,
            int representation, long seed) {
        return new ContrastiveGoalRl(stateSize, goalSize, actions, Objects.requireNonNull(networks), representation,
                seed);
    }

    /**
     * Sets the discount of the geometric future-offset distribution.
     *
     * @param gamma the discount
     * @return this learner
     */
    public ContrastiveGoalRl discount(float gamma) {
        this.gamma = gamma;
        return this;
    }

    /**
     * Sets the exploration rate.
     *
     * @param epsilon the share of random actions while training
     * @return this learner
     */
    public ContrastiveGoalRl epsilon(double epsilon) {
        this.epsilon = epsilon;
        return this;
    }

    /**
     * Sets how many gradient steps follow each episode.
     *
     * @param n the number
     * @return this learner
     */
    public ContrastiveGoalRl updatesPerEpisode(int n) {
        this.updatesPerEpisode = n;
        return this;
    }

    /** @return the InfoNCE loss of the last update */
    public double lastLoss() {
        return lastLoss;
    }

    private float[] state(float[] observation) {
        return java.util.Arrays.copyOf(observation, stateSize);
    }

    private float[] goal(float[] observation) {
        return java.util.Arrays.copyOfRange(observation, observation.length - goalSize, observation.length);
    }

    /**
     * The critic's score for every action.
     *
     * @param observation state followed by goal
     * @return {@code phi(s, a) . psi(g)} per action
     */
    public float[] scores(float[] observation) {
        float[] p = RlTensors.row(phi, state(observation));
        float[] g = RlTensors.row(psi, goal(observation));
        float[] s = new float[actions];
        for (int a = 0; a < actions; a++) {
            for (int k = 0; k < representation; k++) {
                s[a] += p[a * representation + k] * g[k];
            }
        }
        return s;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {RlTensors.argmax(scores(observation))};
    }

    @Override
    public RlResult fit(Environment environment, long steps, RlListener listener) {
        if (!(environment instanceof GoalEnvironment g) || !(environment.actionSpace() instanceof Space.Discrete d)
                || d.n() != actions) {
            throw new IllegalArgumentException("needs a goal environment with " + actions + " discrete actions");
        }
        this.env = g;
        Tracker tracker = new Tracker(listener);
        float[] obs = environment.reset(random.split());
        List<float[]> s = new ArrayList<>(List.of(env.achievedGoal(obs)));
        List<float[]> o = new ArrayList<>(List.of(obs));
        List<Integer> a = new ArrayList<>();
        while (tracker.steps() < steps) {
            float[] act = act(obs, true);
            Environment.Step st = environment.step(act);
            a.add(Math.round(act[0]));
            s.add(env.achievedGoal(st.observation()));
            o.add(st.observation());
            if (tracker.step(st)) {
                states.add(s);
                observations.add(o);
                taken.add(a);
                for (int u = 0; u < updatesPerEpisode; u++) {
                    update();
                }
                obs = environment.reset(random.split());
                s = new ArrayList<>(List.of(env.achievedGoal(obs)));
                o = new ArrayList<>(List.of(obs));
                a = new ArrayList<>();
            } else {
                obs = st.observation();
            }
        }
        return tracker.result();
    }

    private void update() {
        int b = batchSize;
        float[] stateIn = new float[b * stateSize];
        float[] goalIn = new float[b * goalSize];
        float[] actHot = new float[b * actions * representation];
        for (int i = 0; i < b; i++) {
            int e = random.nextInt(states.size());
            List<float[]> traj = observations.get(e);
            int len = taken.get(e).size();
            int t = random.nextInt(len);
            int k = 1;
            while (random.nextDouble() < gamma && t + k < len) {
                k++;
            }
            System.arraycopy(state(traj.get(t)), 0, stateIn, i * stateSize, stateSize);
            System.arraycopy(states.get(e).get(t + k), 0, goalIn, i * goalSize, goalSize);
            int act = taken.get(e).get(t);
            for (int r = 0; r < representation; r++) {
                actHot[(i * actions + act) * representation + r] = 1;
            }
        }
        Var p = phi.forward(Var.of(Tensor.of(Shape.of(b, stateSize), stateIn)), true)
                .mul(Tensor.of(Shape.of(b, actions * representation), actHot))
                .reshape(b, actions, representation).sum(1, false);
        Var q = psi.forward(Var.of(Tensor.of(Shape.of(b, goalSize), goalIn)), true);
        Var logits = p.matmul(q.transpose());
        float[] eye = new float[b * b];
        for (int i = 0; i < b; i++) {
            eye[i * b + i] = 1;
        }
        Var logp = RlTensors.logSoftmax(logits);
        Var lse = logits.sub(logits.value().max(1, true)).exp().sum(1, true).log().add(logits.value().max(1, true));
        Var loss = logp.mul(Tensor.of(Shape.of(b, b), eye)).sum(1, true).neg().mean()
                .add(lse.square().mean().mul(0.01f));
        lastLoss = loss.value().item();
        loss.backward();
        List<Var> params = new ArrayList<>(phi.parameters());
        params.addAll(psi.parameters());
        RlTensors.clip(params, 5f);
        optimizer.step(params);
    }
}
