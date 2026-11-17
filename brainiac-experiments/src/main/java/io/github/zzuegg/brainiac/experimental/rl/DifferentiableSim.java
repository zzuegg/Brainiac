package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * A differentiable mass-spring simulator in the style of DiffTaichi (Hu et al.,
 * <a href="https://arxiv.org/abs/1910.00935">arXiv:1910.00935</a>) and Brax (Freeman et al.,
 * <a href="https://arxiv.org/abs/2106.13281">arXiv:2106.13281</a>): every physics step is written
 * in {@link Var} operations, so the gradient of a task loss flows back through the whole
 * trajectory into the controller (backpropagation through time) and the controller is
 * optimised directly, without a value function or policy-gradient noise.
 *
 * <p>The system is a planar "flexible arm": a driven hand mass and a passive ball mass on a
 * zero-length spring. The task is to bring the ball to a goal at the last step and hold it
 * there: the cost is {@code |ball - goal|^2 + 0.1 |ball velocity|^2} at the horizon. A
 * controller maps the observation {@code (hand, hand velocity, ball, ball velocity,
 * goal - ball)} to a raw force that the simulator squashes with {@code tanh}, so the same
 * network can be trained by BPTT, searched by CEM or learned by PPO through
 * {@link #asEnvironment}, whose per-step reward is the drop in that cost (it telescopes to the
 * same objective).
 *
 * <p>Batched state rows are {@code (hx, hy, hvx, hvy, bx, by, bvx, bvy)}.
 */
public final class DifferentiableSim {
    private final float dt;
    private final int horizon;
    private final float handMass;
    private final float ballMass;
    private final float stiffness;
    private final float handDamping;
    private final float ballDamping;
    private final float maxForce;
    private final float goalRadius;

    private DifferentiableSim(float dt, int horizon, float handMass, float ballMass, float stiffness,
            float handDamping, float ballDamping, float maxForce, float goalRadius) {
        this.dt = dt;
        this.horizon = horizon;
        this.handMass = handMass;
        this.ballMass = ballMass;
        this.stiffness = stiffness;
        this.handDamping = handDamping;
        this.ballDamping = ballDamping;
        this.maxForce = maxForce;
        this.goalRadius = goalRadius;
    }

    /**
     * The default flexible-arm task: 30 steps of 0.1 s, hand mass 1, ball mass 0.5, spring
     * stiffness 4, maximum force 2 and goals in a disc of radius 1.5 around the start.
     *
     * @return the simulator
     */
    public static DifferentiableSim massSpring() {
        return new DifferentiableSim(0.1f, 30, 1f, 0.5f, 4f, 0.5f, 0.1f, 2f, 1.5f);
    }

    /** @return the number of steps per episode */
    public int horizon() {
        return horizon;
    }

    /** @return the observation size, 10 */
    public int observationSize() {
        return 10;
    }

    /** @return the action size, 2 */
    public int actionSize() {
        return 2;
    }

    /**
     * Samples goals uniformly in the goal disc.
     *
     * @param n how many
     * @param random the random source
     * @return {@code [n, 2]} goals
     */
    public Tensor goals(int n, SplittableRandom random) {
        float[] g = new float[2 * n];
        for (int i = 0; i < n; i++) {
            double r = goalRadius * Math.sqrt(random.nextDouble());
            double a = 2 * Math.PI * random.nextDouble();
            g[2 * i] = (float) (r * Math.cos(a));
            g[2 * i + 1] = (float) (r * Math.sin(a));
        }
        return Tensor.of(Shape.of(n, 2), g);
    }

    /**
     * One semi-implicit Euler step.
     *
     * @param state {@code [batch, 8]} states
     * @param action {@code [batch, 2]} raw actions, squashed with {@code tanh}
     * @return the next states
     */
    public Var step(Var state, Var action) {
        Var hand = state.slice(1, 0, 2);
        Var handVelocity = state.slice(1, 2, 4);
        Var ball = state.slice(1, 4, 6);
        Var ballVelocity = state.slice(1, 6, 8);
        Var spring = ball.sub(hand).mul(stiffness);
        Var handAcc = action.tanh().mul(maxForce).add(spring).sub(handVelocity.mul(handDamping)).div(handMass);
        Var ballAcc = spring.neg().sub(ballVelocity.mul(ballDamping)).div(ballMass);
        Var hv = handVelocity.add(handAcc.mul(dt));
        Var bv = ballVelocity.add(ballAcc.mul(dt));
        return Var.concat(1, hand.add(hv.mul(dt)), hv, ball.add(bv.mul(dt)), bv);
    }

    /**
     * The controller's view of a state.
     *
     * @param state {@code [batch, 8]} states
     * @param goals {@code [batch, 2]} goals
     * @return {@code [batch, 10]} observations
     */
    public Var observe(Var state, Var goals) {
        return Var.concat(1, state, goals.sub(state.slice(1, 4, 6)));
    }

    /**
     * The task cost of each state.
     *
     * @param state {@code [batch, 8]} states
     * @param goals {@code [batch, 2]} goals
     * @return {@code [batch, 1]} costs
     */
    public Var cost(Var state, Var goals) {
        Var miss = state.slice(1, 4, 6).sub(goals).square().sum(1, true);
        return miss.add(state.slice(1, 6, 8).square().sum(1, true).mul(0.1f));
    }

    /**
     * Rolls a controller out from rest for the whole horizon, keeping the graph.
     *
     * @param controller observations to raw actions
     * @param goals {@code [batch, 2]} goals
     * @param training whether the controller runs in training mode
     * @return the final states, {@code [batch, 8]}
     */
    public Var rollout(Model controller, Tensor goals, boolean training) {
        Var g = Var.of(goals);
        Var state = Var.of(Tensor.zeros(Shape.of(goals.shape().dim(0), 8)));
        for (int t = 0; t < horizon; t++) {
            state = step(state, controller.forward(observe(state, g), training));
        }
        return state;
    }

    /**
     * The differentiable episode loss: the mean final cost over the batch.
     *
     * @param controller observations to raw actions
     * @param goals {@code [batch, 2]} goals
     * @return the scalar loss
     */
    public Var loss(Model controller, Tensor goals) {
        return cost(rollout(controller, goals, true), Var.of(goals)).mean();
    }

    /**
     * The mean distance of the ball from its goal at the horizon.
     *
     * @param controller observations to raw actions
     * @param goals {@code [batch, 2]} goals
     * @return the mean final distance
     */
    public double finalDistance(Model controller, Tensor goals) {
        float[] s = rollout(controller, goals, false).value().toArray();
        float[] g = goals.toArray();
        double sum = 0;
        int n = g.length / 2;
        for (int i = 0; i < n; i++) {
            sum += Math.hypot(s[8 * i + 4] - g[2 * i], s[8 * i + 5] - g[2 * i + 1]);
        }
        return sum / n;
    }

    /**
     * The task as a plain environment for black-box learners: a random goal per episode,
     * reward = the drop in cost per step, truncation at the horizon.
     *
     * @return the environment
     */
    public Environment asEnvironment() {
        return new Environment() {
            private Var state;
            private Var goal;
            private float lastCost;
            private int steps;

            @Override
            public Space observationSpace() {
                return Space.box(10, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
            }

            @Override
            public Space actionSpace() {
                return Space.box(2, -3f, 3f);
            }

            @Override
            public float[] reset(SplittableRandom random) {
                goal = Var.of(goals(1, random));
                state = Var.of(Tensor.zeros(Shape.of(1, 8)));
                lastCost = cost(state, goal).value().item();
                steps = 0;
                return observe(state, goal).value().toArray();
            }

            @Override
            public Step step(float[] action) {
                state = Var.of(DifferentiableSim.this.step(state, Var.of(Tensor.of(Shape.of(1, 2), action.clone())))
                        .value());
                float c = cost(state, goal).value().item();
                float reward = lastCost - c;
                lastCost = c;
                steps++;
                return new Step(observe(state, goal).value().toArray(), reward, false, steps >= horizon);
            }
        };
    }
}
