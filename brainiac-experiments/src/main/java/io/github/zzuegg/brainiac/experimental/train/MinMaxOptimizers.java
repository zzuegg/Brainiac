package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Optimisers for two-player differentiable games {@code min_x max_y f(x, y)} (and their
 * general-sum form, where each player minimises its own loss), such as GANs. Simultaneous
 * gradient descent-ascent rotates around the equilibrium of the bilinear game
 * {@code f = x y} and spirals outwards; these methods correct for the rotation:
 * <ul>
 *   <li>{@link #simultaneous} and {@link #alternating} gradient descent-ascent, the baselines
 *       (the core {@code Adversarial} loop alternates).</li>
 *   <li>{@link #extragradient}: take a look-ahead step, then update the original point with
 *       the gradient at the look-ahead point (Korpelevich 1976; Gidel et al. 2019,
 *       <a href="https://arxiv.org/abs/1802.10551">arXiv:1802.10551</a>).</li>
 *   <li>{@link #optimisticAdam}: an {@link Optimizer} that steps along
 *       {@code 2 u_t - u_(t-1)} (Adam's update now, minus the previous one), the same
 *       correction without a second gradient (Daskalakis et al. 2018,
 *       <a href="https://arxiv.org/abs/1711.00141">arXiv:1711.00141</a>); drop it into any
 *       simultaneous or alternating loop.</li>
 *   <li>{@link #consensus}: each player also descends {@code gamma/2 |v|^2}, the squared norm
 *       of the joint gradient field {@code v}, which damps the rotation (Mescheder et al. 2017,
 *       <a href="https://arxiv.org/abs/1705.10461">arXiv:1705.10461</a>); it needs double
 *       backpropagation.</li>
 * </ul>
 *
 * <pre>{@code
 * MinMaxOptimizers.Stepper step = MinMaxOptimizers.extragradient(gParams, Optimizer.adam(1e-3f),
 *         dParams, Optimizer.adam(1e-3f), 1e-3f);
 * step.step(() -> new Var[] {generatorLoss(), criticLoss()});
 * }</pre>
 */
public final class MinMaxOptimizers {

    /** One evaluation of the game at the current parameters. */
    @FunctionalInterface
    public interface Game {
        /**
         * @return {@code {loss of the first player, loss of the second player}}, each minimised
         *     by its own player; for a zero-sum game the second is the negated first
         */
        Var[] losses();
    }

    /** Updates both players' parameters by one step of a game. */
    @FunctionalInterface
    public interface Stepper {
        /**
         * @param game evaluates the losses at the current parameters
         */
        void step(Game game);
    }

    private MinMaxOptimizers() {
    }

    private static List<Tensor> grads(Var loss, List<Var> params) {
        List<Var> g = Var.gradients(loss, params, false);
        List<Tensor> out = new ArrayList<>(g.size());
        for (Var v : g) {
            out.add(v.value());
        }
        return out;
    }

    private static void apply(List<Var> params, List<Tensor> grads, Optimizer opt) {
        for (int i = 0; i < params.size(); i++) {
            params.get(i).setGrad(grads.get(i));
        }
        opt.step(params);
    }

    /**
     * Simultaneous gradient descent for both players (descent-ascent for a zero-sum game).
     *
     * @param first the first player's parameters
     * @param firstOptimizer its optimiser
     * @param second the second player's parameters
     * @param secondOptimizer its optimiser
     * @return the stepper
     */
    public static Stepper simultaneous(List<Var> first, Optimizer firstOptimizer, List<Var> second,
            Optimizer secondOptimizer) {
        return game -> {
            Var[] l = game.losses();
            List<Tensor> g1 = grads(l[0], first);
            List<Tensor> g2 = grads(l[1], second);
            apply(first, g1, firstOptimizer);
            apply(second, g2, secondOptimizer);
        };
    }

    /**
     * Alternating updates: the second player moves first, then the first player responds to
     * the updated second player.
     *
     * @param first the first player's parameters
     * @param firstOptimizer its optimiser
     * @param second the second player's parameters
     * @param secondOptimizer its optimiser
     * @return the stepper
     */
    public static Stepper alternating(List<Var> first, Optimizer firstOptimizer, List<Var> second,
            Optimizer secondOptimizer) {
        return game -> {
            apply(second, grads(game.losses()[1], second), secondOptimizer);
            apply(first, grads(game.losses()[0], first), firstOptimizer);
        };
    }

    /**
     * Extragradient: from the current point, take a plain gradient step of size
     * {@code lookahead} for both players, evaluate the gradients there, return to the current
     * point and update it with those look-ahead gradients through the optimisers.
     *
     * @param first the first player's parameters
     * @param firstOptimizer its optimiser
     * @param second the second player's parameters
     * @param secondOptimizer its optimiser
     * @param lookahead the extrapolation step size
     * @return the stepper
     */
    public static Stepper extragradient(List<Var> first, Optimizer firstOptimizer, List<Var> second,
            Optimizer secondOptimizer, float lookahead) {
        return game -> {
            Var[] l = game.losses();
            List<Tensor> g1 = grads(l[0], first);
            List<Tensor> g2 = grads(l[1], second);
            List<Tensor> saved1 = save(first);
            List<Tensor> saved2 = save(second);
            move(first, g1, lookahead);
            move(second, g2, lookahead);
            Var[] la = game.losses();
            List<Tensor> h1 = grads(la[0], first);
            List<Tensor> h2 = grads(la[1], second);
            restore(first, saved1);
            restore(second, saved2);
            apply(first, h1, firstOptimizer);
            apply(second, h2, secondOptimizer);
        };
    }

    /**
     * Consensus optimisation: both players add {@code gamma * |v|^2 / 2} to their losses, where
     * {@code v} is the joint gradient field (each player's gradient of its own loss).
     *
     * @param first the first player's parameters
     * @param firstOptimizer its optimiser
     * @param second the second player's parameters
     * @param secondOptimizer its optimiser
     * @param gamma the weight of the gradient-norm penalty
     * @return the stepper
     */
    public static Stepper consensus(List<Var> first, Optimizer firstOptimizer, List<Var> second,
            Optimizer secondOptimizer, float gamma) {
        return game -> {
            Var[] l = game.losses();
            List<Var> v1 = Var.gradients(l[0], first, true);
            List<Var> v2 = Var.gradients(l[1], second, true);
            Var norm = null;
            for (List<Var> vs : List.of(v1, v2)) {
                for (Var v : vs) {
                    Var s = v.square().sum();
                    norm = norm == null ? s : norm.add(s);
                }
            }
            Var penalty = Objects.requireNonNull(norm).mul(gamma / 2);
            List<Tensor> g1 = grads(l[0].add(penalty), first);
            List<Tensor> g2 = grads(l[1].add(penalty), second);
            apply(first, g1, firstOptimizer);
            apply(second, g2, secondOptimizer);
        };
    }

    private static List<Tensor> save(List<Var> params) {
        List<Tensor> out = new ArrayList<>(params.size());
        for (Var p : params) {
            out.add(p.value().copy());
        }
        return out;
    }

    private static void restore(List<Var> params, List<Tensor> saved) {
        for (int i = 0; i < params.size(); i++) {
            params.get(i).assign(saved.get(i));
        }
    }

    private static void move(List<Var> params, List<Tensor> grads, float rate) {
        for (int i = 0; i < params.size(); i++) {
            params.get(i).assign(params.get(i).value().sub(grads.get(i).mul(rate)));
        }
    }

    /**
     * Optimistic Adam: Adam's moment estimates, but each step applies {@code 2 u_t - u_(t-1)},
     * where {@code u_t = lr * mHat_t / (sqrt(vHat_t) + eps)} is the plain Adam update.
     *
     * @param learningRate the step size
     * @param beta1 the first-moment decay
     * @param beta2 the second-moment decay
     * @return a new optimiser
     */
    public static Optimizer optimisticAdam(float learningRate, float beta1, float beta2) {
        return new OptimisticAdam(learningRate, beta1, beta2);
    }

    /**
     * Optimistic gradient descent: steps along {@code 2 g_t - g_(t-1)}.
     *
     * @param learningRate the step size
     * @return a new optimiser
     */
    public static Optimizer optimisticSgd(float learningRate) {
        return new OptimisticAdam(learningRate, 0, -1);
    }

    /** Optimistic Adam; with {@code beta2 < 0} it is optimistic plain gradient descent. */
    private static final class OptimisticAdam implements Optimizer {
        private float rate;
        private final float beta1;
        private final float beta2;
        private final Map<Var, Tensor[]> state = new IdentityHashMap<>();
        private int t;

        OptimisticAdam(float rate, float beta1, float beta2) {
            this.rate = rate;
            this.beta1 = beta1;
            this.beta2 = beta2;
        }

        @Override
        public void step(List<Var> parameters) {
            t++;
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                Tensor[] s = state.get(p);
                Tensor update;
                if (beta2 < 0) {
                    update = g.mul(rate);
                } else {
                    if (s == null) {
                        s = new Tensor[] {g.mul(0f), g.mul(0f), null};
                    }
                    s[0] = s[0].mul(beta1).add(g.mul(1 - beta1));
                    s[1] = s[1].mul(beta2).add(g.square().mul(1 - beta2));
                    Tensor mHat = s[0].div((float) (1 - Math.pow(beta1, t)));
                    Tensor vHat = s[1].div((float) (1 - Math.pow(beta2, t)));
                    update = mHat.div(vHat.sqrt().add(1e-8f)).mul(rate);
                }
                if (s == null) {
                    s = new Tensor[3];
                }
                Tensor previous = s[2] == null ? update : s[2];
                p.assign(p.value().sub(update.mul(2f)).add(previous));
                s[2] = update;
                state.put(p, s);
                p.zeroGrad();
            }
        }

        @Override
        public float learningRate() {
            return rate;
        }

        @Override
        public void learningRate(float learningRate) {
            this.rate = learningRate;
        }
    }
}
