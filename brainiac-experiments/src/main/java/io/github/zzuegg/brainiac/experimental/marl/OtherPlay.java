package io.github.zzuegg.brainiac.experimental.marl;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Other-play for zero-shot coordination (Hu et al., ICML 2020,
 * <a href="https://arxiv.org/abs/2003.02979">arXiv:2003.02979</a>): instead of maximising the
 * self-play return {@code J(pi_0, pi_1)}, train on {@code E_phi J(pi_0, phi(pi_1))} where
 * {@code phi} is drawn from the game's symmetry group, so the partner is seen through a random
 * relabelling of equivalent actions or messages. Arbitrary conventions ("the left lever is the
 * one") average away and only conventions that survive every relabelling are learned, so
 * independently trained agents can play together.
 *
 * <p>The games are two-player, one-shot and cooperative: player 0 observes {@code o0}, acts
 * {@code a0}; player 1 observes {@code o1 = observe1(o0, a0)} and acts {@code a1}; both get
 * {@code payoff(o0, a0, a1)}. A symmetry relabels player 1's observation and action. Players are
 * tabular, trained by REINFORCE ({@link Learner#POLICY_GRADIENT}) or by independent
 * epsilon-greedy Q-learning ({@link Learner#Q_LEARNING}); {@link #crossPlay} pairs the player 0
 * of one seed with the player 1 of another.
 */
public final class OtherPlay {

    /** How the two players learn. */
    public enum Learner {
        /** tabular softmax policies trained by REINFORCE with a running baseline */
        POLICY_GRADIENT,
        /** independent epsilon-greedy tabular Q-learners */
        Q_LEARNING
    }

    /** What the players are trained on. */
    public enum Training {
        /** the plain game */
        SELF_PLAY,
        /** the game with player 1 seen through a random symmetry */
        OTHER_PLAY
    }

    /**
     * A relabelling of player 1's view.
     *
     * @param observation the observation player 1 sees for each true observation
     * @param action the true action for each action player 1 picks
     */
    public record Symmetry(int[] observation, int[] action) {
    }

    /** A two-player one-shot cooperative game with a symmetry group. */
    public interface Game {
        /** @return player 0's observations */
        int observations0();

        /** @return player 0's actions */
        int actions0();

        /** @return player 1's observations */
        int observations1();

        /** @return player 1's actions */
        int actions1();

        /**
         * @param o0 player 0's observation
         * @param a0 player 0's action
         * @return player 1's observation
         */
        int observe1(int o0, int a0);

        /**
         * @param o0 player 0's observation
         * @param a0 player 0's action
         * @param a1 player 1's action
         * @return the shared payoff
         */
        double payoff(int o0, int a0, int a1);

        /**
         * @param random the source of randomness
         * @return a uniformly drawn symmetry of the game
         */
        Symmetry sample(SplittableRandom random);
    }

    /**
     * Trained players.
     *
     * @param policy0 player 0's greedy action per observation
     * @param policy1 player 1's greedy action per observation
     */
    public record Players(int[] policy0, int[] policy1) {
    }

    /**
     * Self-play and cross-play over independently trained seeds.
     *
     * @param selfPlay mean return of each seed with itself
     * @param crossPlay mean return over pairs of different seeds
     */
    public record Result(double selfPlay, double crossPlay) {
        @Override
        public String toString() {
            return String.format(java.util.Locale.ROOT, "%.2f/%.2f", selfPlay, crossPlay);
        }
    }

    private OtherPlay() {
    }

    private static int[] identity(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }

    private static void shuffle(int[] p, int from, int to, SplittableRandom r) {
        for (int i = to - 1; i > from; i--) {
            int j = from + r.nextInt(i - from + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
    }

    /**
     * The lever game of Hu et al.: both players pick one of {@code levers} levers at once and
     * are paid the lever's value only if they pick the same one; every lever pays 1 except the
     * last, which pays {@code unique}. The symmetries permute the equal levers.
     *
     * @param levers levers
     * @param unique the unique lever's payoff
     * @return the game
     */
    public static Game levers(int levers, double unique) {
        return new Game() {
            @Override
            public int observations0() {
                return 1;
            }

            @Override
            public int actions0() {
                return levers;
            }

            @Override
            public int observations1() {
                return 1;
            }

            @Override
            public int actions1() {
                return levers;
            }

            @Override
            public int observe1(int o0, int a0) {
                return 0;
            }

            @Override
            public double payoff(int o0, int a0, int a1) {
                return a0 != a1 ? 0 : a0 == levers - 1 ? unique : 1;
            }

            @Override
            public Symmetry sample(SplittableRandom random) {
                int[] a = identity(levers);
                shuffle(a, 0, levers - 1, random);
                return new Symmetry(new int[] {0}, a);
            }

            @Override
            public String toString() {
                return "levers(" + levers + ")";
            }
        };
    }

    /**
     * A signalling game with permutable messages: player 0 sees one of {@code targets} targets
     * and either sends one of {@code targets} free messages or points at a target for
     * {@code pointCost}; player 1 sees what was sent and guesses the target, and a correct
     * guess pays 1 minus the cost. The symmetries permute the messages and relabel the targets
     * (the pointing gestures and the listener's guesses together), so a message language, or
     * pointing at anything but the target itself, is a convention that does not survive.
     *
     * @param targets targets (and messages)
     * @param pointCost the cost of pointing
     * @return the game
     */
    public static Game signalling(int targets, double pointCost) {
        return new Game() {
            @Override
            public int observations0() {
                return targets;
            }

            @Override
            public int actions0() {
                return 2 * targets;
            }

            @Override
            public int observations1() {
                return 2 * targets;
            }

            @Override
            public int actions1() {
                return targets;
            }

            @Override
            public int observe1(int o0, int a0) {
                return a0;
            }

            @Override
            public double payoff(int o0, int a0, int a1) {
                double cost = a0 >= targets ? pointCost : 0;
                return (a1 == o0 ? 1 : 0) - cost;
            }

            @Override
            public Symmetry sample(SplittableRandom random) {
                int[] o = identity(2 * targets);
                shuffle(o, 0, targets, random);
                int[] sigma = identity(targets);
                shuffle(sigma, 0, targets, random);
                int[] back = new int[targets];
                for (int j = 0; j < targets; j++) {
                    o[targets + j] = targets + sigma[j];
                    back[sigma[j]] = j;
                }
                return new Symmetry(o, back);
            }

            @Override
            public String toString() {
                return "signalling(" + targets + ")";
            }
        };
    }

    /**
     * Trains one pair.
     *
     * @param game the game
     * @param learner how the players learn
     * @param training self-play or other-play
     * @param episodes training episodes
     * @param seed the seed
     * @return the greedy players
     */
    public static Players train(Game game, Learner learner, Training training, int episodes, long seed) {
        Objects.requireNonNull(game);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t0 = new double[game.observations0()][game.actions0()];
        double[][] t1 = new double[game.observations1()][game.actions1()];
        // small random initial preferences break ties differently per seed
        for (double[] row : t0) {
            for (int a = 0; a < row.length; a++) {
                row[a] = 1e-3 * r.nextDouble();
            }
        }
        for (double[] row : t1) {
            for (int a = 0; a < row.length; a++) {
                row[a] = 1e-3 * r.nextDouble();
            }
        }
        double baseline = 0;
        double lr = learner == Learner.POLICY_GRADIENT ? 2.0 : 0.1;
        for (int e = 0; e < episodes; e++) {
            Symmetry phi = training == Training.OTHER_PLAY ? game.sample(r) : null;
            double eps = Math.max(0.05, 1 - e / (0.6 * episodes));
            int o0 = r.nextInt(game.observations0());
            int a0 = choose(t0[o0], learner, eps, r);
            int trueO1 = game.observe1(o0, a0);
            int o1 = phi == null ? trueO1 : phi.observation()[trueO1];
            int seen = choose(t1[o1], learner, eps, r);
            int a1 = phi == null ? seen : phi.action()[seen];
            double reward = game.payoff(o0, a0, a1);
            if (learner == Learner.Q_LEARNING) {
                t0[o0][a0] += lr * (reward - t0[o0][a0]);
                t1[o1][seen] += lr * (reward - t1[o1][seen]);
            } else {
                double adv = reward - baseline;
                baseline += 0.05 * (reward - baseline);
                reinforce(t0[o0], a0, lr * adv);
                reinforce(t1[o1], seen, lr * adv);
            }
        }
        int[] p0 = new int[t0.length];
        for (int o = 0; o < t0.length; o++) {
            p0[o] = argmax(t0[o]);
        }
        int[] p1 = new int[t1.length];
        for (int o = 0; o < t1.length; o++) {
            p1[o] = argmax(t1[o]);
        }
        return new Players(p0, p1);
    }

    private static int choose(double[] row, Learner learner, double eps, SplittableRandom r) {
        if (learner == Learner.Q_LEARNING) {
            return r.nextDouble() < eps ? r.nextInt(row.length) : argmax(row);
        }
        double[] p = softmax(row);
        double u = r.nextDouble();
        double c = 0;
        for (int a = 0; a < p.length; a++) {
            c += p[a];
            if (u < c) {
                return a;
            }
        }
        return p.length - 1;
    }

    private static void reinforce(double[] logits, int action, double step) {
        double[] p = softmax(logits);
        for (int a = 0; a < logits.length; a++) {
            logits[a] += step * ((a == action ? 1 : 0) - p[a]);
        }
    }

    private static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            max = Math.max(max, v);
        }
        double[] p = new double[x.length];
        double z = 0;
        for (int i = 0; i < x.length; i++) {
            p[i] = Math.exp(x[i] - max);
            z += p[i];
        }
        for (int i = 0; i < x.length; i++) {
            p[i] /= z;
        }
        return p;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b]) {
                b = i;
            }
        }
        return b;
    }

    /**
     * The exact expected return of a pair of greedy players, over player 0's observations.
     *
     * @param game the game
     * @param first supplies player 0
     * @param second supplies player 1
     * @return the expected payoff
     */
    public static double play(Game game, Players first, Players second) {
        double total = 0;
        for (int o0 = 0; o0 < game.observations0(); o0++) {
            int a0 = first.policy0()[o0];
            int a1 = second.policy1()[game.observe1(o0, a0)];
            total += game.payoff(o0, a0, a1);
        }
        return total / game.observations0();
    }

    /**
     * Trains independent seeds and reports self-play and cross-play.
     *
     * @param game the game
     * @param learner how the players learn
     * @param training self-play or other-play
     * @param episodes training episodes per seed
     * @param seeds independent seeds (at least 2)
     * @return self-play and cross-play returns
     */
    public static Result crossPlay(Game game, Learner learner, Training training, int episodes, int seeds) {
        Players[] players = new Players[seeds];
        for (int s = 0; s < seeds; s++) {
            players[s] = train(game, learner, training, episodes, 1_000L * s + 17);
        }
        double self = 0;
        double cross = 0;
        int pairs = 0;
        for (int i = 0; i < seeds; i++) {
            self += play(game, players[i], players[i]);
            for (int j = 0; j < seeds; j++) {
                if (i != j) {
                    cross += play(game, players[i], players[j]);
                    pairs++;
                }
            }
        }
        return new Result(self / seeds, pairs == 0 ? Double.NaN : cross / pairs);
    }
}
