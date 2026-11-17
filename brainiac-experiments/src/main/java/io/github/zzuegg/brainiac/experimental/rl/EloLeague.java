package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * A league of game players with Elo ratings, as used to track and drive self-play
 * (AlphaStar's league, <a href="https://www.nature.com/articles/s41586-019-1724-z">Vinyals et
 * al. 2019</a>): add frozen snapshots of a learner over time, play rated games among them,
 * and sample opponents (uniformly or, for prioritized fictitious self-play, preferring the
 * ones the learner struggles against).
 */
public final class EloLeague {

    /**
     * A member.
     *
     * @param name the name
     * @param player the player
     */
    public record Member(String name, GamePlayer player) {
    }

    private final Supplier<BoardGame> start;
    private final List<Member> members = new ArrayList<>();
    private final List<Double> ratings = new ArrayList<>();
    private double k = 16;

    private EloLeague(Supplier<BoardGame> start) {
        this.start = Objects.requireNonNull(start);
    }

    /**
     * @param start makes the first position of a game
     * @return an empty league
     */
    public static EloLeague of(Supplier<BoardGame> start) {
        return new EloLeague(start);
    }

    /**
     * @param factor the Elo K factor
     * @return this league
     */
    public EloLeague kFactor(double factor) {
        this.k = factor;
        return this;
    }

    /**
     * @param name a name
     * @param player a (usually frozen) player
     * @return its index
     */
    public int add(String name, GamePlayer player) {
        members.add(new Member(name, player));
        ratings.add(1000.0);
        return members.size() - 1;
    }

    /** @return the members */
    public List<Member> members() {
        return List.copyOf(members);
    }

    /**
     * @param i a member
     * @return its rating
     */
    public double rating(int i) {
        return ratings.get(i);
    }

    /**
     * @param ra one rating
     * @param rb the other
     * @return the expected score of the first, {@code 1 / (1 + 10^((rb - ra) / 400))}
     */
    public static double expected(double ra, double rb) {
        return 1 / (1 + Math.pow(10, (rb - ra) / 400));
    }

    /**
     * Plays a rated game, the first member moving first, and updates both ratings.
     *
     * @param a a member
     * @param b another
     * @param random the random source
     * @return the score of {@code a}: 1, 0.5 or 0
     */
    public double play(int a, int b, SplittableRandom random) {
        int w = GamePlayer.playGame(start.get(), members.get(a).player(), members.get(b).player(), random);
        double score = w == 1 ? 1 : w == -1 ? 0.5 : 0;
        double ea = expected(ratings.get(a), ratings.get(b));
        ratings.set(a, ratings.get(a) + k * (score - ea));
        ratings.set(b, ratings.get(b) + k * ((1 - score) - (1 - ea)));
        return score;
    }

    /**
     * Plays every pair a number of times, both seat orders.
     *
     * @param games games per ordered pair
     * @param random the random source
     */
    public void roundRobin(int games, SplittableRandom random) {
        for (int g = 0; g < games; g++) {
            for (int a = 0; a < members.size(); a++) {
                for (int b = 0; b < members.size(); b++) {
                    if (a != b) {
                        play(a, b, random);
                    }
                }
            }
        }
    }

    /**
     * Samples an opponent, preferring strong ones (prioritized fictitious self-play): the
     * weight of a member is its expected score against a player of the given rating, squared.
     *
     * @param rating the learner's rating
     * @param random the random source
     * @return a member index
     */
    public int sampleOpponent(double rating, SplittableRandom random) {
        double[] w = new double[members.size()];
        double sum = 0;
        for (int i = 0; i < w.length; i++) {
            double e = expected(ratings.get(i), rating);
            w[i] = e * e + 1e-3;
            sum += w[i];
        }
        double u = random.nextDouble() * sum;
        for (int i = 0; i < w.length; i++) {
            u -= w[i];
            if (u <= 0) {
                return i;
            }
        }
        return w.length - 1;
    }
}
