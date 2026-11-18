package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.games.ChanceGame;
import io.github.zzuegg.brainiac.experimental.games.CommitmentAndCorrelation;
import io.github.zzuegg.brainiac.experimental.games.DiceRace;
import io.github.zzuegg.brainiac.experimental.games.MinimaxSearch;
import io.github.zzuegg.brainiac.experimental.games.TdLeaf;
import io.github.zzuegg.brainiac.experimental.rl.Mcts;
import io.github.zzuegg.brainiac.experimental.rl.NInARow;
import io.github.zzuegg.brainiac.experimental.rl.TwoPlayerGame;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToIntFunction;

/**
 * Wave 12 items 12.6.12-12.6.13: game-tree search with learned evaluation, and commitment and
 * correlation in bimatrix games.
 *
 * <ul>
 *   <li>{@link #searchNodes} (12.6.12): nodes visited by negamax, plain alpha-beta and alpha-beta with
 *       iterative deepening, transposition table and history ordering at equal depth (all must agree
 *       on the root value);</li>
 *   <li>{@link #alphaBetaVsMcts} (12.6.12): alpha-beta with a hand-written line evaluation against
 *       MCTS with random rollouts ({@code TwoPlayerGame.randomRollouts}) on an n-in-a-row game;</li>
 *   <li>{@link #tdLeaf} (12.6.12): the score of a TD-Leaf(lambda) self-play-trained evaluation and of
 *       its untrained initialisation against the hand-written evaluation, both searched by
 *       alpha-beta;</li>
 *   <li>{@link #expectimax} (12.6.12): expectimax on a dice race against always-safe, always-risky and
 *       random play;</li>
 *   <li>{@link #commitment} (12.6.13): Stackelberg leader value against the leader's best Nash value,
 *       the welfare of the best correlated equilibrium (LP) against the best Nash welfare, and the
 *       security-game coverage against uniform coverage;</li>
 *   <li>{@link #dynamics} (12.6.13): the CE and CCE gaps of Hedge, optimistic Hedge, regret
 *       matching, internal regret matching and fictitious play.</li>
 * </ul>
 */
public final class GameSolving {

    private GameSolving() {
    }

    /**
     * 12.6.12: search effort at equal depth from the empty board.
     *
     * @param rows board rows
     * @param cols board columns
     * @param k the line length
     * @param depth search plies
     * @return method to {@code {nodes, root value}}
     */
    public static Map<String, double[]> searchNodes(int rows, int cols, int k, int depth) {
        Map<String, double[]> out = new LinkedHashMap<>();
        NInARow start = NInARow.of(rows, cols, k, true);
        for (MinimaxSearch.Method m : MinimaxSearch.Method.values()) {
            MinimaxSearch.Result r = MinimaxSearch.of(m, TdLeaf.handWritten(rows, cols, k)).depth(depth)
                    .search(start);
            out.put(m.name(), new double[] {r.nodes(), r.value()});
        }
        return out;
    }

    /** Plays {@code games} games alternating the first player; returns {wins, draws, losses} of {@code a}. */
    private static double[] match(TwoPlayerGame start, TwoPlayerGame.Player a, TwoPlayerGame.Player b, int games) {
        double[] wdl = new double[3];
        for (int g = 0; g < games; g++) {
            int w = g % 2 == 0 ? TwoPlayerGame.playOut(start, a, b) : -TwoPlayerGame.playOut(start, b, a);
            wdl[w > 0 ? 0 : w == 0 ? 1 : 2]++;
        }
        return wdl;
    }

    /** A player that plays randomly while fewer than two stones are on the board, then follows {@code inner}. */
    private static TwoPlayerGame.Player randomOpening(TwoPlayerGame.Player inner, long seed) {
        TwoPlayerGame.Player random = TwoPlayerGame.Player.random(seed);
        return g -> {
            int stones = 0;
            for (float v : g.encode()) {
                stones += v != 0 ? 1 : 0;
            }
            return stones < 2 ? random.move(g) : inner.move(g);
        };
    }

    /**
     * 12.6.12: alpha-beta against MCTS with random rollouts.
     *
     * @param rows board rows
     * @param cols board columns
     * @param k the line length
     * @param depth alpha-beta plies
     * @param simulations MCTS simulations per move
     * @param games games (first player alternating; the first two plies are random)
     * @return pairing to {@code {wins, draws, losses}} of the first-named player
     */
    public static Map<String, double[]> alphaBetaVsMcts(int rows, int cols, int k, int depth, int simulations,
            int games) {
        NInARow start = NInARow.of(rows, cols, k, true);
        Map<String, double[]> out = new LinkedHashMap<>();
        Mcts<TwoPlayerGame> mcts = Mcts.of(TwoPlayerGame.searchModel(start.actions(),
                TwoPlayerGame.randomRollouts(7))).simulations(simulations);
        SplittableRandom r = new SplittableRandom(11);
        TwoPlayerGame.Player mctsPlayer = g -> mcts.search(g, r).best();
        TwoPlayerGame.Player ab = MinimaxSearch.of(MinimaxSearch.Method.ALPHA_BETA, TdLeaf.handWritten(rows, cols, k))
                .player(depth);
        out.put("alpha-beta vs MCTS (random rollouts)", match(start, randomOpening(ab, 1),
                randomOpening(mctsPlayer, 2), games));
        out.put("alpha-beta vs random", match(start, ab, TwoPlayerGame.Player.random(3), games));
        out.put("MCTS vs random", match(start, mctsPlayer, TwoPlayerGame.Player.random(4), games));
        return out;
    }

    /**
     * 12.6.12: TD-Leaf(lambda).
     *
     * @param rows board rows
     * @param cols board columns
     * @param k the line length
     * @param trainGames self-play games
     * @param depth search plies (training and play)
     * @param games evaluation games against the hand-written evaluation
     * @param seed the seed
     * @return evaluation to {@code {wins, draws, losses}} against the hand-written evaluation
     */
    public static Map<String, double[]> tdLeaf(int rows, int cols, int k, int trainGames, int depth, int games,
            long seed) {
        NInARow start = NInARow.of(rows, cols, k, true);
        int inputs = TdLeaf.lineFeatureCount(rows, cols, k);
        TdLeaf untrained = TdLeaf.of(TdLeaf.lineFeatures(rows, cols, k), inputs, 16, seed);
        TdLeaf trained = TdLeaf.of(TdLeaf.lineFeatures(rows, cols, k), inputs, 16, seed).selfPlay(start, trainGames,
                depth);
        TwoPlayerGame.Player hand = MinimaxSearch.of(MinimaxSearch.Method.ALPHA_BETA, TdLeaf.handWritten(rows, cols, k))
                .player(depth);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("TD-Leaf (trained)", match(start,
                randomOpening(MinimaxSearch.of(MinimaxSearch.Method.ALPHA_BETA, trained.evaluator()).player(depth),
                        seed), randomOpening(hand, seed + 1), games));
        out.put("untrained network", match(start,
                randomOpening(MinimaxSearch.of(MinimaxSearch.Method.ALPHA_BETA, untrained.evaluator()).player(depth),
                        seed), randomOpening(hand, seed + 1), games));
        return out;
    }

    /** Plays a dice race; returns the winner from the first player's view. */
    private static int race(int target, ToIntFunction<ChanceGame> first, ToIntFunction<ChanceGame> second,
            SplittableRandom dice) {
        DiceRace g = DiceRace.start(target);
        while (!g.over()) {
            int a = g.toMove() == 1 ? first.applyAsInt(g) : second.applyAsInt(g);
            g = g.play(a).roll(dice);
        }
        return g.winner();
    }

    /**
     * 12.6.12: expectimax on the dice race.
     *
     * @param target the score to reach
     * @param depth expectimax decision plies
     * @param games games (half as first player)
     * @param seed the dice seed
     * @return opponent to the expectimax win rate
     */
    public static Map<String, Double> expectimax(int target, int depth, int games, long seed) {
        MinimaxSearch.Expectimax search = MinimaxSearch.Expectimax.of(DiceRace::scoreDifference);
        ToIntFunction<ChanceGame> em = g -> search.bestMove(g, depth);
        Map<String, ToIntFunction<ChanceGame>> opponents = new LinkedHashMap<>();
        opponents.put("always safe (one die)", g -> 0);
        opponents.put("always risky (two dice)", g -> 1);
        SplittableRandom coin = new SplittableRandom(seed + 1);
        opponents.put("random", g -> coin.nextInt(2));
        Map<String, Double> out = new LinkedHashMap<>();
        SplittableRandom dice = new SplittableRandom(seed);
        for (Map.Entry<String, ToIntFunction<ChanceGame>> e : opponents.entrySet()) {
            double wins = 0;
            for (int g = 0; g < games; g++) {
                int w = g % 2 == 0 ? race(target, em, e.getValue(), dice) : -race(target, e.getValue(), em, dice);
                wins += w > 0 ? 1 : 0;
            }
            out.put(e.getKey(), wins / games);
        }
        return out;
    }

    /**
     * 12.6.13: commitment and correlation on the classic games and a random one.
     *
     * @param randomSize the side of the random game
     * @param seed the seed
     * @return game to {@code {Stackelberg leader value, best Nash leader value, best CE welfare, best Nash
     *     welfare}}; the row {@code "security game"} holds {@code {defender value with SSE coverage, with
     *     uniform coverage, 0, 0}}
     */
    public static Map<String, double[]> commitment(int randomSize, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (CommitmentAndCorrelation.Bimatrix g : List.of(CommitmentAndCorrelation.chicken(),
                CommitmentAndCorrelation.battleOfTheSexes(), CommitmentAndCorrelation.shapley(),
                CommitmentAndCorrelation.random(randomSize, randomSize, seed))) {
            CommitmentAndCorrelation.Commitment c = CommitmentAndCorrelation.stackelberg(g);
            double nashLeader = Double.NEGATIVE_INFINITY;
            double nashWelfare = Double.NEGATIVE_INFINITY;
            for (double[][] eq : CommitmentAndCorrelation.nashEquilibria(g)) {
                double[][] joint = new double[g.rows()][g.columns()];
                double leader = 0;
                for (int i = 0; i < g.rows(); i++) {
                    for (int j = 0; j < g.columns(); j++) {
                        joint[i][j] = eq[0][i] * eq[1][j];
                        leader += joint[i][j] * g.row()[i][j];
                    }
                }
                nashLeader = Math.max(nashLeader, leader);
                nashWelfare = Math.max(nashWelfare, CommitmentAndCorrelation.welfare(g, joint));
            }
            double ce = CommitmentAndCorrelation.welfare(g, CommitmentAndCorrelation.maxWelfareCe(g));
            out.put(g.name(), new double[] {c.leaderValue(), nashLeader, ce, nashWelfare});
        }
        CommitmentAndCorrelation.SecurityGame sg = CommitmentAndCorrelation.securityGame(8, 3, seed);
        double[] uniform = new double[sg.targets()];
        java.util.Arrays.fill(uniform, 3.0 / sg.targets());
        out.put("security game", new double[] {sg.defenderValue(CommitmentAndCorrelation.securityCoverage(sg)),
            sg.defenderValue(uniform), 0, 0});
        return out;
    }

    /**
     * 12.6.13: no-regret dynamics.
     *
     * @param game the game
     * @param iterations rounds
     * @return dynamics to {@code {CE gap, CCE gap}} of the empirical joint play after {@code iterations}
     */
    public static Map<String, double[]> dynamics(CommitmentAndCorrelation.Bimatrix game, int iterations) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int[] at = {iterations};
        for (CommitmentAndCorrelation.Dynamics d : CommitmentAndCorrelation.Dynamics.values()) {
            double eta = Math.sqrt(8 * Math.log(Math.max(game.rows(), game.columns())) / iterations);
            out.put(d.name(), new double[] {CommitmentAndCorrelation.dynamics(game, d, iterations, eta, at, true)[0],
                CommitmentAndCorrelation.dynamics(game, d, iterations, eta, at, false)[0]});
        }
        return out;
    }
}
