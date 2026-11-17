package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.rl.BoardGame;
import io.github.zzuegg.brainiac.experimental.rl.EloLeague;
import io.github.zzuegg.brainiac.experimental.rl.GamePlayer;
import io.github.zzuegg.brainiac.experimental.rl.TicTacToe;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * SPIRAL, self-play on zero-sum text games (Liu et al., "SPIRAL: self-play on zero-sum games
 * incentivizes reasoning via multi-agent multi-turn reinforcement learning",
 * <a href="https://arxiv.org/abs/2506.24119">arXiv:2506.24119</a>): one language model plays
 * <em>both</em> sides of a game written as text, and is trained by REINFORCE with
 * <em>role-conditioned advantage estimation</em> (RAE): each role keeps its own running mean of
 * returns as a baseline, because in self-play the first player's expected outcome is not the
 * second's, and a shared baseline biases both.
 *
 * <p>The game is {@link TicTacToe} as text over the {@link ArithmeticTask} characters: the
 * prompt is {@code [r]} with the role digit, the nine cells ({@code ?} empty, {@code +} first
 * player, {@code *} second) and {@code =}; the move is one digit. Sampling is restricted to legal
 * moves. Compared with training against a fixed random opponent, and with self-play without RAE
 * (one shared baseline); the players then meet a random and a perfect (minimax) player in an
 * {@link EloLeague}. Transfer is measured as the change in {@link ArithmeticTask} accuracy of a
 * model first supervised on arithmetic, since the paper reports that game play improves math.
 */
public final class SelfPlayGames {
    private static final CharacterTokenizer TOK = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private static final int OPEN = TOK.encode("[")[0];
    private static final int CLOSE = TOK.encode("]")[0];
    private static final int EQUALS = TOK.encode("=")[0];
    private static final int EMPTY = TOK.encode("?")[0];
    private static final int FIRST = TOK.encode("+")[0];
    private static final int SECOND = TOK.encode("*")[0];

    /** How the learner is trained. */
    public enum Mode {
        /** Self-play with a baseline per role (SPIRAL). */
        SELF_PLAY_RAE,
        /** Self-play with one shared baseline. */
        SELF_PLAY_SHARED,
        /** Against a fixed uniformly random opponent, the learner's seat drawn per game. */
        FIXED_OPPONENT
    }

    /**
     * A trained player.
     *
     * @param mode how it was trained
     * @param elo its league rating
     * @param winVsRandom its score against the random player (win 1, draw 0.5), both seats
     * @param drawVsMinimax its score against the perfect player, both seats
     * @param arithmeticBefore greedy addition accuracy before game training
     * @param arithmeticAfter and after
     */
    public record Player(Mode mode, double elo, double winVsRandom, double drawVsMinimax, double arithmeticBefore,
            double arithmeticAfter) {
    }

    /**
     * The league.
     *
     * @param players the trained players
     * @param randomElo the random player's rating
     * @param minimaxElo the perfect player's rating
     */
    public record Result(List<Player> players, double randomElo, double minimaxElo) {
    }

    private SelfPlayGames() {
    }

    /**
     * The text prompt of a position.
     *
     * @param game the position
     * @return {@code [r]} cells {@code =}
     */
    public static int[] prompt(BoardGame game) {
        String key = game.key();
        int[] p = new int[13];
        p[0] = OPEN;
        p[1] = TOK.encode(String.valueOf(game.toMove()))[0];
        p[2] = CLOSE;
        for (int i = 0; i < 9; i++) {
            char c = key.charAt(i);
            p[3 + i] = c == '0' ? EMPTY : c == '1' ? FIRST : SECOND;
        }
        p[12] = EQUALS;
        return p;
    }

    /**
     * The model as a player: it samples among the legal moves at a temperature.
     *
     * @param lm the model
     * @param temperature 0 for the most likely legal move
     * @return the player
     */
    public static GamePlayer player(WindowLm lm, double temperature) {
        return (position, random) -> pick(lm.logits(prompt(position)), position.legalMoves(), temperature, random);
    }

    private static int pick(float[] logits, List<Integer> legal, double temperature, SplittableRandom random) {
        double[] w = new double[legal.size()];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < w.length; i++) {
            max = Math.max(max, logits[legal.get(i)]);
        }
        int best = 0;
        double total = 0;
        for (int i = 0; i < w.length; i++) {
            if (logits[legal.get(i)] > logits[legal.get(best)]) {
                best = i;
            }
            w[i] = temperature == 0 ? 0 : Math.exp((logits[legal.get(i)] - max) / temperature);
            total += w[i];
        }
        if (temperature == 0) {
            return legal.get(best);
        }
        double u = random.nextDouble() * total;
        for (int i = 0; i < w.length; i++) {
            u -= w[i];
            if (u <= 0) {
                return legal.get(i);
            }
        }
        return legal.get(w.length - 1);
    }

    /** @return a perfect TicTacToe player (memoized minimax, ties broken at random) */
    public static GamePlayer minimax() {
        Map<String, Integer> memo = new HashMap<>();
        return (position, random) -> {
            List<Integer> best = new ArrayList<>();
            int bestValue = Integer.MIN_VALUE;
            for (int m : position.legalMoves()) {
                int v = -value(position.play(m), memo);
                if (v > bestValue) {
                    bestValue = v;
                    best.clear();
                }
                if (v == bestValue) {
                    best.add(m);
                }
            }
            return best.get(random.nextInt(best.size()));
        };
    }

    /** The value of a position for the player to move: 1 win, 0 draw, -1 loss. */
    private static int value(BoardGame g, Map<String, Integer> memo) {
        Integer known = memo.get(g.key());
        if (known != null) {
            return known;
        }
        int w = g.winner();
        int v;
        if (w == -1) {
            v = 0;
        } else if (w != 0) {
            v = w == g.toMove() ? 1 : -1;
        } else {
            v = Integer.MIN_VALUE;
            for (int m : g.legalMoves()) {
                v = Math.max(v, -value(g.play(m), memo));
            }
        }
        memo.put(g.key(), v);
        return v;
    }

    /**
     * Trains a model that was first supervised on addition.
     *
     * @param lm the model, trained in place
     * @param mode the training mode
     * @param iterations policy-gradient updates
     * @param games games per update, played in lockstep
     * @param learningRate the learning rate
     * @param seed the seed
     */
    public static void train(WindowLm lm, Mode mode, int iterations, int games, float learningRate, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        double[] baseline = new double[3];
        for (int it = 0; it < iterations; it++) {
            BoardGame[] g = new BoardGame[games];
            int[] learnerSeat = new int[games];
            List<List<int[]>> prompts = new ArrayList<>();
            List<List<int[]>> moves = new ArrayList<>();
            List<List<Integer>> roles = new ArrayList<>();
            for (int i = 0; i < games; i++) {
                g[i] = TicTacToe.start();
                learnerSeat[i] = mode == Mode.FIXED_OPPONENT ? 1 + random.nextInt(2) : 0;
                prompts.add(new ArrayList<>());
                moves.add(new ArrayList<>());
                roles.add(new ArrayList<>());
            }
            while (true) {
                List<int[]> batch = new ArrayList<>();
                List<Integer> who = new ArrayList<>();
                for (int i = 0; i < games; i++) {
                    if (g[i].isOver()) {
                        continue;
                    }
                    if (learnerSeat[i] != 0 && g[i].toMove() != learnerSeat[i]) {
                        List<Integer> legal = g[i].legalMoves();
                        g[i] = g[i].play(legal.get(random.nextInt(legal.size())));
                        continue;
                    }
                    batch.add(prompt(g[i]));
                    who.add(i);
                }
                if (batch.isEmpty()) {
                    boolean live = false;
                    for (BoardGame x : g) {
                        live |= !x.isOver();
                    }
                    if (!live) {
                        break;
                    }
                    continue;
                }
                float[] logits = lm.logits(batch).value().toArray();
                int vocab = lm.vocabSize();
                for (int j = 0; j < who.size(); j++) {
                    int i = who.get(j);
                    float[] row = java.util.Arrays.copyOfRange(logits, j * vocab, (j + 1) * vocab);
                    int move = pick(row, g[i].legalMoves(), 1.0, random);
                    prompts.get(i).add(batch.get(j));
                    moves.get(i).add(new int[] {move});
                    roles.get(i).add(g[i].toMove());
                    g[i] = g[i].play(move);
                }
            }
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            List<Float> adv = new ArrayList<>();
            double[] sum = new double[3];
            int[] count = new int[3];
            for (int i = 0; i < games; i++) {
                int w = g[i].winner();
                for (int role = 1; role <= 2; role++) {
                    if (!roles.get(i).contains(role)) {
                        continue;
                    }
                    double ret = w == -1 ? 0 : w == role ? 1 : -1;
                    int slot = mode == Mode.SELF_PLAY_SHARED ? 0 : role;
                    sum[slot] += ret;
                    count[slot]++;
                    for (int t = 0; t < roles.get(i).size(); t++) {
                        if (roles.get(i).get(t) == role) {
                            ps.add(prompts.get(i).get(t));
                            cs.add(moves.get(i).get(t));
                            adv.add((float) (ret - baseline[slot]));
                        }
                    }
                }
            }
            for (int s = 0; s < 3; s++) {
                if (count[s] > 0) {
                    baseline[s] = 0.9 * baseline[s] + 0.1 * sum[s] / count[s];
                }
            }
            float[] a = new float[adv.size()];
            for (int i = 0; i < a.length; i++) {
                a[i] = adv.get(i);
            }
            Var logp = lm.logProbs(ps, cs);
            Var loss = logp.mul(Var.of(Tensor.of(Shape.of(a.length, 1), a))).mean().neg();
            loss.backward();
            opt.step(lm.parameters());
        }
    }

    /**
     * Scores {@code a} against {@code b} over games in both seats.
     *
     * @param a a player
     * @param b another
     * @param games games per seat
     * @param random the random source
     * @return {@code a}'s mean score (win 1, draw 0.5)
     */
    public static double score(GamePlayer a, GamePlayer b, int games, SplittableRandom random) {
        double s = 0;
        for (int i = 0; i < games; i++) {
            int w1 = GamePlayer.playGame(TicTacToe.start(), a, b, random);
            s += w1 == 1 ? 1 : w1 == -1 ? 0.5 : 0;
            int w2 = GamePlayer.playGame(TicTacToe.start(), b, a, random);
            s += w2 == 2 ? 1 : w2 == -1 ? 0.5 : 0;
        }
        return s / (2 * games);
    }

    /**
     * Runs the comparison.
     *
     * @param iterations policy-gradient updates per learner
     * @param games games per update
     * @param seed the seed
     * @return ratings, scores and arithmetic transfer per mode
     */
    public static Result run(int iterations, int games, long seed) {
        ArithmeticTask add = ArithmeticTask.addition(6);
        WindowLm start = WindowLm.of(TOK.size(), 13, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 17);
        List<int[]> ps = new ArrayList<>();
        List<int[]> gold = new ArrayList<>();
        for (int[] p : add.allPrompts()) {
            if (r.nextDouble() < 0.5) {
                ps.add(p);
                gold.add(add.solution(p));
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 60; s++) {
            start.supervisedStep(ps, gold, opt);
        }
        double before = ArithmeticRl.accuracy(start, add);
        EloLeague league = EloLeague.of(TicTacToe::start).kFactor(8);
        int randomId = league.add("random", GamePlayer.random());
        int minimaxId = league.add("minimax", minimax());
        List<WindowLm> trained = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        for (Mode mode : Mode.values()) {
            WindowLm lm = start.copy();
            train(lm, mode, iterations, games, 1e-2f, seed + mode.ordinal());
            trained.add(lm);
            ids.add(league.add(mode.name(), player(lm, 0.3)));
        }
        SplittableRandom lr = new SplittableRandom(seed + 5);
        league.roundRobin(20, lr);
        List<Player> players = new ArrayList<>();
        for (int i = 0; i < trained.size(); i++) {
            GamePlayer p = player(trained.get(i), 0.3);
            players.add(new Player(Mode.values()[i], league.rating(ids.get(i)),
                    score(p, GamePlayer.random(), 50, lr), score(p, minimax(), 50, lr), before,
                    ArithmeticRl.accuracy(trained.get(i), add)));
        }
        return new Result(players, league.rating(randomId), league.rating(minimaxId));
    }
}
