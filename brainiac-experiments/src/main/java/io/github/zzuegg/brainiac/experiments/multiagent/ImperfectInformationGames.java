package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.rl.CounterfactualRegret;
import io.github.zzuegg.brainiac.experimental.rl.Exploitability;
import io.github.zzuegg.brainiac.experimental.rl.ExtensiveFormGame;
import io.github.zzuegg.brainiac.experimental.rl.GameTree;
import io.github.zzuegg.brainiac.experimental.rl.MatrixGame;
import io.github.zzuegg.brainiac.experimental.rl.NeuralFictitiousSelfPlay;
import io.github.zzuegg.brainiac.experimental.rl.PolicySpaceResponseOracles;
import io.github.zzuegg.brainiac.experimental.rl.RegularizedEquilibriumLearner;

import java.util.ArrayList;
import java.util.List;

/**
 * Equilibrium finding in Kuhn and Leduc poker, measured by exact exploitability in
 * milli-big-blinds per hand ({@link Exploitability}).
 * <ul>
 *   <li>{@link #regretMinimisation}: CFR, CFR+ and external-sampling MCCFR against the
 *       iterations, the nodes they touch and the wall time (W7 5.1).</li>
 *   <li>{@link #fictitiousSelfPlay}: NFSP's average-policy networks against DQN self-play,
 *       the greedy best-response networks alone, at equal environment steps (W7 5.2).</li>
 *   <li>{@link #responseOracles}: PSRO with a Nash meta-solver (double oracle), with a
 *       uniform meta-solver (a fictitious-play league), with a policy-gradient oracle, and
 *       naive self-play that responds to the latest policy (W7 5.3); {@link #nonTransitive}
 *       repeats this on rock-paper-scissors with skill levels.</li>
 *   <li>{@link #lastIterate}: the last iterate of magnetic mirror descent, R-NaD and Nash
 *       policy gradient against unregularised multiplicative weights and CFR+'s current
 *       strategy (W7 5.4).</li>
 * </ul>
 * Uniform random play is the floor every row is compared with.
 */
public final class ImperfectInformationGames {

    /**
     * One measured method.
     *
     * @param method the method
     * @param milliBlinds exploitability of its output, in milli-big-blinds per hand
     * @param work its work measure (iterations, nodes touched or environment steps)
     * @param seconds wall time
     */
    public record Row(String method, double milliBlinds, long work, double seconds) {
    }

    private ImperfectInformationGames() {
    }

    private static double mbb(GameTree tree, double[][] policy) {
        return Exploitability.milliBlinds(Exploitability.of(tree, policy));
    }

    /**
     * @param game the game
     * @return the exploitability of uniform random play, mbb/hand
     */
    public static double uniform(ExtensiveFormGame game) {
        GameTree tree = GameTree.compile(game);
        return mbb(tree, tree.uniformPolicy());
    }

    /**
     * Runs the three CFR variants; MCCFR gets {@code mccfrFactor} times as many (cheap)
     * iterations.
     *
     * @param game the game
     * @param iterations full-width iterations of CFR and CFR+
     * @param mccfrFactor MCCFR iterations per full-width iteration
     * @param seed MCCFR's sampling seed
     * @return one row per variant, work = nodes touched
     */
    public static List<Row> regretMinimisation(ExtensiveFormGame game, int iterations, int mccfrFactor, long seed) {
        GameTree tree = GameTree.compile(game);
        List<Row> rows = new ArrayList<>();
        for (CounterfactualRegret solver : new CounterfactualRegret[] {CounterfactualRegret.cfr(tree),
            CounterfactualRegret.cfrPlus(tree), CounterfactualRegret.mccfr(tree, seed)}) {
            long t0 = System.nanoTime();
            solver.run(solver.variant() == CounterfactualRegret.Variant.MCCFR ? iterations * mccfrFactor : iterations);
            double seconds = (System.nanoTime() - t0) / 1e9;
            rows.add(new Row(solver.variant().toString(), mbb(tree, solver.averagePolicy()), solver.nodesTouched(),
                    seconds));
        }
        return rows;
    }

    /**
     * NFSP against DQN self-play at the same number of episodes.
     *
     * @param game the game
     * @param episodes self-play episodes
     * @param seed the seed
     * @return rows "NFSP average policy", "NFSP best response" (greedy Q, eta = 0.1) and
     *     "DQN self-play" (eta = 1, greedy Q); work = environment steps
     */
    public static List<Row> fictitiousSelfPlay(ExtensiveFormGame game, int episodes, long seed) {
        GameTree tree = GameTree.compile(game);
        List<Row> rows = new ArrayList<>();
        long t0 = System.nanoTime();
        NeuralFictitiousSelfPlay nfsp = NeuralFictitiousSelfPlay.of(tree, 64, seed).anticipatory(0.1).trainEvery(16)
                .learningRates(0.01f, 0.005f).fit(episodes);
        double seconds = (System.nanoTime() - t0) / 1e9;
        rows.add(new Row("NFSP average policy", mbb(tree, nfsp.averagePolicy()), nfsp.steps(), seconds));
        rows.add(new Row("NFSP best response", mbb(tree, nfsp.bestResponsePolicy()), nfsp.steps(), seconds));
        t0 = System.nanoTime();
        NeuralFictitiousSelfPlay dqn = NeuralFictitiousSelfPlay.of(tree, 64, seed).anticipatory(1).trainEvery(16)
                .learningRates(0.01f, 0.005f).fit(episodes);
        rows.add(new Row("DQN self-play", mbb(tree, dqn.bestResponsePolicy()), dqn.steps(),
                (System.nanoTime() - t0) / 1e9));
        return rows;
    }

    /**
     * PSRO variants with the same number of iterations.
     *
     * @param game the game
     * @param iterations oracle calls per player
     * @return rows per variant; work = population size
     */
    public static List<Row> responseOracles(ExtensiveFormGame game, int iterations) {
        GameTree tree = GameTree.compile(game);
        List<Row> rows = new ArrayList<>();
        Object[][] variants = {
            {"PSRO, Nash meta-solver (double oracle)", PolicySpaceResponseOracles.MetaSolver.NASH,
                PolicySpaceResponseOracles.Oracle.exact()},
            {"PSRO, Nash meta-solver, policy-gradient oracle", PolicySpaceResponseOracles.MetaSolver.NASH,
                PolicySpaceResponseOracles.Oracle.policyGradient(100, 5)},
            {"league (uniform meta-solver, fictitious play)", PolicySpaceResponseOracles.MetaSolver.UNIFORM,
                PolicySpaceResponseOracles.Oracle.exact()},
            {"naive self-play (respond to latest)", PolicySpaceResponseOracles.MetaSolver.LATEST,
                PolicySpaceResponseOracles.Oracle.exact()}};
        for (Object[] v : variants) {
            long t0 = System.nanoTime();
            PolicySpaceResponseOracles psro = PolicySpaceResponseOracles
                    .of(tree, (PolicySpaceResponseOracles.MetaSolver) v[1])
                    .oracle((PolicySpaceResponseOracles.Oracle) v[2]).run(iterations);
            rows.add(new Row((String) v[0], mbb(tree, psro.policy()), psro.populationSize(0),
                    (System.nanoTime() - t0) / 1e9));
        }
        return rows;
    }

    /**
     * PSRO against naive self-play on rock-paper-scissors with skill levels, where the type
     * is non-transitive and the skill transitive.
     *
     * @param levels skill levels per type
     * @param iterations oracle calls
     * @return rows "PSRO-Nash", "league" and "self-play"; milliBlinds holds the
     *     exploitability times 1000
     */
    public static List<Row> nonTransitive(int levels, int iterations) {
        GameTree tree = GameTree.compile(MatrixGame.rockPaperScissorsWithSkill(levels));
        List<Row> rows = new ArrayList<>();
        String[] names = {"PSRO-Nash", "league", "self-play"};
        PolicySpaceResponseOracles.MetaSolver[] solvers = PolicySpaceResponseOracles.MetaSolver.values();
        for (int i = 0; i < 3; i++) {
            long t0 = System.nanoTime();
            PolicySpaceResponseOracles psro = PolicySpaceResponseOracles.of(tree, solvers[i]).run(iterations);
            rows.add(new Row(names[i], mbb(tree, psro.policy()), psro.populationSize(0),
                    (System.nanoTime() - t0) / 1e9));
        }
        return rows;
    }

    /**
     * Last-iterate exploitability of the regularised learners and the cycling baselines,
     * at the same number of full-tree iterations.
     *
     * @param game the game
     * @param iterations iterations
     * @return rows per method; work = iterations
     */
    public static List<Row> lastIterate(ExtensiveFormGame game, int iterations) {
        GameTree tree = GameTree.compile(game);
        List<Row> rows = new ArrayList<>();
        RegularizedEquilibriumLearner[] learners = {
            RegularizedEquilibriumLearner.unregularized(tree, 0.05),
            RegularizedEquilibriumLearner.mmd(tree, 0.1, 0.05),
            RegularizedEquilibriumLearner.rnad(tree, 0.5, 0.05, 100),
            RegularizedEquilibriumLearner.nashPg(tree, 0.5, 0.5, 100)};
        String[] names = {"multiplicative weights (no magnet)", "MMD (uniform magnet)", "R-NaD (moving magnet)",
            "Nash policy gradient"};
        for (int i = 0; i < learners.length; i++) {
            long t0 = System.nanoTime();
            learners[i].run(iterations);
            rows.add(new Row(names[i], mbb(tree, learners[i].policy()), iterations, (System.nanoTime() - t0) / 1e9));
        }
        long t0 = System.nanoTime();
        CounterfactualRegret cfr = CounterfactualRegret.cfrPlus(tree).run(iterations);
        double seconds = (System.nanoTime() - t0) / 1e9;
        rows.add(new Row("CFR+ current strategy", mbb(tree, cfr.currentPolicy()), iterations, seconds));
        rows.add(new Row("CFR+ average strategy", mbb(tree, cfr.averagePolicy()), iterations, seconds));
        return rows;
    }
}
