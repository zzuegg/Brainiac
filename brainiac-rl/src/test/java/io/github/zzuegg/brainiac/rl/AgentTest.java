package io.github.zzuegg.brainiac.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Every learner learns something it should. */
class AgentTest {

    @Test
    void qLearningTakesTheEdgeAndSarsaTheSafePathAlongTheCliff() {
        Tabular q = Tabular.qLearning(4).learningRate(0.5).seed(1);
        q.fit(GridWorld.cliff(), 20_000);
        Tabular sarsa = Tabular.sarsa(4).learningRate(0.5).seed(1);
        sarsa.fit(GridWorld.cliff(), 20_000);
        double qReturn = Rl.evaluate(q, GridWorld.cliff(), 1, 1);
        double sarsaReturn = Rl.evaluate(sarsa, GridWorld.cliff(), 1, 1);
        assertEquals(-12, qReturn, 1e-9, "13 steps along the edge");
        assertTrue(sarsaReturn < qReturn && sarsaReturn > -30, "SARSA's path is longer: " + sarsaReturn);
        Tabular expected = Tabular.of(Tabular.Method.EXPECTED_SARSA, 4).learningRate(0.5).seed(1);
        expected.fit(GridWorld.cliff(), 20_000);
        assertTrue(Rl.evaluate(expected, GridWorld.cliff(), 1, 1) > -30);
        assertTrue(q.states() > 30);
        assertTrue(GridWorld.cliff().renderPolicy(q).contains(">"));
    }

    @Test
    void reinforceFindsTheBestBanditArm() {
        Bandit bandit = new Bandit(new double[] {0, 1, 0.2, -0.5}, 0.5);
        Reinforce r = Reinforce.of(Nets.mlp(1, 4, 1), bandit.actionSpace())
                .baseline(Nets.mlp(1, 1, 2)).optimizer(Optimizer.adam(0.05f)).episodesPerUpdate(16).seed(1);
        r.fit(bandit, 3_000);
        int best = 0;
        for (int i = 0; i < 50; i++) {
            best += Math.round(r.act(new float[] {1}, true)[0]) == 1 ? 1 : 0;
        }
        assertTrue(best > 40, "best arm picked " + best + " of 50 times");
    }

    @Test
    void reinforceBalancesTheCartPole() {
        Reinforce r = Reinforce.of(Nets.mlp(4, 2, 1, 32), Space.discrete(2)).baseline(Nets.mlp(4, 1, 2, 32))
                .entropyBonus(0.001f).clipGradientNorm(5).seed(1);
        RlResult result = r.fit(new CartPole(), 40_000);
        assertTrue(result.meanReturn(20) > 100, result.toString());
    }

    @Test
    void ppoSolvesTheCartPole() {
        Ppo ppo = Ppo.of(Nets.mlp(4, 2, 1, 64, 64), Nets.mlp(4, 1, 2, 64, 64), Space.discrete(2))
                .learningRate(1e-3f).targetKl(0.05f).seed(1);
        RlResult result = ppo.fit(CartPole::new, 4, 40_000, RlListener.NONE);
        assertTrue(Rl.evaluate(ppo, new CartPole(), 3, 9) > 400, result.toString());
        assertTrue(ppo.diagnostics().containsKey("approxKl"));
        assertTrue(result.episodes().size() > 50);
        assertEquals(result.episodes().get(result.episodes().size() - 1).totalSteps() <= result.totalSteps(), true);
    }

    @Test
    void a2cAndContinuousPpoImprove() {
        Ppo a2c = Ppo.a2c(Nets.mlp(4, 2, 1, 64), Nets.mlp(4, 1, 2, 64), Space.discrete(2)).seed(1);
        RlResult r = a2c.fit(CartPole::new, 8, 50_000, RlListener.NONE);
        assertTrue(r.meanReturn(50) > 2 * r.episodes().subList(0, 50).stream()
                .mapToDouble(Episode::totalReward).average().orElseThrow(), r.toString());
        Pendulum pendulum = new Pendulum();
        Ppo ppo = Ppo.of(Nets.mlp(3, 1, 1, 64, 64), Nets.mlp(3, 1, 2, 64, 64), pendulum.actionSpace())
                .learningRate(1e-3f).discount(0.9f).rolloutSteps(512).seed(1);
        double before = Rl.evaluate(ppo, pendulum, 3, 5);
        ppo.fit(Pendulum::new, 4, 40_000, RlListener.NONE);
        double after = Rl.evaluate(ppo, pendulum, 3, 5);
        assertTrue(after > before + 200, "pendulum return " + before + " -> " + after);
    }

    @Test
    void dqnBalancesTheCartPole() {
        Dqn dqn = Dqn.of(() -> Nets.mlp(4, 2, Activation.RELU, 1, 64, 64), 2).nSteps(3).seed(1);
        RlResult result = dqn.fit(new CartPole(), 25_000);
        assertTrue(result.bestReturn() >= 300, result.toString());
        assertTrue(dqn.updates() > 20_000);
        assertTrue(Double.isFinite(dqn.lastLoss()));
    }

    @Test
    void dqnVariantsRun() {
        for (String variant : List.of("dueling", "prioritized", "soft", "plain")) {
            Dqn dqn = Dqn.of(() -> Nets.mlp(4, variant.equals("dueling") ? 3 : 2, Activation.RELU, 1, 32), 2)
                    .learningStarts(100).seed(2);
            switch (variant) {
                case "dueling" -> dqn.dueling(true);
                case "prioritized" -> dqn.prioritized(0.6, Schedule.linear(0.4, 1, 2_000));
                case "soft" -> dqn.softTargetUpdate(0.01f);
                default -> dqn.doubleQ(false).huberDelta(0);
            }
            RlResult r = dqn.fit(new CartPole(), 2_000);
            assertTrue(Double.isFinite(dqn.lastLoss()), variant);
            assertEquals(2, dqn.actionValues(new float[4]).length, variant);
            assertTrue(r.totalSteps() == 2_000, variant);
        }
        Dqn wrong = Dqn.of(() -> Nets.mlp(4, 2, 1), 2).dueling(true);
        assertThrows(IllegalArgumentException.class, () -> wrong.fit(new CartPole(), 10));
        assertThrows(IllegalArgumentException.class, () -> Dqn.of(() -> Nets.mlp(4, 3, 1), 3).fit(new CartPole(), 10));
    }

    @Test
    void randomAgentAndWatching() {
        Agent random = Agent.random(Space.discrete(2), 1);
        double r = Rl.evaluate(random, new CartPole(), 20, 1);
        assertTrue(r > 8 && r < 50, "random cart-pole return " + r);
        List<String> frames = Rl.watch(random, new CartPole(), 1, 5);
        assertEquals(5, frames.size());
        assertTrue(frames.get(0).contains("angle"));
    }

    @Test
    void resultsAverage() {
        RlResult r = new RlResult();
        for (int i = 1; i <= 4; i++) {
            r.add(new Episode(i, 1, i, i));
        }
        assertEquals(3.5, r.meanReturn(2), 1e-9);
        assertEquals(4, r.bestReturn(), 1e-9);
        assertEquals(3, r.movingAverage(2).length);
        assertEquals(2.5, r.movingAverage(2)[1], 1e-9);
    }
}
