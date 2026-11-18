package io.github.zzuegg.brainiac.experiments.bandits;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.bandit.BanditPolicy;
import io.github.zzuegg.brainiac.experimental.bandit.StochasticBandits.Method;
import io.github.zzuegg.brainiac.experimental.data.BanditSuites;
import io.github.zzuegg.brainiac.rl.Bandit;
import io.github.zzuegg.brainiac.rl.Learner;

import java.util.Map;

import org.junit.jupiter.api.Test;

class StochasticBanditsTest {

    @Test
    void klUcbThompsonAndImedBeatUcb1OnLowMeanArms() {
        BanditSuites.Instance low = BanditSuites.lowMeanBernoulli(5, 0.1, 0.05);
        Map<String, double[]> t = StochasticBandits.stochastic(low, low.means(0), 5_000, 40);
        System.out.printf("11.6.1 regret@5000 (5 arms, best 0.1, gap 0.05): %s%n", StochasticBandits.line(t));
        double ucb = t.get("UCB1")[0];
        assertTrue(t.get("KL_UCB")[0] < ucb && t.get("THOMPSON_BETA")[0] < ucb && t.get("IMED")[0] < ucb,
                StochasticBandits.line(t));
        assertTrue(t.get("THOMPSON_BETA")[0] < 0.7 * ucb, StochasticBandits.line(t));
    }

    @Test
    void adversaryExploitsIndexPoliciesButNotExp3() {
        Map<String, double[]> adv = StochasticBandits.regimes(StochasticBandits.Regime.ADAPTIVE_ADVERSARY, 4, 3_000, 10);
        Map<String, double[]> sto = StochasticBandits.regimes(StochasticBandits.Regime.STOCHASTIC, 4, 3_000, 10);
        Map<String, double[]> sw = StochasticBandits.regimes(StochasticBandits.Regime.SWITCHING, 4, 3_000, 10);
        System.out.printf("11.6.2 regret@3000 adversary: %s| stochastic: %s| switching: %s%n",
                StochasticBandits.line(adv), StochasticBandits.line(sto), StochasticBandits.line(sw));
        assertTrue(adv.get("KL_UCB")[0] > 2 * adv.get("EXP3")[0], StochasticBandits.line(adv));
        assertTrue(adv.get("TSALLIS_INF")[0] < adv.get("KL_UCB")[0], StochasticBandits.line(adv));
        assertTrue(sto.get("TSALLIS_INF")[0] < sto.get("EXP3")[0], StochasticBandits.line(sto));
    }

    @Test
    void policyDropsIntoCoreBanditAsLearner() {
        Bandit env = new Bandit(new double[] {0.1, 0.9, 0.2}, 0.1);
        BanditPolicy p = io.github.zzuegg.brainiac.experimental.bandit.StochasticBandits.of(Method.KL_UCB, 3, 500, 1)
                .gaussian(0.1);
        Learner l = p.learner(1);
        l.fit(env, 500);
        assertTrue(Math.round(l.act(new float[1], false)[0]) == env.bestArm());
    }
}
