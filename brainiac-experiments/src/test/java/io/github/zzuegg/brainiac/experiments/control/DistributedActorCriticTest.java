package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DistributedActorCriticTest {

    @Test
    void a3cLearnsCartPoleWithWorkerThreads() {
        Map<String, DistributedActorCritic.Outcome> r = DistributedActorCritic.a3c(4, 40_000, 1);
        DistributedActorCritic.Outcome a3c = r.get("A3C");
        DistributedActorCritic.Outcome a2c = r.get("A2C (core)");
        System.out.printf("W6 3.2 A3C CartPole return %.1f in %d ms vs sync A2C %.1f in %d ms (4 envs, 40k steps)%n",
                a3c.greedyReturn(), a3c.millis(), a2c.greedyReturn(), a2c.millis());
        assertTrue(a3c.greedyReturn() > 40, r.toString());
    }

    @Test
    void vtraceCopesWithActorLag() {
        Map<String, Double> r = DistributedActorCritic.lag(new int[] {0, 8, 32}, 16_000, 3);
        System.out.printf("W6 3.3 lag 0/8/32 CartPole return: V-trace %.1f/%.1f/%.1f vs uncorrected %.1f/%.1f/%.1f,"
                + " 1-step IS %.1f/%.1f/%.1f%n", r.get("VTRACE, lag 0"), r.get("VTRACE, lag 8"), r.get("VTRACE, lag 32"),
                r.get("NONE, lag 0"), r.get("NONE, lag 8"), r.get("NONE, lag 32"), r.get("IMPORTANCE_SAMPLING, lag 0"),
                r.get("IMPORTANCE_SAMPLING, lag 8"), r.get("IMPORTANCE_SAMPLING, lag 32"));
        assertTrue(r.get("VTRACE, lag 32") > 40, r.toString());
    }

    @Test
    void retraceEvaluatesTheTargetPolicyOffPolicy() {
        Map<String, Double> early = DistributedActorCritic.retrace(8, 3, 1);
        Map<String, Double> r = DistributedActorCritic.retrace(8, 40, 1);
        System.out.printf("W6 3.3 Retrace Q^pi RMSE after 3/40 passes %.3f/%.3f vs uncorrected 8-step %.3f/%.3f,"
                + " 1-step expected SARSA %.3f/%.3f%n", early.get("Retrace(1)"), r.get("Retrace(1)"),
                early.get("uncorrected n-step"), r.get("uncorrected n-step"), early.get("1-step expected SARSA"),
                r.get("1-step expected SARSA"));
        assertTrue(r.get("Retrace(1)") < r.get("uncorrected n-step"), r.toString());
        assertTrue(early.get("Retrace(1)") < early.get("1-step expected SARSA"), early.toString());
    }
}
