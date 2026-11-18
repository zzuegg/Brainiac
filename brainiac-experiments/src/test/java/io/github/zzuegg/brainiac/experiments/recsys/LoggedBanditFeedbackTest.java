package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class LoggedBanditFeedbackTest {

    private static String line(String title, Map<String, Double> r) {
        StringBuilder s = new StringBuilder(title);
        r.forEach((k, v) -> s.append(String.format(" %s %.4f", k, v)));
        return s.toString();
    }

    @Test
    void doublyRobustAndSnipsBeatTheNaiveAndDirectEstimates() {
        for (int n : new int[] {300, 2000}) {
            for (double beta : new double[] {1, 4}) {
                Map<String, Double> r = LoggedBanditFeedback.evaluation(true, n, beta, 20, 1);
                System.out.println(line("OPE RMSE nonlinear n=" + n + " beta=" + beta + ":", r));
                assertTrue(r.get("DR") < r.get("IPS"), r.toString());
                assertTrue(r.get("SNIPS") < r.get("IPS"), r.toString());
                if (beta == 1) {
                    assertTrue(r.get("DR") < Math.min(r.get("naive"), r.get("DM")), r.toString());
                }
            }
        }
        Map<String, Double> linear = LoggedBanditFeedback.evaluation(false, 2000, 4, 20, 1);
        System.out.println(line("OPE RMSE linear n=2000 beta=4:", linear));
        assertTrue(linear.get("DR") < linear.get("IPS"), linear.toString());
    }

    @Test
    void counterfactualRiskMinimisationImprovesOnTheLogger() {
        Map<String, Double> r = LoggedBanditFeedback.learning(3000, 2);
        System.out.println(line("off-policy learning true value:", r));
        assertTrue(r.get("CRM") > r.get("logging") + 0.02, r.toString());
    }

    @Test
    void contextualBanditsHaveLessRegretThanContextFreeUcb() {
        Map<String, Double> linear = LoggedBanditFeedback.online(false, 10000, 3);
        System.out.println(line("bandit regret linear 10000 rounds:", linear));
        Map<String, Double> nonlinear = LoggedBanditFeedback.online(true, 10000, 3);
        System.out.println(line("bandit regret nonlinear 10000 rounds:", nonlinear));
        assertTrue(linear.get("LinUCB") < linear.get("UCB1 (no context)"), linear.toString());
        assertTrue(linear.get("linear TS") < linear.get("UCB1 (no context)"), linear.toString());
        assertTrue(nonlinear.get("neural-linear") < nonlinear.get("UCB1 (no context)"), nonlinear.toString());
    }
}
