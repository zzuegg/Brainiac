package io.github.zzuegg.brainiac.experiments.privacy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class FederatedShootoutTest {

    @Test
    void scaffoldAndServerOptimizersCorrectDriftUnderLabelSkew() {
        double[] scaffoldGain = new double[3];
        for (int s = 0; s < 3; s++) {
            List<FederatedShootout.Row> rows = FederatedShootout.drift(FederatedShootout.Task.DIGITS, 0.1, 20, 0.2,
                    30, 10, false, 0.5, 1 + s);
            for (FederatedShootout.Row r : rows) {
                System.out.printf("Fed drift digits a=0.1 seed %d  %-11s acc %.3f  rounds to 0.5 %d%n", 1 + s,
                        r.method(), r.accuracy(), r.roundsToTarget());
            }
            scaffoldGain[s] = rows.get(3).accuracy() - rows.get(1).accuracy();
            assertTrue(rows.get(0).accuracy() > 0.8, rows.get(0).toString());
            assertTrue(rows.get(1).accuracy() > 0.3, rows.get(1).toString());
        }
        double mean = (scaffoldGain[0] + scaffoldGain[1] + scaffoldGain[2]) / 3;
        System.out.printf("Fed drift: SCAFFOLD - FedAvg accuracy over 3 seeds %+.3f%n", mean);
        assertTrue(mean > 0.05, "SCAFFOLD gain " + mean);
    }

    @Test
    void allMethodsTieOnIidClientsAndFedNovaHandlesVariableSteps() {
        List<FederatedShootout.Row> iid = FederatedShootout.drift(FederatedShootout.Task.SHAPES, 100, 20, 0.2, 30, 10,
                false, 0.8, 1);
        List<FederatedShootout.Row> varying = FederatedShootout.drift(FederatedShootout.Task.DIGITS, 0.1, 20, 0.2, 30,
                10, true, 0.8, 1);
        for (FederatedShootout.Row r : iid) {
            System.out.printf("Fed shapes a=100      %-11s acc %.3f%n", r.method(), r.accuracy());
        }
        for (FederatedShootout.Row r : varying) {
            System.out.printf("Fed digits a=0.1 tau~U[1,20] %-11s acc %.3f%n", r.method(), r.accuracy());
        }
        double lo = 1;
        double hi = 0;
        for (FederatedShootout.Row r : iid.subList(1, 4)) {
            lo = Math.min(lo, r.accuracy());
            hi = Math.max(hi, r.accuracy());
        }
        assertTrue(hi - lo < 0.1, "FedAvg, FedProx and SCAFFOLD should tie on iid clients: " + lo + ".." + hi);
        assertTrue(varying.get(7).accuracy() > 0.3, varying.get(7).toString());
    }

    @Test
    void fedAdamIsLeastSensitiveToTheServerRate() {
        float[] rates = {0.1f, 0.3f, 1f, 3f};
        double[][] acc = FederatedShootout.serverRates(0.1, rates, 25, 1);
        String[] names = {"server SGD", "FedAvgM", "FedAdam"};
        for (int m = 0; m < 3; m++) {
            double min = 1;
            double max = 0;
            for (double a : acc[m]) {
                min = Math.min(min, a);
                max = Math.max(max, a);
            }
            System.out.printf("Fed server-rate sweep %-20s acc %s  spread %.3f%n", names[m],
                    java.util.Arrays.toString(java.util.Arrays.stream(acc[m]).map(v -> Math.round(v * 1000) / 1000.0)
                            .toArray()), max - min);
        }
        assertEquals(3, acc.length);
    }

    @Test
    void personalisationBeatsOneGlobalModelUnderConceptShift() {
        List<FederatedShootout.Personal> shift = FederatedShootout.personalization(true, 20, 30, 1);
        List<FederatedShootout.Personal> skew = FederatedShootout.personalization(false, 20, 30, 1);
        for (FederatedShootout.Personal p : shift) {
            System.out.printf("Personal concept shift %-19s mean %.3f  worst decile %.3f%n", p.method(), p.mean(),
                    p.worstDecile());
        }
        for (FederatedShootout.Personal p : skew) {
            System.out.printf("Personal label skew    %-19s mean %.3f  worst decile %.3f%n", p.method(), p.mean(),
                    p.worstDecile());
        }
        double fedAvg = shift.get(0).mean();
        double fedRep = shift.get(4).mean();
        assertTrue(fedRep > fedAvg + 0.1, "FedRep " + fedRep + " vs FedAvg " + fedAvg);
        assertTrue(skew.get(1).mean() > skew.get(0).mean(), "fine-tuning helps under label skew");
    }

    @Test
    void userLevelDpCostsLittleWithManyUsers() {
        List<FederatedShootout.Private> rows = FederatedShootout.userLevelDp(200, new double[] {2, 8}, 40, 0.1, 100, 1);
        for (FederatedShootout.Private p : rows) {
            System.out.printf("User-level DP %-28s eps %6.2f  acc %.3f  audit eps >= %.2f%n", p.method(), p.epsilon(),
                    p.accuracy(), p.auditEpsilon());
        }
        assertTrue(rows.get(0).accuracy() > 0.6, rows.get(0).toString());
        assertTrue(rows.get(2).epsilon() <= 8.001, rows.get(2).toString());
        assertTrue(rows.get(1).auditEpsilon() <= rows.get(1).epsilon(), rows.get(1).toString());
    }

    @Test
    void secureAggregationIsExactUnderDropout() {
        List<FederatedShootout.SecureRow> rows = FederatedShootout.secureAggregation(50, 500,
                new double[] {0, 0.1, 0.3}, 1);
        for (FederatedShootout.SecureRow r : rows) {
            System.out.printf("SecAgg dropout %.1f  survivors %d  max error %.2e  recovered secrets %d  %.1f ms%n",
                    r.dropout(), r.survivors(), r.maxError(), r.recovered(), r.millis());
            assertTrue(r.maxError() < 1e-4, r.toString());
        }
    }
}
