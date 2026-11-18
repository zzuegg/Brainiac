package io.github.zzuegg.brainiac.experiments.privacy;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MechanismShootoutTest {

    @Test
    void localDpFrequencyOracleCrossoverAndCentralGap() {
        for (int k : new int[] {16, 256}) {
            for (double eps : new double[] {1, 4}) {
                List<MechanismShootout.Frequency> rows = MechanismShootout.localDp(20000, k, true, eps, 1);
                StringBuilder b = new StringBuilder();
                for (MechanismShootout.Frequency f : rows) {
                    b.append(String.format(" %s %.2e/%.2e r%.1f;", f.mechanism(), f.mse(), f.projectedMse(),
                            f.topRecall()));
                }
                System.out.printf("Local DP zipf k=%d eps=%.0f MSE raw/projected, top-10 recall:%s%n", k, eps, b);
                double grr = rows.get(0).mse();
                double oue = rows.get(2).mse();
                double olh = rows.get(3).mse();
                double central = rows.get(4).mse();
                assertTrue(Math.min(grr, Math.min(oue, olh)) > 3 * central, "local error above central");
                if (k == 256 && eps == 1) {
                    assertTrue(Math.min(oue, olh) < grr, "OUE/OLH win on large domains");
                }
                if (k == 16 && eps == 4) {
                    assertTrue(grr < oue, "GRR wins on small domains");
                }
            }
        }
    }

    @Test
    void permuteAndFlipIsNeverWorseThanTheExponentialMechanism() {
        for (double eps : new double[] {0.1, 0.5}) {
            List<MechanismShootout.Selection> mode = MechanismShootout.selection(200, 32, eps, 1);
            System.out.printf("Selection mode of 32 items eps %.2f gaps:%s%n", eps, gaps(mode));
            assertTrue(mode.get(3).gap() <= mode.get(0).gap() * 1.05 + 0.5, "PF <= EM");
            assertTrue(mode.get(0).gap() < mode.get(4).gap(), "EM beats uniform");
        }
        List<MechanismShootout.Selection> hp = MechanismShootout.hyperparameters(1, 1);
        System.out.printf("Hyperparameter selection of 16 configs eps 1 test-accuracy gaps:%s%n", gaps(hp));
        assertTrue(hp.get(3).gap() < hp.get(4).gap(), "PF beats uniform");
    }

    private static String gaps(List<MechanismShootout.Selection> rows) {
        StringBuilder b = new StringBuilder();
        for (MechanismShootout.Selection s : rows) {
            b.append(String.format(" %s %.3f", s.mechanism(), s.gap()));
        }
        return b.toString();
    }

    @Test
    void brokenSparseVectorVariantsLeakMoreThanClaimed() {
        double[] e = MechanismShootout.sparseVector(1, 5, 40000, 1);
        System.out.printf("Sparse vector claimed eps 1, measured: correct %.2f, no query noise %.2f, unbounded %.2f%n",
                e[0], e[1], e[2]);
        assertTrue(e[0] <= 1.3, "correct variant within the claim (sampling slack) " + e[0]);
        assertTrue(e[1] > 2 && e[2] > 1.3, "broken variants exceed the claim");
    }

    @Test
    void synthesisersAgainstIndependentMarginals() {
        for (double eps : new double[] {0.1, 1, 10}) {
            List<MechanismShootout.Release> rows = MechanismShootout.release(eps, 2000, 1);
            StringBuilder b = new StringBuilder();
            for (MechanismShootout.Release r : rows) {
                b.append(String.format(" %s err %.4f TSTR %.3f;", r.mechanism(), r.workloadError(), r.tstr()));
            }
            System.out.printf("Query release eps %.1f:%s%n", eps, b);
            assertTrue(rows.get(3).tstr() > 0.6, "real data TSTR");
            if (eps == 10) {
                assertTrue(rows.get(1).tstr() > 0.55 || rows.get(2).tstr() > 0.55, "a synthesiser keeps utility");
            }
        }
    }

    @Test
    void correlatedNoiseBeatsIndependentNoiseOnPrefixSums() {
        List<MechanismShootout.Prefix> rows = MechanismShootout.prefixSums(512, 4, 128);
        StringBuilder b = new StringBuilder();
        for (MechanismShootout.Prefix p : rows) {
            b.append(String.format(" %s rmse %.2f state %d;", p.mechanism(), p.rmse(), p.state()));
        }
        System.out.printf("Prefix sums n=512 k=4 sep=128:%s%n", b);
        double indep = rows.get(0).rmse();
        assertTrue(rows.get(1).rmse() < indep && rows.get(2).rmse() < indep && rows.get(3).rmse() < indep,
                "correlated noise beats independent");
        assertTrue(rows.get(3).state() < rows.get(2).state(), "BLT keeps less state than banded MF");
    }
}
