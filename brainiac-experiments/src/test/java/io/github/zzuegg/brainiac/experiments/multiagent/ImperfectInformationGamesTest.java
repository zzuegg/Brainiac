package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.rl.KuhnPoker;
import io.github.zzuegg.brainiac.experimental.rl.LeducPoker;
import io.github.zzuegg.brainiac.experiments.multiagent.ImperfectInformationGames.Row;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ImperfectInformationGamesTest {

    private static String format(List<Row> rows) {
        StringBuilder sb = new StringBuilder();
        for (Row r : rows) {
            sb.append(String.format("%s %.1f (work %d, %.2f s); ", r.method(), r.milliBlinds(), r.work(), r.seconds()));
        }
        return sb.toString();
    }

    @Test
    void cfrPlusConvergesFastestAndMccfrIsCheap() {
        double kuhnUniform = ImperfectInformationGames.uniform(new KuhnPoker());
        double leducUniform = ImperfectInformationGames.uniform(new LeducPoker());
        List<Row> kuhn = ImperfectInformationGames.regretMinimisation(new KuhnPoker(), 500, 10, 1);
        List<Row> leduc = ImperfectInformationGames.regretMinimisation(new LeducPoker(), 300, 10, 1);
        double mccfr = 0;
        for (long seed = 1; seed <= 3; seed++) {
            mccfr += ImperfectInformationGames.regretMinimisation(new LeducPoker(), 300, 10, seed).get(2).milliBlinds();
        }
        System.out.printf("Kuhn (uniform %.0f mbb/h): %s%n", kuhnUniform, format(kuhn));
        System.out.printf("Leduc (uniform %.0f mbb/h): %s MCCFR mean of 3 seeds %.1f%n", leducUniform, format(leduc),
                mccfr / 3);
        assertTrue(kuhn.get(1).milliBlinds() < kuhn.get(0).milliBlinds(), "CFR+ beats CFR on Kuhn");
        assertTrue(leduc.get(1).milliBlinds() < leduc.get(0).milliBlinds(), "CFR+ beats CFR on Leduc");
        assertTrue(kuhn.get(1).milliBlinds() < 5, "CFR+ nearly solves Kuhn");
        assertTrue(leduc.get(2).milliBlinds() < 0.2 * leducUniform, "MCCFR learns Leduc");
        assertTrue(leduc.get(2).work() < leduc.get(0).work(), "MCCFR touches fewer nodes");
        assertTrue(mccfr / 3 < 0.2 * leducUniform, "MCCFR learns Leduc on every seed");
    }

    @Test
    void nfspAveragePolicyIsLessExploitableThanUniform() {
        double uniform = ImperfectInformationGames.uniform(new KuhnPoker());
        double nfsp = 0;
        double dqn = 0;
        long steps = 0;
        for (long seed = 1; seed <= 3; seed++) {
            List<Row> rows = ImperfectInformationGames.fictitiousSelfPlay(new KuhnPoker(), 3000, seed);
            nfsp += rows.get(0).milliBlinds() / 3;
            dqn += rows.get(2).milliBlinds() / 3;
            steps = rows.get(0).work();
        }
        System.out.printf("Kuhn, 3000 episodes (~%d steps), 3 seeds: NFSP average policy %.0f mbb/h vs DQN self-play "
                + "%.0f vs uniform %.0f (CFR+ reference ~0)%n", steps, nfsp, dqn, uniform);
        assertTrue(nfsp < 0.6 * uniform, "NFSP learns a less exploitable policy");
    }

    @Test
    void doubleOracleAvoidsCycling() {
        List<Row> kuhn = ImperfectInformationGames.responseOracles(new KuhnPoker(), 8);
        List<Row> leduc = ImperfectInformationGames.responseOracles(new LeducPoker(), 50);
        List<Row> rps = ImperfectInformationGames.nonTransitive(4, 8);
        System.out.printf("PSRO Kuhn, 8 iterations: %s%n", format(kuhn));
        System.out.printf("PSRO Leduc, 50 iterations: %s%n", format(leduc));
        System.out.printf("RPS with skill (x1000): %s%n", format(rps));
        assertTrue(kuhn.get(0).milliBlinds() < kuhn.get(3).milliBlinds(), "double oracle beats naive self-play");
        assertTrue(leduc.get(0).milliBlinds() < leduc.get(3).milliBlinds(), "double oracle beats naive self-play");
        assertTrue(rps.get(0).milliBlinds() < 1 && rps.get(2).milliBlinds() > 100, "self-play cycles on RPS");
    }

    @Test
    void magnetMakesTheLastIterateConverge() {
        List<Row> kuhn = ImperfectInformationGames.lastIterate(new KuhnPoker(), 1000);
        List<Row> leduc = ImperfectInformationGames.lastIterate(new LeducPoker(), 1000);
        System.out.printf("last iterate, Kuhn 1000 it: %s%n", format(kuhn));
        System.out.printf("last iterate, Leduc 1000 it: %s%n", format(leduc));
        for (List<Row> rows : List.of(kuhn, leduc)) {
            assertTrue(rows.get(2).milliBlinds() < rows.get(0).milliBlinds(), "R-NaD beats multiplicative weights");
            assertTrue(rows.get(1).milliBlinds() < rows.get(0).milliBlinds(), "MMD beats multiplicative weights");
        }
    }
}
