package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.eval.KernelDivergenceScore;

import java.util.List;

import org.junit.jupiter.api.Test;

class ContaminationAuditTest {

    @Test
    void setLevelScoresAgainstTheContaminationRate() {
        List<ContaminationAudit.SetLevel> rows = ContaminationAudit.setLevel(new double[] {0, 0.25, 0.5, 0.75, 1},
                600, 1);
        rows.forEach(r -> System.out.printf("contamination rate %.2f: KDS %.4f, Min-K%%++ %.4f, Min-K%% %.4f, "
                + "-loss %.4f%n", r.rate(), r.kds(), r.minKPlusPlus(), r.minK(), r.negativeLoss()));
        double[] rho = ContaminationAudit.spearman(rows);
        System.out.printf("set-level contamination, Spearman with the rate: KDS %.2f, Min-K%%++ %.2f, Min-K%% %.2f, "
                + "-loss %.2f%n", rho[0], rho[1], rho[2], rho[3]);
        assertTrue(rho[3] > 0.5, "planting lowers the benchmark loss");
        assertTrue(Math.abs(rho[0]) > 0.8, "KDS is monotone in the rate (on this toy with the sign reversed: "
                + "fine-tuning moves the memorised samples more)");
    }

    @Test
    void paraphrasesInflateScoresAndEvadeNgramDecontamination() {
        ContaminationAudit.Paraphrase p = ContaminationAudit.paraphrase(600, 2);
        System.out.println("paraphrased contamination: " + p);
        assertEquals(1.0, p.ngramVerbatim(), 1e-12);
        assertTrue(p.ngramParaphrase() < 0.6, "n-gram overlap misses most paraphrases");
        assertTrue(p.paraphraseLoss() < p.cleanLoss(), "paraphrases still inflate the score");
    }

    @Test
    void identicalEmbeddingsHaveZeroDivergence() {
        double[][] z = {{0, 0}, {1, 0}, {0, 2}};
        assertEquals(0, KernelDivergenceScore.score(z, z), 1e-12);
        double[][] moved = {{0, 0}, {3, 0}, {0, 2}};
        assertTrue(KernelDivergenceScore.score(z, moved) < 0);
    }
}
