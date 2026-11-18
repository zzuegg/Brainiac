package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class JudgeAuditTest {

    private static JudgeAudit.Correction row(List<JudgeAudit.Correction> rows, String name) {
        return rows.stream().filter(r -> r.estimator().equals(name)).findFirst().orElseThrow();
    }

    @Test
    void plantedBiasesAndTheirMitigation() {
        JudgeAudit.ToyJudge judge = JudgeAudit.ToyJudge.train(400, 1);
        JudgeAudit.Audit a = JudgeAudit.audit(judge, 150, 2);
        System.out.println("judge audit: " + a);
        assertTrue(a.positionBias() > 0.03, "planted position bias is learnt: " + a.positionBias());
        assertTrue(a.verbosityEffect() > 0.03, "planted verbosity bias is learnt: " + a.verbosityEffect());
        assertTrue(a.spearmanLengthControlled() >= a.spearmanRaw(), a.toString());

        JudgeAudit.Labelled test = JudgeAudit.passVerdicts(judge, JudgeAudit.SYSTEMS, 10_000, 3);
        JudgeAudit.Labelled calibration = JudgeAudit.passVerdicts(judge, JudgeAudit.SYSTEMS, 4_000, 4);
        List<JudgeAudit.Correction> rows = JudgeAudit.correction(test, calibration, 200, 300, 5);
        rows.forEach(r -> System.out.println("trained judge, 200 gold labels: " + r));
        assertTrue(row(rows, "PPI++").coverage() > 0.85, rows.toString());
    }

    @Test
    void correctionsOfAnImperfectJudge() {
        JudgeAudit.Labelled test = JudgeAudit.parametric(0.6, 0.9, 0.7, 10_000, 1);
        JudgeAudit.Labelled same = JudgeAudit.parametric(0.6, 0.9, 0.7, 5_000, 2);
        JudgeAudit.Labelled shifted = JudgeAudit.parametric(0.3, 0.9, 0.7, 5_000, 3);
        List<JudgeAudit.Correction> rows = JudgeAudit.correction(test, same, 200, 500, 4);
        List<JudgeAudit.Correction> shift = JudgeAudit.correction(test, shifted, 200, 500, 5);
        rows.forEach(r -> System.out.println("judge correction, 200 labels: " + r));
        shift.forEach(r -> System.out.println("judge correction, 200 labels, calibration pass rate 0.3: " + r));
        JudgeAudit.Correction ppi = row(rows, "PPI++");
        JudgeAudit.Correction human = row(rows, "human only");
        System.out.printf("judge correction: raw judge coverage %.2f; PPI++ coverage %.2f, width %.3f vs human-only "
                + "%.3f (%.0f%% narrower); under label-rate shift Rogan-Gladen coverage %.2f vs PPI %.2f%n",
                row(rows, "judge only").coverage(), ppi.coverage(), ppi.width(), human.width(),
                100 * (1 - ppi.width() / human.width()), row(shift, "Rogan-Gladen").coverage(),
                row(shift, "PPI").coverage());
        assertTrue(row(rows, "judge only").coverage() < 0.1);
        assertTrue(ppi.coverage() > 0.9);
        assertTrue(ppi.width() < human.width());
        assertTrue(row(shift, "Rogan-Gladen").coverage() > 0.85);
        assertTrue(row(shift, "PPI").coverage() < row(shift, "Rogan-Gladen").coverage());
    }
}
