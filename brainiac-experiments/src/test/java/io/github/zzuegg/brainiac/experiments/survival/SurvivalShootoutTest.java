package io.github.zzuegg.brainiac.experiments.survival;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SurvivalData;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SurvivalShootoutTest {

    static String line(String title, Map<String, SurvivalShootout.Score> r) {
        StringBuilder b = new StringBuilder(title);
        r.forEach((k, v) -> b.append(String.format(" %s %.3f/%.3f |", k, v.cIndex(), v.brier())));
        return b.toString();
    }

    @Test
    void coxBeatsCensoringNaiveRegressionAtHeavyCensoring() {
        Map<String, SurvivalShootout.Score> r = SurvivalShootout.proportional(SurvivalData.Risk.LINEAR, 0.6, 1000,
                300, 1);
        String line = line("linear risk, 60% censored, C-index/IBS:", r);
        System.out.println(line);
        assertTrue(r.get("Cox").cIndex() > 0.65, line);
        assertTrue(r.get("Cox").cIndex() > r.get("MLP log-time (censored dropped)").cIndex(), line);
        assertTrue(r.get("Cox").brier() < r.get("Kaplan-Meier").brier(), line);
    }

    @Test
    void deepSurvBeatsCoxOnTheNonlinearRisk() {
        Map<String, SurvivalShootout.Score> r = SurvivalShootout.proportional(SurvivalData.Risk.NONLINEAR, 0.3, 1000,
                500, 1);
        String line = line("nonlinear risk, 30% censored, C-index/IBS:", r);
        System.out.println(line);
        assertTrue(r.get("DeepSurv").cIndex() > r.get("Cox").cIndex(), line);
    }
}
