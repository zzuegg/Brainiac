package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.SennRow;

import java.util.List;

import org.junit.jupiter.api.Test;

class SelfExplainingNetworkTest {

    @Test
    void sennRecoversVaryingCoefficientsStably() {
        List<SennRow> rows = InterpretableNetworks.senn(1200, 1);
        for (SennRow r : rows) {
            System.out.printf("SENN %-22s test MSE %.4f coefficient correlation %.3f local Lipschitz %.3f%n", r.model(),
                    r.mse(), r.coefficientCorrelation(), r.lipschitz());
        }
        SennRow senn = rows.get(0).coefficientCorrelation() > rows.get(1).coefficientCorrelation() ? rows.get(0)
                : rows.get(1);
        SennRow nam = rows.get(5);
        assertTrue(senn.coefficientCorrelation() > 0.8, "SENN relevances track the true local coefficients");
        assertTrue(senn.coefficientCorrelation() > nam.coefficientCorrelation(), "an additive model cannot");
    }
}
