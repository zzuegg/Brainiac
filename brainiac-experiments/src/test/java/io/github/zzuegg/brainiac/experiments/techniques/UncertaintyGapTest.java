package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark.RegressionMethod;
import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark.RegressionScores;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class UncertaintyGapTest {

    @Test
    void weightPosteriorsWidenTheirErrorBarsInTheGap() {
        Map<RegressionMethod, RegressionScores> all = new EnumMap<>(RegressionMethod.class);
        for (RegressionMethod m : RegressionMethod.values()) {
            RegressionScores s = UncertaintyBenchmark.regression(m, 1);
            all.put(m, s);
            System.out.printf("Gap regression %-17s NLL %.3f gap NLL %.3f std gap %.3f data %.3f%n", m, s.nll(),
                    s.gapNll(), s.gapStd(), s.dataStd());
        }
        assertTrue(all.get(RegressionMethod.GP).gapStd() > 2 * all.get(RegressionMethod.GP).dataStd(),
                "the GP widens in the gap");
        for (RegressionMethod m : RegressionMethod.values()) {
            assertTrue(Double.isFinite(all.get(m).nll()), m + " scores");
        }
    }
}
