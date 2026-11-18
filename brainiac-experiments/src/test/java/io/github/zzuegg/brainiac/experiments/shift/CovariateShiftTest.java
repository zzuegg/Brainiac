package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class CovariateShiftTest {

    @Test
    void importanceWeightingAndWeightedConformal() {
        for (int dim : new int[] {1, 5}) {
            Map<DomainGeneralization.Weighting, DomainGeneralization.CovariateScores> r =
                    DomainGeneralization.covariate(dim, 200, 1);
            StringBuilder sb = new StringBuilder();
            r.forEach((k, v) -> sb.append(String.format("  %s %.3f/%.2f/%.0f", k, v.targetMse(), v.ratioError(),
                    v.effectiveSampleSize())));
            System.out.printf("Covariate shift dim %d (target MSE/ratio error/ESS)%s%n", dim, sb);
            if (dim == 1) {
                assertTrue(r.get(DomainGeneralization.Weighting.ORACLE).targetMse()
                        < r.get(DomainGeneralization.Weighting.NONE).targetMse(), "oracle weights help in 1-D");
            }
        }
        DomainGeneralization.ConformalScores c = DomainGeneralization.conformal(0.1, 1);
        System.out.printf("Weighted conformal coverage (target 0.90): plain %.3f weighted %.3f oracle %.3f; "
                + "set size %.2f vs %.2f%n", c.plain(), c.weighted(), c.oracle(), c.plainSize(), c.weightedSize());
        assertTrue(Math.abs(c.weighted() - 0.9) <= Math.abs(c.plain() - 0.9) + 0.03,
                "weighted coverage is about as close to nominal as plain or closer");
    }
}
