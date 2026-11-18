package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;

import org.junit.jupiter.api.Test;

import java.util.Map;

class IntermittentDemandTest {

    @Test
    void crostonVariantsOnIntermittentSeries() {
        Map<String, double[]> st = IntermittentDemand.compare(IntermittentDemand.Scenario.STATIONARY,
                SeasonalSeries.Demand.POISSON_GAMMA, 0.8, 15, 1);
        System.out.printf("11.3.16 stationary 80%% zeros {sMAE, RMSSE, cum. bias}: %s%n", StateSpaceEmTest.line(st));
        Map<String, double[]> ln = IntermittentDemand.compare(IntermittentDemand.Scenario.STATIONARY,
                SeasonalSeries.Demand.BERNOULLI_LOGNORMAL, 0.9, 15, 2);
        System.out.printf("11.3.16 log-normal 90%% zeros {sMAE, RMSSE, cum. bias}: %s%n", StateSpaceEmTest.line(ln));
        Map<String, double[]> drop = IntermittentDemand.compare(IntermittentDemand.Scenario.RATE_DROP,
                SeasonalSeries.Demand.POISSON_GAMMA, 0.6, 15, 3);
        System.out.printf("11.3.16 rate drop {sMAE, RMSSE, cum. bias}: %s%n", StateSpaceEmTest.line(drop));
        Map<String, double[]> obs = IntermittentDemand.compare(IntermittentDemand.Scenario.OBSOLESCENCE,
                SeasonalSeries.Demand.POISSON_GAMMA, 0.7, 15, 4);
        System.out.printf("11.3.16 obsolescence {sMAE, RMSSE, cum. bias}: %s%n", StateSpaceEmTest.line(obs));
        assertTrue(st.get("SBA")[2] < st.get("CROSTON")[2], StateSpaceEmTest.line(st));
        assertTrue(obs.get("TSB")[2] < obs.get("CROSTON")[2] && obs.get("TSB")[2] < obs.get("SBA")[2],
                StateSpaceEmTest.line(obs));
        assertTrue(st.get("SBA")[1] < st.get("zero forecast")[1] + 0.5, StateSpaceEmTest.line(st));
    }
}
