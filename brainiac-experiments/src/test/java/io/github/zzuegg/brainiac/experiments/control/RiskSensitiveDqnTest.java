package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RiskSensitiveDqnTest {

    @Test
    void cvarActingAvoidsTheWindyBridge() {
        Map<String, RiskSensitiveDqn.Behaviour> r = RiskSensitiveDqn.risk(2000, 0.25f, 3);
        RiskSensitiveDqn.Behaviour neutral = r.get("risk-neutral");
        RiskSensitiveDqn.Behaviour cvar = r.get("CVaR");
        System.out.printf("W6 3.5 risky bridge: CVaR(0.25) falls %.3f, bridge %.2f, return %.3f vs risk-neutral falls"
                + " %.3f, bridge %.2f, return %.3f%n", cvar.fallRate(), cvar.bridgeRate(), cvar.meanReturn(),
                neutral.fallRate(), neutral.bridgeRate(), neutral.meanReturn());
        assertTrue(cvar.fallRate() < neutral.fallRate(), r.toString());
        assertTrue(cvar.bridgeRate() < 0.5 && neutral.bridgeRate() > 0.5, r.toString());
    }
}
