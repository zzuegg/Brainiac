package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.8: GFlowNets with trajectory balance. */
class GFlowNetSamplingTest {

    @Test
    void gflownetSamplesInProportionToTheReward() {
        GFlowNetSampling.Hypergrid grid = new GFlowNetSampling.Hypergrid(8);
        GFlowNetSampling.Result gfn = GFlowNetSampling.gflownet(grid, 400, 1);
        GFlowNetSampling.Result ppo = GFlowNetSampling.ppo(grid, 30_000, 1);
        GFlowNetSampling.Result mcmc = GFlowNetSampling.mcmc(grid, gfn.rewardCalls(), 1);
        GFlowNetSampling.Result me = GFlowNetSampling.mapElites(grid, (int) gfn.rewardCalls(), 1);
        List<GFlowNetSampling.Result> seq = GFlowNetSampling.sequences(new GFlowNetSampling.Sequences(3), 300, 1);
        System.out.printf("hypergrid 8x8 L1 to R/Z / modes of 4 / reward calls: GFlowNet %.2f/%d/%d, PPO %.2f/%d/%d,"
                + " MCMC %.2f/%d/%d, MAP-Elites -/%d/%d; sequences L1 / modes of 4: GFlowNet %.2f/%d, MCMC %.2f/%d%n",
                gfn.l1(), gfn.modes(), gfn.rewardCalls(), ppo.l1(), ppo.modes(), ppo.rewardCalls(), mcmc.l1(),
                mcmc.modes(), mcmc.rewardCalls(), me.modes(), me.rewardCalls(), seq.get(0).l1(), seq.get(0).modes(),
                seq.get(1).l1(), seq.get(1).modes());
        assertEquals(4, gfn.modes(), "GFlowNet finds every mode");
        assertTrue(gfn.l1() < ppo.l1(), "GFlowNet matches R/Z better than reward-maximizing PPO");
        assertTrue(gfn.l1() < 0.5, "GFlowNet L1 " + gfn.l1());
    }
}
