package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.ContinualPretrainer;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ContinualPretrainingTest {

    @Test
    void rewarmingReplayAndTheStabilityGap() {
        Map<ContinualPretraining.Method, ContinualPretraining.Run> runs = new EnumMap<>(ContinualPretraining.Method.class);
        for (ContinualPretraining.Method m : ContinualPretraining.Method.values()) {
            runs.put(m, ContinualPretraining.run(m, 600, 300, 1));
            System.out.println("continual pretraining " + runs.get(m));
        }
        ContinualPretraining.Run replay = runs.get(ContinualPretraining.Method.REWARM_REPLAY);
        ContinualPretraining.Run rewarm = runs.get(ContinualPretraining.Method.REWARM);
        ContinualPretraining.Run retrain = runs.get(ContinualPretraining.Method.RETRAIN);
        System.out.printf("continual pretraining: 5%% replay + re-warm A %.4f / B %.4f vs retrain on A+B %.4f / %.4f "
                + "(mean gap %.4f nats); no replay A %.4f%n", replay.lossA(), replay.lossB(), retrain.lossA(),
                retrain.lossB(), 0.5 * (replay.lossA() + replay.lossB() - retrain.lossA() - retrain.lossB()),
                rewarm.lossA());
        assertTrue(replay.lossA() < rewarm.lossA(), "replay limits forgetting of A");
        assertTrue(replay.gapDepth() < rewarm.gapDepth(), "replay makes the rise of the A loss shallower");
        assertTrue(runs.get(ContinualPretraining.Method.CONSTANT).gapDepth() < rewarm.gapDepth(),
                "re-warming to a high rate forgets more");
        assertTrue(Double.isFinite(replay.lossB()) && replay.lossB() < 3.0);
    }

    @Test
    void gapTrackerMeasuresDepthAndLength() {
        ContinualPretrainer.GapTracker g = new ContinualPretrainer.GapTracker();
        double[] loss = {1.0, 1.3, 1.2, 1.05, 0.95, 0.9};
        for (int i = 0; i < loss.length; i++) {
            g.record(10L * i, loss[i]);
        }
        assertEquals(0.3, g.depth(), 1e-9);
        assertEquals(40, g.length());
        assertEquals(1, ContinualPretrainer.lowPerplexitySubset(new double[] {3, 1, 2}, 0.34)[0]);
    }
}
