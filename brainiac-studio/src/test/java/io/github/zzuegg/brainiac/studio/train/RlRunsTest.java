package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Agents learn the built-in environments in runs of their own. */
class RlRunsTest {

    private static Map<String, Object> await(RlRuns runs, String id) throws InterruptedException {
        for (int i = 0; i < 1200; i++) {
            Map<String, Object> r = runs.get(id).orElseThrow();
            if (!"running".equals(r.get("state"))) {
                return r;
            }
            Thread.sleep(50);
        }
        throw new AssertionError("the run did not end: " + runs.get(id).orElseThrow());
    }

    @Test
    void aTableLearnsTheCliffWalkAndCanBeWatched() throws Exception {
        try (RlRuns runs = new RlRuns()) {
            Map<String, Object> all = runs.toJson();
            assertEquals(6, Json.array(all.get("environments")).size());
            assertEquals(6, Json.array(all.get("algorithms")).size());
            Json.write(all);

            String id = (String) runs.start("cliff", "qlearning", 20_000, 1, null).get("id");
            Map<String, Object> run = await(runs, id);
            assertEquals("done", run.get("state"), String.valueOf(run.get("error")));
            assertTrue(((Number) run.get("episodes")).intValue() > 10);
            // the shortest safe walk along the cliff gives -13
            assertTrue(((Number) run.get("score")).doubleValue() > -30, run.get("score").toString());
            assertFalse(Json.array(run.get("curve")).isEmpty());
            assertTrue(Json.array(run.get("curve")).size() <= RlRuns.CURVE_POINTS);
            assertTrue(Json.array(run.get("frames")).size() > 5);
            Json.write(run);
            assertEquals(1, Json.array(runs.toJson().get("runs")).size());
            assertTrue(runs.delete(id));
            assertFalse(runs.delete(id));
        }
    }

    @Test
    void aNetworkLearnsAndARunCanBeStopped() throws Exception {
        try (RlRuns runs = new RlRuns()) {
            String id = (String) runs.start("cartpole", "ppo", 2_000, 3, List.of(16)).get("id");
            Map<String, Object> run = await(runs, id);
            assertEquals("done", run.get("state"), String.valueOf(run.get("error")));
            assertEquals(List.of(16), run.get("hidden"));
            assertTrue(((Number) run.get("score")).doubleValue() > 0);

            String grid = (String) runs.start("lake", "dqn", 1_500, 1, List.of(16)).get("id");
            assertEquals("done", await(runs, grid).get("state"));

            String longRun = (String) runs.start("cartpole", "a2c", RlRuns.MAX_STEPS, 1, List.of(16)).get("id");
            assertThrows(IllegalArgumentException.class, () -> runs.delete(longRun));
            assertTrue(runs.stop(longRun));
            assertEquals("stopped", await(runs, longRun).get("state"));
        }
    }

    @Test
    void whatDoesNotFitIsRefused() {
        try (RlRuns runs = new RlRuns()) {
            assertTrue(assertThrows(IllegalArgumentException.class, () -> runs.start("cartpole", "qlearning", 1000, 1, null))
                    .getMessage().contains("grid"));
            assertTrue(assertThrows(IllegalArgumentException.class, () -> runs.start("pendulum", "dqn", 1000, 1, null))
                    .getMessage().contains("continuous"));
            assertThrows(IllegalArgumentException.class, () -> runs.start("chess", "ppo", 1000, 1, null));
            assertThrows(IllegalArgumentException.class, () -> runs.start("cartpole", "ppo", 5, 1, null));
            assertThrows(IllegalArgumentException.class, () -> runs.start("cartpole", "ppo", 1000, 1, List.of(2)));
            assertThrows(IllegalArgumentException.class, () -> runs.start("cartpole", "ppo", 1000, 1, List.of(8, 8, 8)));
            assertFalse(runs.stop("9"));
        }
    }
}
