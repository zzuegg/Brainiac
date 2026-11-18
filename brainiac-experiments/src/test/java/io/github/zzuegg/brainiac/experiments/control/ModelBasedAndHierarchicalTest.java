package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.explore.Hiro;
import io.github.zzuegg.brainiac.experimental.explore.PointMaze;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelBasedAndHierarchicalTest {

    @Test
    void hierarchy() {
        Map<String, Double> r = ModelBasedAndHierarchical.hierarchy(PointMaze.Layout.OPEN, 60, 2_000, 1, 32,
                new Hiro.Mode[] {Hiro.Mode.HAC, Hiro.Mode.FLAT_HER}, false);
        System.out.printf("11.6.10 open point maze success rate at 2k steps: %s%n", r);
        assertTrue(r.values().stream().allMatch(v -> v >= 0 && v <= 1));
    }
}
