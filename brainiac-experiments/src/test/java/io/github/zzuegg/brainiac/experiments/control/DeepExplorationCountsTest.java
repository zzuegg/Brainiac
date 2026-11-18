package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class DeepExplorationCountsTest {

    @Test
    void countBonusesWidenCoverage() {
        Map<String, double[]> sea = DeepExploration.countBonuses(DeepExploration.CountTask.DEEP_SEA, 1_500, 1);
        Map<String, double[]> car = DeepExploration.countBonuses(DeepExploration.CountTask.MOUNTAIN_CAR, 1_500, 1);
        System.out.printf("11.6.7 {episodes to first success, cells visited} deep sea 12: %s| mountain car: %s%n",
                DeepExploration.fmt(sea), DeepExploration.fmt(car));
        assertTrue(sea.get("exact counts (oracle)")[1] >= sea.get("eps-greedy (no bonus)")[1], DeepExploration.fmt(sea));
        assertTrue(car.get("SimHash k=16")[1] > 0 && car.get("eps-greedy (no bonus)")[1] > 0, DeepExploration.fmt(car));
    }

    @Test
    void learnedHashOnPixels() {
        Map<String, double[]> px = DeepExploration.countBonuses(DeepExploration.CountTask.PIXEL_KEY_DOOR, 800, 1);
        System.out.printf("11.6.7 {episodes to first success, cells visited} pixel key-door: %s%n",
                DeepExploration.fmt(px));
        assertTrue(px.get("learned-code SimHash")[1] > 0, DeepExploration.fmt(px));
    }
}
