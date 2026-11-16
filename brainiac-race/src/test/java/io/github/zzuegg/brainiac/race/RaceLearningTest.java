package io.github.zzuegg.brainiac.race;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.RlListener;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** PPO learns to drive with every car in the race sharing one policy. */
class RaceLearningTest {

    private static double distance(Agent agent, RaceConfig config) {
        Race race = new Race(config.copy().randomStart(false));
        SplittableRandom random = new SplittableRandom(3);
        float[][] obs = race.reset(random.split());
        for (int t = 0; t < config.raceSteps(); t++) {
            float[][] a = new float[race.count()][];
            // the race's last step cuts every episode off; the cars stay where they finished
            for (int i = 0; i < a.length; i++) {
                a[i] = agent.act(obs[i], false);
            }
            var steps = race.step(a);
            for (int i = 0; i < a.length; i++) {
                obs[i] = steps[i].terminated() ? race.reset(i, random.split()) : steps[i].observation();
            }
        }
        return race.view().cars().stream().mapToDouble(RaceView.CarView::distance).average().orElse(0);
    }

    @Test
    void selfPlayPpoLearnsToDrive() {
        // a quick check: fixed sensors and no memory, so there is little to learn
        RaceConfig config = new RaceConfig().cars(4).raceSteps(600).wallProbes(0).memory(1);
        Race race = new Race(config);
        int obs = config.observationSize();
        Ppo ppo = Ppo.of(Nets.mlp(obs, config.actionSize(), 1, 64, 64), Nets.mlp(obs, 1, 2, 64, 64), race.actionSpace())
                .rolloutSteps(256).minibatchSize(256).epochs(5).initialLogStd(-0.5f).seed(1);
        double before = distance(ppo, config);
        ppo.fit(race, 100_000, RlListener.NONE);
        double after = distance(ppo, config);
        assertTrue(after > before + 1000, "distance per car in 20 s: " + before + " -> " + after);
    }
}
