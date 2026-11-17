package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.ActiveInferenceAgent;
import io.github.zzuegg.brainiac.experimental.rl.CuedTMaze;
import io.github.zzuegg.brainiac.rl.Environment;
import java.util.SplittableRandom;

/**
 * Active inference in the cued T-maze. Planning by sophisticated inference, the agent first
 * visits the cue, then the arm it points to, because it anticipates the belief the cue will
 * give it. Classic active inference scores fixed two-step policies under its current belief;
 * the cue then looks no better than an arm (whose outcome is just as informative), and it
 * gambles.
 */
public final class EpistemicForaging {

    /**
     * @param rewardRate the share of rewarded episodes
     * @param cueFirst the share of episodes that started by visiting the cue
     */
    public record Score(double rewardRate, double cueFirst) {
    }

    /**
     * @param sophisticated sophisticated inference
     * @param sophisticatedPragmatic sophisticated inference without epistemic value
     * @param classic classic policies with epistemic value
     */
    public record Result(Score sophisticated, Score sophisticatedPragmatic, Score classic) {
    }

    private EpistemicForaging() {
    }

    /**
     * @param episodes episodes per agent
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int episodes, long seed) {
        CuedTMaze.GenerativeModel model = CuedTMaze.model(0.95, 3);
        return new Result(play(model.agent(2, seed), episodes, seed),
                play(model.agent(2, seed).epistemic(false), episodes, seed),
                play(model.agent(2, seed).sophisticated(false), episodes, seed));
    }

    private static Score play(ActiveInferenceAgent agent, int episodes, long seed) {
        CuedTMaze env = CuedTMaze.of(0.95);
        SplittableRandom random = new SplittableRandom(seed);
        int rewarded = 0;
        int cue = 0;
        for (int e = 0; e < episodes; e++) {
            agent.beginEpisode();
            float[] obs = env.reset(random.split());
            boolean done = false;
            boolean first = true;
            while (!done) {
                float[] a = agent.act(obs, true);
                if (first && a[0] == 3) {
                    cue++;
                }
                first = false;
                Environment.Step s = env.step(a);
                rewarded += s.reward() > 0 ? 1 : 0;
                obs = s.observation();
                done = s.done();
            }
        }
        return new Score((double) rewarded / episodes, (double) cue / episodes);
    }
}
