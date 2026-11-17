package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Counts steps and rewards of a training run and reports finished episodes: the public
 * twin of the core's package-private {@code Tracker}, so learners outside
 * {@code io.github.zzuegg.brainiac.rl} can return a proper {@link RlResult} from
 * {@link io.github.zzuegg.brainiac.rl.Learner#fit}.
 *
 * <p>{@link RlResult} can only be filled from inside the core package, so until it gets a
 * public way to add episodes, {@link #result()} rebuilds it by replaying the recorded
 * step counts and returns through a core {@link Tabular} learner on a one-state replay
 * environment. Once the core hook exists this class collapses to a thin wrapper.
 */
public final class EpisodeRecorder {

    private final RlListener listener;
    private final List<Episode> episodes = new ArrayList<>();
    private long steps;
    private int episodeSteps;
    private double episodeReward;

    /**
     * Creates a recorder.
     *
     * @param listener told about every finished episode; may be {@code null} or
     *     {@link RlListener#NONE}
     */
    public EpisodeRecorder(RlListener listener) {
        this.listener = listener == null ? RlListener.NONE : listener;
    }

    /**
     * Records one environment step.
     *
     * @param s the step
     * @return whether the episode ended
     */
    public boolean step(Environment.Step s) {
        return step(s.reward(), s.done());
    }

    /**
     * Records one environment step.
     *
     * @param reward the reward
     * @param done whether the episode ended
     * @return whether the episode ended
     */
    public boolean step(double reward, boolean done) {
        steps++;
        episodeSteps++;
        episodeReward += reward;
        if (done) {
            Episode e = new Episode(episodes.size() + 1, episodeSteps, episodeReward, steps);
            episodes.add(e);
            episodeSteps = 0;
            episodeReward = 0;
            if (listener != RlListener.NONE) {
                listener.onEpisode(e, result());
            }
            return true;
        }
        return false;
    }

    /** @return the environment steps recorded */
    public long steps() {
        return steps;
    }

    /** @return the finished episodes, oldest first */
    public List<Episode> episodes() {
        return List.copyOf(episodes);
    }

    /**
     * @param last how many of the latest episodes to average
     * @return their mean total reward, or {@code NaN} with none
     */
    public double meanReturn(int last) {
        int n = Math.min(last, episodes.size());
        if (n == 0) {
            return Double.NaN;
        }
        double sum = 0;
        for (int i = episodes.size() - n; i < episodes.size(); i++) {
            sum += episodes.get(i).totalReward();
        }
        return sum / n;
    }

    /** @return the run so far as a core {@link RlResult} */
    public RlResult result() {
        return Tabular.qLearning(1).fit(new Replay(episodes, steps), steps, RlListener.NONE);
    }

    /** Plays back recorded episodes: the whole return on each episode's first step. */
    private static final class Replay implements Environment {
        private final List<Episode> episodes;
        private int episode;
        private int stepInEpisode;

        Replay(List<Episode> recorded, long total) {
            this.episodes = new ArrayList<>(recorded);
            long done = recorded.isEmpty() ? 0 : recorded.getLast().totalSteps();
            if (total > done) {
                this.episodes.add(new Episode(0, Math.toIntExact(total - done + 1), 0, total));
            }
        }

        @Override
        public Space observationSpace() {
            return Space.discrete(1);
        }

        @Override
        public Space actionSpace() {
            return Space.discrete(1);
        }

        @Override
        public float[] reset(SplittableRandom random) {
            stepInEpisode = 0;
            return new float[] {0};
        }

        @Override
        public Step step(float[] action) {
            Episode e = episodes.get(episode);
            stepInEpisode++;
            float reward = stepInEpisode == 1 ? (float) e.totalReward() : 0f;
            boolean end = stepInEpisode >= e.steps();
            if (end) {
                episode++;
            }
            return new Step(new float[] {0}, reward, end, false);
        }
    }
}
