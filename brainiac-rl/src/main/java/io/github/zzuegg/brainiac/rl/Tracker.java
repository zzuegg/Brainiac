package io.github.zzuegg.brainiac.rl;

/**
 * Counts steps and rewards and reports finished episodes: the bookkeeping the built-in
 * learners share, for learners written outside this module.
 */
public final class Tracker {

    final RlResult result = new RlResult();
    private final RlListener listener;
    private long steps;
    private int episodeSteps;
    private double episodeReward;

    /**
     * Creates a tracker.
     *
     * @param listener told about each finished episode; {@code null} for none
     */
    public Tracker(RlListener listener) {
        this.listener = listener == null ? RlListener.NONE : listener;
    }

    /**
     * Records one step.
     *
     * @param s the step
     * @return whether the episode ended
     */
    public boolean step(Environment.Step s) {
        steps++;
        episodeSteps++;
        episodeReward += s.reward();
        result.steps(steps);
        if (s.done()) {
            Episode e = new Episode(result.episodes().size() + 1, episodeSteps, episodeReward, steps);
            result.add(e);
            episodeSteps = 0;
            episodeReward = 0;
            listener.onEpisode(e, result);
            return true;
        }
        return false;
    }

    /** @return the steps recorded */
    public long steps() {
        return steps;
    }

    /** @return the episodes so far; the same object as training goes on */
    public RlResult result() {
        return result;
    }
}
