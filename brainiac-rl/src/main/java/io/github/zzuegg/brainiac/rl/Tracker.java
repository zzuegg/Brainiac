package io.github.zzuegg.brainiac.rl;

/** Counts steps and rewards and reports finished episodes. */
final class Tracker {

    final RlResult result = new RlResult();
    private final RlListener listener;
    private long steps;
    private int episodeSteps;
    private double episodeReward;

    Tracker(RlListener listener) {
        this.listener = listener == null ? RlListener.NONE : listener;
    }

    /** Records one step; returns whether the episode ended. */
    boolean step(Environment.Step s) {
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

    long steps() {
        return steps;
    }
}
