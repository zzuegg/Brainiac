package io.github.zzuegg.brainiac.rl;

/** Counts steps and episodes during training and tells the listener about each episode. */
final class Tracker {

    final RlResult result = new RlResult();
    private final RlListener listener;
    private long steps;
    private int[] episodeSteps = new int[1];
    private double[] episodeReward = new double[1];

    Tracker(RlListener listener) {
        this.listener = listener == null ? RlListener.NONE : listener;
    }

    boolean step(Environment.Step s) {
        return step(0, s);
    }

    /**
     * Records a step of one of several environments run side by side; each keeps its own
     * episode length and return.
     */
    boolean step(int env, Environment.Step s) {
        if (env >= episodeSteps.length) {
            episodeSteps = java.util.Arrays.copyOf(episodeSteps, env + 1);
            episodeReward = java.util.Arrays.copyOf(episodeReward, env + 1);
        }
        steps++;
        episodeSteps[env]++;
        episodeReward[env] += s.reward();
        result.steps(steps);
        if (s.done()) {
            Episode e = new Episode(result.episodes().size() + 1, episodeSteps[env], episodeReward[env], steps);
            result.add(e);
            episodeSteps[env] = 0;
            episodeReward[env] = 0;
            listener.onEpisode(e, result);
            return true;
        }
        return false;
    }

    long steps() {
        return steps;
    }
}
