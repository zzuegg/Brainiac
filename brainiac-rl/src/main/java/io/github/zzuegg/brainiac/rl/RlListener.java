package io.github.zzuegg.brainiac.rl;

/** Hears about training progress. */
@FunctionalInterface
public interface RlListener {

    /** Ignores everything. */
    RlListener NONE = (e, r) -> { };

    /**
     * Called after every finished episode.
     *
     * @param episode the episode
     * @param result every episode so far
     */
    void onEpisode(Episode episode, RlResult result);

    /**
     * Prints the mean return of the last {@code n} episodes every {@code n} episodes.
     *
     * @param n how often to print
     * @return the listener
     */
    static RlListener every(int n) {
        return (e, r) -> {
            if (e.number() % n == 0) {
                System.out.printf("episode %5d  steps %8d  mean return (last %d) %9.2f%n", e.number(),
                        e.totalSteps(), n, r.meanReturn(n));
            }
        };
    }
}
