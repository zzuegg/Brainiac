package io.github.zzuegg.brainiac.train;

import java.io.PrintStream;

/** Ready-made listeners that print training progress. */
public final class Progress {

    private Progress() {
    }

    /**
     * Prints every {@code n}-th epoch to standard output.
     *
     * @param n how often to print
     * @return the listener
     */
    public static TrainingListener every(int n) {
        return every(n, System.out);
    }

    /**
     * Prints every {@code n}-th epoch.
     *
     * @param n how often to print
     * @param out where to print
     * @return the listener
     */
    public static TrainingListener every(int n, PrintStream out) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be at least 1, got " + n);
        }
        return r -> {
            if (r.epoch() % n == 0 || r.epoch() == 1) {
                out.println(r);
            }
        };
    }
}
