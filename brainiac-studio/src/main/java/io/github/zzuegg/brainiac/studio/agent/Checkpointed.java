package io.github.zzuegg.brainiac.studio.agent;

import java.io.IOException;
import java.nio.file.Path;

/** A tool source that changes files, and checkpoints them first so a run can be undone. */
public interface Checkpointed {

    /**
     * Gives the source where checkpoints are kept.
     *
     * @param checkpoints the run's checkpoints
     */
    void checkpoints(Checkpoints checkpoints);

    /**
     * Checkpoints a file for the run the call belongs to, if any.
     *
     * @param checkpoints the checkpoints, or null
     * @param file the file about to change
     * @throws IOException if the copy cannot be made
     */
    static void before(Checkpoints checkpoints, Path file) throws IOException {
        if (checkpoints != null && AgentRuns.RUN.isBound()) {
            checkpoints.before(AgentRuns.RUN.get(), file);
        }
    }
}
