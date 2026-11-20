package io.github.zzuegg.brainiac.studio.agent.js;

/** A script that cannot be read, or that went past a limit. Scripts cannot catch it. */
public final class ScriptError extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Creates the error.
     *
     * @param message what went wrong, with the line
     */
    public ScriptError(String message) {
        super(message, null, false, false);
    }
}
