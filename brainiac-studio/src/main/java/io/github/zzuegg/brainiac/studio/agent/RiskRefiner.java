package io.github.zzuegg.brainiac.studio.agent;

import java.util.Map;

/**
 * A tool source that decides a call's tier from its arguments, such as writing a new file
 * ({@link Risk#CREATE}) against overwriting one ({@link Risk#WRITE}).
 */
@FunctionalInterface
public interface RiskRefiner {

    /**
     * Says which paths a call reads or writes, so the rules can judge paths outside the project.
     *
     * @param tool the tool's name
     * @param arguments the call's arguments
     * @return the paths; none by default
     */
    default java.util.List<Reach.Access> reach(String tool, Map<String, Object> arguments) {
        return java.util.List.of();
    }

    /**
     * Decides a call's tier.
     *
     * @param tool the tool's name
     * @param arguments the call's arguments
     * @param declared the tier the tool declares
     * @return the call's tier
     */
    Risk refine(String tool, Map<String, Object> arguments, Risk declared);
}
