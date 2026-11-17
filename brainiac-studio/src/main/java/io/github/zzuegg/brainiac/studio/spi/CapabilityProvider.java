package io.github.zzuegg.brainiac.studio.spi;

import java.util.List;

/**
 * Adds capabilities to the studio. Implementations are found with {@link java.util.ServiceLoader}:
 * declare {@code provides io.github.zzuegg.brainiac.studio.spi.CapabilityProvider with ...} in a
 * module, or list the class in {@code META-INF/services} on the class path.
 *
 * <p>When two providers describe the same key, an available one wins over a planned one, so a
 * module that implements a planned capability replaces the placeholder.
 */
public interface CapabilityProvider {

    /**
     * Returns the capabilities this provider offers. Called once at startup.
     *
     * @return the capabilities
     */
    List<Capability> capabilities();
}
