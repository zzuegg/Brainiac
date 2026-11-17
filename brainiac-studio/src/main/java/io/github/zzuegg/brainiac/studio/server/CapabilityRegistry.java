package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.studio.spi.Capability;
import io.github.zzuegg.brainiac.studio.spi.CapabilityProvider;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;

/** Every capability found at startup, with one entry per key. */
public final class CapabilityRegistry {

    private static final System.Logger LOG = System.getLogger(CapabilityRegistry.class.getName());

    private final List<Capability> capabilities;

    private CapabilityRegistry(List<Capability> capabilities) {
        this.capabilities = List.copyOf(capabilities);
    }

    /**
     * Collects the capabilities of every {@link CapabilityProvider} the service loader finds.
     * A provider that fails is logged and skipped, so one broken module cannot stop the studio.
     *
     * @return the registry
     */
    public static CapabilityRegistry discover() {
        List<CapabilityProvider> providers = new ArrayList<>();
        ServiceLoader.load(CapabilityProvider.class).stream().forEach(p -> {
            try {
                providers.add(p.get());
            } catch (ServiceConfigurationError e) {
                LOG.log(System.Logger.Level.WARNING, "skipping capability provider " + p.type().getName(), e);
            }
        });
        if (providers.stream().noneMatch(p -> p instanceof CoreCapabilities)) {
            providers.add(new CoreCapabilities());   // on the class path without a services file
        }
        return of(providers);
    }

    /**
     * Merges the capabilities of the given providers: for a key offered more than once, an
     * available capability replaces a planned one, otherwise the first wins.
     *
     * @param providers the providers
     * @return the registry
     */
    public static CapabilityRegistry of(List<? extends CapabilityProvider> providers) {
        Map<String, Capability> byKey = new LinkedHashMap<>();
        for (CapabilityProvider provider : providers) {
            List<Capability> offered;
            try {
                offered = provider.capabilities();
            } catch (RuntimeException e) {
                LOG.log(System.Logger.Level.WARNING, "capability provider " + provider.getClass().getName() + " failed", e);
                continue;
            }
            for (Capability c : offered) {
                Capability old = byKey.get(c.key());
                if (old == null || (old.status() == Capability.Status.PLANNED
                        && c.status() == Capability.Status.AVAILABLE)) {
                    byKey.put(c.key(), c);
                }
            }
        }
        List<Capability> sorted = new ArrayList<>(byKey.values());
        sorted.sort(Comparator.comparing((Capability c) -> c.status() == Capability.Status.PLANNED)
                .thenComparing(c -> c.medium().ordinal())
                .thenComparing(Capability::key));
        return new CapabilityRegistry(sorted);
    }

    /**
     * Returns the capabilities: available ones first, then by medium and key.
     *
     * @return the capabilities
     */
    public List<Capability> all() {
        return capabilities;
    }

    /**
     * Returns the capabilities as JSON-ready maps.
     *
     * @return one map per capability
     */
    public List<Map<String, Object>> toJson() {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Capability c : capabilities) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("key", c.key());
            m.put("medium", c.medium().id());
            m.put("title", c.title());
            m.put("description", c.description());
            m.put("source", c.source());
            m.put("status", c.status() == Capability.Status.AVAILABLE ? "available" : "planned");
            m.put("note", c.note());
            out.add(m);
        }
        return out;
    }
}
