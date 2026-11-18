package io.github.zzuegg.brainiac.cloud;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

/**
 * Finds the Vast.ai API key without ever printing it.
 *
 * <p>Looked for in this order: the {@code VAST_API_KEY} environment variable, then the
 * studio's key file ({@code ~/.brainiac/studio/keys.properties}), where the studio saves a
 * provider's keys as {@code <provider>.order} (names, one per line) and
 * {@code <provider>.key.<name>}; the provider is {@code vastai} (or {@code vast}).
 */
public final class VastKey {

    /** The environment variable that holds the key. */
    public static final String ENV = "VAST_API_KEY";

    private static final List<String> PROVIDERS = List.of("vastai", "vast");

    private VastKey() {
    }

    /**
     * Finds the key in the environment or the studio's key file.
     *
     * @return the key, if there is one
     */
    public static Optional<String> find() {
        return find(System.getenv(), Path.of(System.getProperty("user.home"), ".brainiac", "studio", "keys.properties"));
    }

    /**
     * Finds the key in the given environment or key file.
     *
     * @param env the environment
     * @param keyFile the studio's key file; it need not exist
     * @return the key, if there is one
     */
    public static Optional<String> find(Map<String, String> env, Path keyFile) {
        String e = env.get(ENV);
        if (e != null && !e.isBlank()) {
            return Optional.of(e.strip());
        }
        if (!Files.isRegularFile(keyFile)) {
            return Optional.empty();
        }
        Properties p = new Properties();
        try (InputStream in = Files.newInputStream(keyFile)) {
            p.load(in);
        } catch (IOException x) {
            throw new UncheckedIOException("cannot read " + keyFile, x);
        }
        for (String provider : PROVIDERS) {
            for (String label : p.getProperty(provider + ".order", "").split("\n")) {
                String k = p.getProperty(provider + ".key." + label.strip());
                if (!label.isBlank() && k != null && !k.isBlank()) {
                    return Optional.of(k.strip());
                }
            }
            // a key saved without an order list
            for (String name : p.stringPropertyNames().stream().sorted().toList()) {
                if (name.startsWith(provider + ".key.") && !p.getProperty(name).isBlank()) {
                    return Optional.of(p.getProperty(name).strip());
                }
            }
        }
        return Optional.empty();
    }
}
