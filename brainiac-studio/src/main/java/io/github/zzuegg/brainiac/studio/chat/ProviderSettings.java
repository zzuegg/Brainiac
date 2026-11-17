package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * What the person chose per provider, kept in a plain properties file next to the keys: whether
 * Auto may use the provider, which of its models to show, another address, and options a
 * provider understands (such as the organization Hugging Face bills). Nothing in it is secret.
 */
public final class ProviderSettings {

    /** Option: the Hugging Face organization that pays, sent as {@code X-HF-Bill-To}. */
    public static final String BILL_TO = "billTo";

    private static final Pattern ID = Pattern.compile("[a-z][a-z0-9-]{0,40}");
    private static final Pattern MODEL = Pattern.compile("[^,\\s]{1,200}");
    private static final Pattern ORG = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,95}");

    private final Path file;
    private final Properties p = new Properties();

    /**
     * Opens the settings.
     *
     * @param file the file; it need not exist yet
     */
    public ProviderSettings(Path file) {
        this.file = file;
        if (Files.isRegularFile(file)) {
            try (InputStream in = Files.newInputStream(file)) {
                p.load(in);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + file, e);
            }
        }
    }

    /**
     * Says whether Auto may pick this provider's models.
     *
     * @param provider the provider id
     * @return true unless turned off
     */
    public synchronized boolean auto(String provider) {
        return !"false".equals(p.getProperty(provider + ".auto"));
    }

    /**
     * Returns the models picked to show, or nothing when all are shown.
     *
     * @param provider the provider id
     * @return the model ids, in the provider's own form
     */
    public synchronized Optional<Set<String>> models(String provider) {
        String v = p.getProperty(provider + ".models");
        if (v == null) {
            return Optional.empty();
        }
        Set<String> out = new LinkedHashSet<>();
        Arrays.stream(v.split(",")).map(String::strip).filter(s -> !s.isEmpty()).forEach(out::add);
        return Optional.of(out);
    }

    /**
     * Returns the address chosen for a provider, if any.
     *
     * @param provider the provider id
     * @return the address
     */
    public synchronized Optional<URI> url(String provider) {
        return Optional.ofNullable(p.getProperty(provider + ".url")).map(URI::create);
    }

    /**
     * Returns an option, or an empty string.
     *
     * @param provider the provider id
     * @param name the option, such as {@link #BILL_TO}
     * @return its value
     */
    public synchronized String option(String provider, String name) {
        return p.getProperty(provider + ".option." + name, "");
    }

    /**
     * Turns Auto on or off for a provider.
     *
     * @param provider the provider id
     * @param on false to keep Auto away from it
     * @throws IOException if the file cannot be written
     */
    public synchronized void auto(String provider, boolean on) throws IOException {
        set(provider, "auto", on ? null : "false");
    }

    /**
     * Chooses which models to show.
     *
     * @param provider the provider id
     * @param models the model ids, or null to show all
     * @throws IOException if the file cannot be written
     */
    public synchronized void models(String provider, List<String> models) throws IOException {
        if (models != null) {
            for (String m : models) {
                if (!MODEL.matcher(m).matches()) {
                    throw new IllegalArgumentException("not a model id: " + m);
                }
            }
        }
        set(provider, "models", models == null ? null : String.join(",", new LinkedHashSet<>(models)));
    }

    /**
     * Sends a provider to another address, such as a gateway or another region.
     *
     * @param provider the provider id
     * @param url an http or https address, or null or blank for the provider's own
     * @throws IOException if the file cannot be written
     */
    public synchronized void url(String provider, String url) throws IOException {
        String v = url == null || url.isBlank() ? null : url.strip();
        if (v != null) {
            URI u;
            try {
                u = URI.create(v.endsWith("/") ? v : v + "/");
            } catch (IllegalArgumentException bad) {
                throw new IllegalArgumentException("not an address: " + v);
            }
            if (!("https".equals(u.getScheme()) || "http".equals(u.getScheme())) || u.getHost() == null
                    || u.getUserInfo() != null || u.getQuery() != null || u.getFragment() != null) {
                throw new IllegalArgumentException("the address must be http or https, with a host and no query: " + v);
            }
            v = u.toString();
        }
        set(provider, "url", v);
    }

    /**
     * Sets an option.
     *
     * @param provider the provider id
     * @param name the option, such as {@link #BILL_TO}
     * @param value its value, or null or blank to clear it
     * @throws IOException if the file cannot be written
     */
    public synchronized void option(String provider, String name, String value) throws IOException {
        String v = value == null || value.isBlank() ? null : value.strip();
        if (BILL_TO.equals(name)) {
            if (v != null && !ORG.matcher(v).matches()) {
                throw new IllegalArgumentException("not an organization name: " + v);
            }
        } else {
            throw new IllegalArgumentException("no option " + name);
        }
        set(provider, "option." + name, v);
    }

    private void set(String provider, String name, String value) throws IOException {
        if (!ID.matcher(provider).matches()) {
            throw new IllegalArgumentException("not a provider id: " + provider);
        }
        if (value == null) {
            p.remove(provider + "." + name);
        } else {
            p.setProperty(provider + "." + name, value);
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try (OutputStream out = Files.newOutputStream(tmp)) {
            p.store(out, "Brainiac Studio provider settings (no secrets; keys are in keys.properties)");
        }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
}
