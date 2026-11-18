package io.github.zzuegg.brainiac.cloud;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.regex.Pattern;

/**
 * Secrets, such as a Hugging Face token, that a job on a rented instance needs as environment
 * variables, without them ever reaching the instance's disk, a command line, Vast's API (an
 * instance's {@code env} and {@code onstart} fields are readable there) or a log.
 *
 * <p>{@code --secret NAME} or {@code --secret NAME=PROVIDER} on the launcher's command line
 * names one. Its value is looked up here, on this machine, in this order:
 * <ol>
 *   <li>the environment variable {@code NAME};</li>
 *   <li>the studio's key file ({@code ~/.brainiac/studio/keys.properties}) under the provider
 *       ({@code <provider>.order} names its keys, {@code <provider>.key.<name>} holds each);
 *       {@code HF_TOKEN} and {@code HUGGING_FACE_HUB_TOKEN} default to the provider
 *       {@code huggingface};</li>
 *   <li>for those two names, the token files the studio and {@code huggingface-cli login} write:
 *       {@code ~/.brainiac/hf_token}, {@code $HF_TOKEN_PATH}, {@code $HF_HOME/token},
 *       {@code ~/.cache/huggingface/token}.</li>
 * </ol>
 * The values go to the instance over the SSH connection's input (see {@link VastRunner#run}),
 * where the job's shell reads them into exported variables, and {@link #mask} hides them in
 * everything the launcher prints or logs.
 */
public final class VastSecrets {

    private static final Pattern NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final Pattern PROVIDER = Pattern.compile("[A-Za-z0-9_.-]+");
    private static final List<String> HF_NAMES = List.of("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN");

    private final Map<String, String> values;

    private VastSecrets(Map<String, String> values) {
        this.values = Collections.unmodifiableMap(values);
    }

    /** No secrets. */
    public static final VastSecrets NONE = new VastSecrets(Map.of());

    /**
     * Makes secrets from known values, for tests and callers that hold them already.
     *
     * @param values variable name to value
     * @return the secrets
     */
    public static VastSecrets of(Map<String, String> values) {
        Map<String, String> m = new LinkedHashMap<>();
        values.forEach((k, v) -> m.put(checkName(k), checkValue(k, v)));
        return new VastSecrets(m);
    }

    /**
     * Looks up the secrets named on the command line on this machine.
     *
     * @param specs {@code NAME} or {@code NAME=PROVIDER} each
     * @return the secrets
     * @throws IllegalArgumentException if a name is malformed or a value cannot be found; the
     *         message names the secret, never a value
     */
    public static VastSecrets resolve(List<String> specs) {
        Path home = Path.of(System.getProperty("user.home"));
        return resolve(specs, System.getenv(), home.resolve(".brainiac").resolve("studio").resolve("keys.properties"), home);
    }

    /**
     * Looks up secrets in the given environment, key file and home folder.
     *
     * @param specs {@code NAME} or {@code NAME=PROVIDER} each
     * @param env this machine's environment
     * @param keyFile the studio's key file; it need not exist
     * @param home the home folder, for the Hugging Face token files
     * @return the secrets
     */
    static VastSecrets resolve(List<String> specs, Map<String, String> env, Path keyFile, Path home) {
        Map<String, String> m = new LinkedHashMap<>();
        Properties keys = null;
        for (String spec : specs) {
            int eq = spec.indexOf('=');
            String name = checkName(eq < 0 ? spec : spec.substring(0, eq));
            String provider = eq < 0 ? (HF_NAMES.contains(name) ? "huggingface" : null) : spec.substring(eq + 1);
            if (provider != null && !PROVIDER.matcher(provider).matches()) {
                throw new IllegalArgumentException("--secret " + name + ": bad provider name '" + provider + "'");
            }
            String value = env.get(name);
            if (value == null || value.isBlank()) {
                if (keys == null) {
                    keys = load(keyFile);
                }
                value = provider == null ? null : fromKeys(keys, provider).orElse(null);
            }
            if ((value == null || value.isBlank()) && HF_NAMES.contains(name)) {
                value = hfTokenFile(env, home).orElse(null);
            }
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException("--secret " + name + ": no value found. Set the environment variable " + name
                        + (provider == null ? ", or name a studio key provider with --secret " + name + "=PROVIDER"
                        : ", or save a " + provider + " key in the studio (" + keyFile + ")") + ".");
            }
            m.put(name, checkValue(name, value.strip()));
        }
        return new VastSecrets(m);
    }

    /**
     * The secrets' names, in the order given.
     *
     * @return the names
     */
    public List<String> names() {
        return List.copyOf(values.keySet());
    }

    /**
     * Tells whether there are none.
     *
     * @return true when empty
     */
    public boolean isEmpty() {
        return values.isEmpty();
    }

    /**
     * The shell text that reads the secrets from the remote shell's input, one line each in
     * {@link #names()} order, and exports them; it ends with {@code " && "} so the job's command
     * follows on the same line. Empty when there are no secrets.
     *
     * @return the prefix
     */
    String readPrefix() {
        StringBuilder s = new StringBuilder();
        for (String n : values.keySet()) {
            s.append("IFS= read -r ").append(n).append(" && export ").append(n).append(" && ");
        }
        return s.toString();
    }

    /**
     * The values as the lines {@link #readPrefix()} reads.
     *
     * @return one line per secret, each ending in a newline
     */
    String valueLines() {
        StringBuilder s = new StringBuilder();
        values.values().forEach(v -> s.append(v).append('\n'));
        return s.toString();
    }

    /**
     * Replaces every secret value in a piece of output with {@code ***NAME***}.
     *
     * @param text the output
     * @return it with the values hidden
     */
    public String mask(String text) {
        String out = text;
        for (Map.Entry<String, String> e : values.entrySet()) {
            out = out.replace(e.getValue(), "***" + e.getKey() + "***");
        }
        return out;
    }

    @Override
    public String toString() {
        // never the values
        return "VastSecrets" + values.keySet();
    }

    private static String checkName(String name) {
        if (!NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("--secret: '" + name + "' is not an environment variable name");
        }
        return name;
    }

    private static String checkValue(String name, String value) {
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("--secret " + name + " is empty");
        }
        if (value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0 || value.indexOf('\0') >= 0) {
            throw new IllegalArgumentException("--secret " + name + " spans more than one line");
        }
        if (value.length() < 4) {
            // too short to mask in output without hiding ordinary text
            throw new IllegalArgumentException("--secret " + name + " is shorter than 4 characters");
        }
        return value;
    }

    private static Properties load(Path keyFile) {
        Properties p = new Properties();
        if (Files.isRegularFile(keyFile)) {
            try (InputStream in = Files.newInputStream(keyFile)) {
                p.load(in);
            } catch (IOException x) {
                throw new UncheckedIOException("cannot read " + keyFile, x);
            }
        }
        return p;
    }

    /** The provider's first key: by its order list, else the first saved key by name. */
    private static Optional<String> fromKeys(Properties p, String provider) {
        for (String label : p.getProperty(provider + ".order", "").split("\n")) {
            String k = p.getProperty(provider + ".key." + label.strip());
            if (!label.isBlank() && k != null && !k.isBlank()) {
                return Optional.of(k.strip());
            }
        }
        for (String name : p.stringPropertyNames().stream().sorted().toList()) {
            if (name.startsWith(provider + ".key.") && !p.getProperty(name).isBlank()) {
                return Optional.of(p.getProperty(name).strip());
            }
        }
        return Optional.empty();
    }

    private static Optional<String> hfTokenFile(Map<String, String> env, Path home) {
        List<Path> files = new ArrayList<>();
        files.add(home.resolve(".brainiac").resolve("hf_token"));
        String path = env.get("HF_TOKEN_PATH");
        if (path != null && !path.isBlank()) {
            files.add(Path.of(path));
        }
        String hfHome = env.get("HF_HOME");
        files.add(hfHome != null && !hfHome.isBlank() ? Path.of(hfHome, "token")
                : home.resolve(".cache").resolve("huggingface").resolve("token"));
        for (Path f : files) {
            try {
                if (Files.isRegularFile(f)) {
                    String t = Files.readString(f, StandardCharsets.UTF_8).strip();
                    if (!t.isEmpty()) {
                        return Optional.of(t);
                    }
                }
            } catch (IOException e) {
                // try the next one
            }
        }
        return Optional.empty();
    }
}
