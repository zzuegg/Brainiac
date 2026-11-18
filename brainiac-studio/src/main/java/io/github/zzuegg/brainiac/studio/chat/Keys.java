package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.AclEntry;
import java.nio.file.attribute.AclEntryPermission;
import java.nio.file.attribute.AclEntryType;
import java.nio.file.attribute.AclFileAttributeView;
import java.nio.file.attribute.PosixFilePermissions;
import java.nio.file.attribute.UserPrincipal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * API keys for remote providers. Each provider can hold several keys, each with a name such as
 * {@code work} or {@code personal}; they are tried in the order they were added. Keys saved in
 * the studio go to a file only the user's account can read, and the provider's usual
 * environment variable counts as one more key after them. Keys are never sent to the page:
 * only their names, where they come from, and their last four characters.
 */
public final class Keys {

    /** The providers that take keys, and the environment variables each one reads. */
    static final Map<String, List<String>> ENV = env();

    private static Map<String, List<String>> env() {
        Map<String, List<String>> m = new java.util.LinkedHashMap<>();
        m.put("anthropic", List.of("ANTHROPIC_API_KEY"));
        m.put(Preset.OPENAI.id(), Preset.OPENAI.env());
        m.put("google", List.of("GEMINI_API_KEY", "GOOGLE_API_KEY"));
        Preset.COMPATIBLE.forEach(p -> m.put(p.id(), p.env()));
        m.put(VastAi.PROVIDER, List.of("VAST_API_KEY"));   // GPUs to rent, not a chat provider
        return Map.copyOf(m);
    }

    private static final Pattern LABEL = Pattern.compile("[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}");

    /**
     * One key, described without its value.
     *
     * @param label its name
     * @param source {@code saved}, {@code environment}, or {@code file} for a token file
     * @param hint its last four characters
     */
    public record Entry(String label, String source, String hint) {
    }

    /** The Hugging Face token file Brainiac shares between the studio and the core's downloader. */
    public static final String HF_TOKEN_FILE = "hf_token";

    private final Path file;
    private final Map<String, String> env;
    private final Map<String, List<TokenFile>> files;
    private final Properties saved = new Properties();
    private final Map<String, String> lent = new java.util.HashMap<>();   // in memory only

    /** The name of a key another studio lent this one: it is tried first and never written anywhere. */
    public static final String LENT = "lent by the studio";

    /**
     * Holds a key in memory for as long as this program runs, ahead of every other key of the
     * provider. It is never written to a file.
     *
     * @param provider the provider id
     * @param key the key, or null or blank to forget the one held
     */
    public synchronized void lend(String provider, String key) {
        if (key == null || key.isBlank()) {
            lent.remove(provider);
        } else {
            lent.put(provider, key.strip());
        }
    }

    /**
     * Returns whether a key is held in memory for a provider.
     *
     * @param provider the provider id
     * @return whether one was lent
     */
    public synchronized boolean lent(String provider) {
        return lent.containsKey(provider);
    }

    /**
     * A file a provider's token may be read from.
     *
     * @param label how the page names it
     * @param path where it is
     * @param managed whether the studio writes and forgets it, or only reads it
     */
    public record TokenFile(String label, Path path, boolean managed) {
    }

    /**
     * Opens the key store with no token files.
     *
     * @param file the file saved keys go to; it need not exist yet
     * @param env the environment to fall back to
     */
    public Keys(Path file, Map<String, String> env) {
        this(file, env, Map.of());
    }

    /**
     * Opens the key store.
     *
     * @param file the file saved keys go to; it need not exist yet
     * @param env the environment to fall back to
     * @param files token files per provider: a managed one is tried right after the saved keys,
     *     the others after the environment
     */
    public Keys(Path file, Map<String, String> env, Map<String, List<TokenFile>> files) {
        this.file = file;
        this.env = Map.copyOf(env);
        this.files = Map.copyOf(files);
        if (Files.isRegularFile(file)) {
            try (InputStream in = Files.newInputStream(file)) {
                saved.load(in);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + file, e);
            }
        }
    }

    /**
     * Lists a provider's keys in the order they are tried.
     *
     * @param provider the provider id
     * @return the keys, described without their values
     */
    public synchronized List<Entry> list(String provider) {
        List<Entry> out = new ArrayList<>();
        if (lent.containsKey(provider)) {
            out.add(new Entry(LENT, "memory", hint(lent.get(provider))));
        }
        for (String label : order(provider)) {
            out.add(new Entry(label, "saved", hint(saved.getProperty(name(provider, label)))));
        }
        for (TokenFile f : files.getOrDefault(provider, List.of())) {
            if (f.managed()) {
                read(f).ifPresent(k -> out.add(new Entry(f.label(), "file", hint(k))));
            }
        }
        envKey(provider).ifPresent(e -> out.add(new Entry(e[0], "environment", hint(e[1]))));
        for (TokenFile f : files.getOrDefault(provider, List.of())) {
            if (!f.managed()) {
                read(f).ifPresent(k -> out.add(new Entry(f.label(), "file", hint(k))));
            }
        }
        return out;
    }

    /**
     * The standard token files: for Hugging Face, {@code ~/.brainiac/hf_token}, which the studio
     * writes and the core's downloader reads too, then the file {@code huggingface-cli login}
     * writes ({@code $HF_TOKEN_PATH}, or {@code token} in {@code $HF_HOME} or {@code ~/.cache/huggingface}).
     *
     * @param home the user's home folder
     * @param env the environment
     * @return the files per provider
     */
    public static Map<String, List<TokenFile>> standardFiles(Path home, Map<String, String> env) {
        List<TokenFile> hf = new ArrayList<>();
        hf.add(new TokenFile("~/.brainiac/" + HF_TOKEN_FILE, home.resolve(".brainiac").resolve(HF_TOKEN_FILE), true));
        String path = env.get("HF_TOKEN_PATH");
        String hfHome = env.get("HF_HOME");
        Path cli = path != null && !path.isBlank() ? Path.of(path)
                : hfHome != null && !hfHome.isBlank() ? Path.of(hfHome, "token")
                : home.resolve(".cache").resolve("huggingface").resolve("token");
        hf.add(new TokenFile("huggingface-cli login", cli, false));
        return Map.of("huggingface", List.copyOf(hf));
    }

    /**
     * Writes a provider's managed token file, readable only by this account.
     *
     * @param provider the provider id
     * @param token the token
     * @throws IOException if it cannot be written
     * @throws IllegalArgumentException if the provider has no managed file or the token is not one
     */
    public synchronized void saveFile(String provider, String token) throws IOException {
        TokenFile f = managed(provider);
        String t = token == null ? "" : token.strip();
        if (t.isEmpty() || t.length() > 400 || t.chars().anyMatch(c -> c <= ' ' || c > '~')) {
            throw new IllegalArgumentException("that does not look like a token: paste it on its own, without spaces");
        }
        Files.createDirectories(f.path().getParent());
        Path tmp = f.path().resolveSibling(f.path().getFileName() + ".tmp");
        Files.deleteIfExists(tmp);
        ownerOnly(Files.createFile(tmp));
        Files.writeString(tmp, t + "\n", StandardCharsets.UTF_8);
        try {
            Files.move(tmp, f.path(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, f.path(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Deletes a provider's managed token file.
     *
     * @param provider the provider id
     * @return false if there was none
     * @throws IOException if it cannot be deleted
     */
    public synchronized boolean removeFile(String provider) throws IOException {
        return Files.deleteIfExists(managed(provider).path());
    }

    private TokenFile managed(String provider) {
        return files.getOrDefault(provider, List.of()).stream().filter(TokenFile::managed).findFirst()
                .orElseThrow(() -> new IllegalArgumentException(provider + " has no token file"));
    }

    private static Optional<String> read(TokenFile f) {
        try {
            if (!Files.isRegularFile(f.path())) {
                return Optional.empty();
            }
            String t = Files.readString(f.path(), StandardCharsets.UTF_8).strip();
            return t.isEmpty() ? Optional.empty() : Optional.of(t);
        } catch (IOException | RuntimeException e) {
            return Optional.empty();
        }
    }

    /**
     * Returns whether a provider has any key.
     *
     * @param provider the provider id
     * @return true if it has one
     */
    public boolean has(String provider) {
        return !list(provider).isEmpty();
    }

    /**
     * Returns a key by name.
     *
     * @param provider the provider id
     * @param label the key's name, as {@link #list} gives it
     * @return the key, if there is one by that name
     */
    public synchronized Optional<String> get(String provider, String label) {
        if (label.equals(LENT) && lent.containsKey(provider)) {
            return Optional.of(lent.get(provider));
        }
        String k = saved.getProperty(name(provider, label));
        if (k != null && order(provider).contains(label)) {
            return Optional.of(k);
        }
        for (TokenFile f : files.getOrDefault(provider, List.of())) {
            if (f.label().equals(label)) {
                return read(f);
            }
        }
        return envKey(provider).filter(e -> e[0].equals(label)).map(e -> e[1]);
    }

    /**
     * Saves a key under a name, replacing a key of the same name and keeping its place.
     *
     * @param provider the provider id
     * @param label the name, such as {@code work}
     * @param key the key
     * @throws IOException if the file cannot be written
     */
    public synchronized void add(String provider, String label, String key) throws IOException {
        check(provider, label);
        String k = key == null ? "" : key.trim();
        if (k.isEmpty() || k.chars().anyMatch(c -> c < 0x21 || c > 0x7e)) {
            throw new IllegalArgumentException("an API key is printable ASCII without spaces");
        }
        List<String> order = order(provider);
        if (!order.contains(label)) {
            order.add(label);
        }
        saved.setProperty(name(provider, label), k);
        saved.setProperty(provider + ".order", String.join("\n", order));
        store();
    }

    /**
     * Removes a saved key.
     *
     * @param provider the provider id
     * @param label its name
     * @return true if there was one
     * @throws IOException if the file cannot be written
     */
    public synchronized boolean remove(String provider, String label) throws IOException {
        check(provider, label);
        List<String> order = order(provider);
        if (!order.remove(label)) {
            return false;
        }
        saved.remove(name(provider, label));
        saved.setProperty(provider + ".order", String.join("\n", order));
        store();
        return true;
    }

    /**
     * Moves a saved key to the front, so it is tried first.
     *
     * @param provider the provider id
     * @param label its name
     * @throws IOException if the file cannot be written
     */
    public synchronized void makeFirst(String provider, String label) throws IOException {
        check(provider, label);
        List<String> order = order(provider);
        if (!order.remove(label)) {
            throw new IllegalArgumentException("no saved " + provider + " key named " + label);
        }
        order.addFirst(label);
        saved.setProperty(provider + ".order", String.join("\n", order));
        store();
    }

    private static void check(String provider, String label) {
        if (!ENV.containsKey(provider)) {
            throw new IllegalArgumentException("unknown provider: " + provider);
        }
        if (label == null || !LABEL.matcher(label).matches()) {
            throw new IllegalArgumentException("a key's name is 1 to 40 letters, digits, spaces, dots, dashes"
                    + " or underscores");
        }
    }

    private static String name(String provider, String label) {
        return provider + ".key." + label;
    }

    private List<String> order(String provider) {
        String o = saved.getProperty(provider + ".order", "");
        List<String> labels = new ArrayList<>(o.isEmpty() ? List.of() : Arrays.asList(o.split("\n")));
        labels.removeIf(l -> saved.getProperty(name(provider, l)) == null);
        return labels;
    }

    private Optional<String[]> envKey(String provider) {
        for (String var : ENV.getOrDefault(provider, List.of())) {
            String v = env.get(var);
            if (v != null && !v.isBlank()) {
                return Optional.of(new String[] {var, v.trim()});
            }
        }
        return Optional.empty();
    }

    private static String hint(String key) {
        return key == null || key.length() < 12 ? "…" : "…" + key.substring(key.length() - 4);
    }

    private void store() throws IOException {
        Files.createDirectories(file.getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.deleteIfExists(tmp);
        ownerOnly(Files.createFile(tmp));
        try (OutputStream out = Files.newOutputStream(tmp)) {
            saved.store(out, "Brainiac Studio API keys. Readable only by this account.");
        }
        try {
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Makes a file readable and writable only by this account: rw------- on POSIX, an ACL with
     * only the owner on Windows.
     *
     * @param p the file
     * @throws IOException if its permissions cannot be set
     */
    public static void ownerOnly(Path p) throws IOException {
        if (FileSystems.getDefault().supportedFileAttributeViews().contains("posix")) {
            Files.setPosixFilePermissions(p, PosixFilePermissions.fromString("rw-------"));
            return;
        }
        AclFileAttributeView acl = Files.getFileAttributeView(p, AclFileAttributeView.class);
        if (acl != null) {
            UserPrincipal owner = Files.getOwner(p);
            acl.setAcl(List.of(AclEntry.newBuilder().setType(AclEntryType.ALLOW).setPrincipal(owner)
                    .setPermissions(Set.of(AclEntryPermission.values())).build()));
        }
    }
}
