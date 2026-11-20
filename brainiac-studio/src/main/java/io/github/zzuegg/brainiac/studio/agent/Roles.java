package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * The roles agents can take: the ones the studio ships, and the user's own, kept as one JSON
 * file each in a folder. A user's role with a built-in role's id replaces it.
 */
public final class Roles {

    /** The role an agent takes when none is named. */
    public static final String DEFAULT = "assistant";

    /** How many roles a chain of {@code extends} may hold, the role itself included. */
    public static final int MAX_CHAIN = 3;

    /** Limits a run has unless its role sets them. */
    public static final Map<String, Integer> DEFAULT_BUDGET = Map.of("steps", 30, "tokens", 200_000, "media", 10,
            "turnTokens", 4096, "context", 32_000);

    /** The highest each limit may go. */
    public static final Map<String, Integer> MAX_BUDGET = Map.of("steps", 100, "tokens", 5_000_000, "media", 200,
            "turnTokens", 32_768, "context", 1_000_000);

    private static final System.Logger LOG = System.getLogger(Roles.class.getName());

    private final Path folder;
    private final Map<String, Role> builtin = new LinkedHashMap<>();
    private final Map<String, Role> own = new LinkedHashMap<>();

    /**
     * A role with what it inherits filled in.
     *
     * @param role the role as written
     * @param chain the roles it is made of, itself first
     * @param instructions everything its agents are told, the root's first
     * @param tools the tools it gets, or null for all
     * @param model the chat model, {@code auto} if none is set
     * @param routes a preferred model per kind of task
     * @param mode {@code plan}, {@code ask} or {@code auto}
     * @param policy the policy per tier where it differs from the mode's
     * @param budget every limit, with the defaults filled in
     * @param canSpawn roles it may start, {@code *} for any
     */
    public record Effective(Role role, List<String> chain, String instructions, Set<String> tools, String model,
            Map<String, String> routes, String mode, Map<Risk, Risk.Policy> policy, Map<String, Integer> budget,
            List<String> canSpawn) {

        /**
         * What happens to a call of a tier in a mode.
         *
         * @param risk the call's tier
         * @param mode the run's mode now, which may be stricter than the role's
         * @return allow, ask or deny
         */
        public Risk.Policy policyFor(Risk risk, String mode) {
            Risk.Policy base = risk.byDefault(mode);
            Risk.Policy set = policy.get(risk);
            if (set == null) {
                return base;
            }
            // a role may loosen ask to allow only in auto mode; it can always tighten
            if (set == Risk.Policy.ALLOW && !"auto".equals(mode)) {
                return base;
            }
            return risk == Risk.OUTWARD ? set.strictest(Risk.Policy.ASK) : set;
        }

        /**
         * Whether it may start agents of a role.
         *
         * @param role the role's id
         * @return true if listed, or if it may start any
         */
        public boolean spawns(String role) {
            return canSpawn.contains("*") || canSpawn.contains(role);
        }

        /**
         * Describes it.
         *
         * @return its JSON, with the role as written under {@code role}
         */
        public Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("role", role.toJson());
            m.put("chain", chain);
            m.put("instructions", instructions);
            m.put("tools", tools == null ? null : List.copyOf(tools));
            m.put("model", model);
            m.put("routes", routes);
            m.put("mode", mode);
            Map<String, Object> p = new LinkedHashMap<>();
            for (Risk r : Risk.values()) {
                p.put(r.key(), policyFor(r, mode).key());
            }
            m.put("policy", p);
            m.put("budget", budget);
            m.put("canSpawn", canSpawn);
            return m;
        }
    }

    /**
     * Opens the roles: the built-in ones and those in a folder.
     *
     * @param folder where the user's roles are kept, or null for none
     */
    public Roles(Path folder) {
        this.folder = folder;
        try (InputStream in = Roles.class.getResourceAsStream("roles.json")) {
            if (in != null) {
                for (Object o : Json.array(Json.object(Json.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8)))
                        .get("roles"))) {
                    Role r = Role.fromJson(Json.object(o), true);
                    builtin.put(r.id(), r);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read the built-in roles", e);
        }
        if (!builtin.containsKey(DEFAULT)) {
            builtin.put(DEFAULT, new Role(DEFAULT, "General assistant", "general", "Helps with anything, with every tool",
                    null, "", null, null, null, null, null, "ask", null, null, null, true));
        }
        load();
    }

    /**
     * Lists every role, built-in ones first, each family together.
     *
     * @return the roles
     */
    public synchronized List<Role> list() {
        Map<String, Role> all = new LinkedHashMap<>(builtin);
        all.putAll(own);
        List<Role> out = new ArrayList<>();
        for (String f : Role.FAMILIES) {
            all.values().stream().filter(r -> r.family().equals(f)).forEach(out::add);
        }
        return out;
    }

    /**
     * Finds a role.
     *
     * @param id its id
     * @return the user's role with that id, else the built-in one
     */
    public synchronized Optional<Role> get(String id) {
        Role r = own.get(id);
        return Optional.ofNullable(r != null ? r : builtin.get(id));
    }

    /**
     * Fills in what a role inherits.
     *
     * @param id the role's id, or null for {@value #DEFAULT}
     * @param known the tools there are, so a role naming one that is missing still works
     * @return the role as its agents get it
     */
    public synchronized Effective resolve(String id, Collection<String> known) {
        String want = id == null || id.isBlank() ? DEFAULT : id;
        List<Role> chain = new ArrayList<>();
        for (String at = want; at != null; ) {
            String here = at;
            Role r = get(here).orElseThrow(() -> new IllegalArgumentException("there is no role " + here + "; the roles are "
                    + String.join(", ", list().stream().map(Role::id).toList())));
            if (chain.stream().anyMatch(c -> c.id().equals(r.id()))) {
                throw new IllegalArgumentException("the role " + want + " extends itself through " + r.id());
            }
            chain.add(r);
            if (chain.size() > MAX_CHAIN) {
                throw new IllegalArgumentException("the role " + want + " extends more than " + (MAX_CHAIN - 1)
                        + " roles; keep chains short");
            }
            at = r.parent();
        }
        StringBuilder instructions = new StringBuilder();
        Set<String> tools = null;
        String model = "auto";
        Map<String, String> routes = new LinkedHashMap<>();
        String mode = "ask";
        Map<Risk, Risk.Policy> policy = new LinkedHashMap<>();
        Map<String, Integer> budget = new LinkedHashMap<>(DEFAULT_BUDGET);
        List<String> canSpawn = List.of();
        for (Role r : chain.reversed()) {
            if (!r.instructions().isEmpty()) {
                if (!instructions.isEmpty()) {
                    instructions.append("\n\n");
                }
                instructions.append(r.instructions());
            }
            if (r.tools() != null) {
                tools = new LinkedHashSet<>(r.tools());
            }
            if (!r.addTools().isEmpty()) {
                if (tools != null) {
                    tools.addAll(r.addTools());
                }
            }
            if (tools != null) {
                tools.removeAll(r.removeTools());
            }
            if (r.model() != null) {
                model = r.model();
            }
            routes.putAll(r.routes());
            if (r.mode() != null) {
                mode = r.mode();
            }
            policy.putAll(r.policy());
            budget.putAll(r.budget());
            if (r.canSpawn() != null) {
                canSpawn = r.canSpawn();
            }
        }
        if (tools != null && known != null) {
            tools.retainAll(known);
        }
        budget.replaceAll((k, v) -> Math.min(v, MAX_BUDGET.get(k)));
        return new Effective(chain.getFirst(), chain.stream().map(Role::id).toList(), instructions.toString(), tools,
                model, Map.copyOf(routes), mode, Map.copyOf(policy), Map.copyOf(budget), canSpawn);
    }

    /**
     * Keeps one of the user's roles.
     *
     * @param role the role
     * @return the role as kept
     * @throws IOException if it cannot be written
     */
    public synchronized Role save(Role role) throws IOException {
        if (folder == null) {
            throw new IllegalStateException("roles are not kept anywhere");
        }
        Role r = new Role(role.id(), role.name(), role.family(), role.description(), role.parent(), role.instructions(),
                role.tools(), role.addTools(), role.removeTools(), role.model(), role.routes(), role.mode(),
                role.policy(), role.budget(), role.canSpawn(), false);
        Role before = own.put(r.id(), r);
        try {
            resolve(r.id(), null);   // its chain must hold together
        } catch (IllegalArgumentException e) {
            if (before == null) {
                own.remove(r.id());
            } else {
                own.put(r.id(), before);
            }
            throw e;
        }
        Files.createDirectories(folder);
        Path tmp = folder.resolve(r.id() + ".json.tmp");
        Files.writeString(tmp, Json.write(r.toJson()), StandardCharsets.UTF_8);
        Files.move(tmp, folder.resolve(r.id() + ".json"), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        return r;
    }

    /**
     * Removes one of the user's roles; a built-in role it replaced comes back.
     *
     * @param id the role
     * @return whether there was one
     * @throws IOException if its file cannot be removed
     */
    public synchronized boolean delete(String id) throws IOException {
        if (!own.containsKey(id)) {
            if (builtin.containsKey(id)) {
                throw new IllegalArgumentException("the studio's own roles cannot be removed; save a role with the "
                        + "same id to change it");
            }
            return false;
        }
        for (Role r : list()) {
            if (id.equals(r.parent()) && !r.id().equals(id) && !(builtin.containsKey(id))) {
                throw new IllegalArgumentException("the role " + r.id() + " extends " + id + "; change it first");
            }
        }
        own.remove(id);
        if (folder != null) {
            Files.deleteIfExists(folder.resolve(id + ".json"));
        }
        return true;
    }

    /**
     * Whether a role is the studio's own, as shipped.
     *
     * @param id the role
     * @return true if it is built in and not replaced
     */
    public synchronized boolean shipped(String id) {
        return builtin.containsKey(id) && !own.containsKey(id);
    }

    private void load() {
        if (folder == null || !Files.isDirectory(folder)) {
            return;
        }
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(folder, "*.json")) {
            for (Path p : ds) {
                try {
                    Role r = Role.fromJson(Json.object(Json.parse(Files.readString(p, StandardCharsets.UTF_8))), false);
                    own.put(r.id(), r);
                } catch (IOException | RuntimeException e) {
                    LOG.log(System.Logger.Level.WARNING, "cannot read role " + p, e);
                }
            }
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot list " + folder, e);
        }
    }
}
