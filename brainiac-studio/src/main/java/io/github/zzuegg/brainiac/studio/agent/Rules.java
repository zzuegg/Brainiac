package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

/**
 * The user's rules for agent calls, in order: the first rule that matches a call decides
 * whether it runs, waits for the user or is refused. A rule can name tools, risk tiers, paths
 * (inside or outside the project's folder, or patterns), reading or writing, a project and a
 * role. Calls no rule matches are judged by their tier, role and mode. Some floors hold
 * whatever the rules say: what leaves the PC always asks, and so does anything risky after an
 * agent read text from outside.
 */
public final class Rules {

    /** Paths inside the project's folder. */
    public static final String PROJECT = "@project";
    /** Paths outside the project's folder. */
    public static final String OUTSIDE = "@outside";

    /**
     * A rule.
     *
     * @param id its id
     * @param effect what it does to the calls it matches
     * @param tools tool names, {@code *} for any part; empty for every tool
     * @param risks tiers; empty for every tier
     * @param paths {@code @project}, {@code @outside} or glob patterns such as {@code ~/Documents/**};
     *     empty for calls whatever they reach
     * @param access {@code read}, {@code write} or {@code any}
     * @param project a project's id, or empty for every project
     * @param role a role's id, or empty for every role
     * @param note why the rule is there
     * @param enabled whether it is used
     */
    public record Rule(String id, Risk.Policy effect, List<String> tools, List<Risk> risks, List<String> paths,
            String access, String project, String role, String note, boolean enabled) {

        /** Checks the rule. */
        public Rule {
            if (id == null || id.isBlank()) {
                id = UUID.randomUUID().toString().substring(0, 8);
            }
            if (effect == null) {
                throw new IllegalArgumentException("a rule says allow, ask or deny");
            }
            tools = tools == null ? List.of() : List.copyOf(tools);
            risks = risks == null ? List.of() : List.copyOf(risks);
            paths = paths == null ? List.of() : List.copyOf(paths);
            access = access == null || access.isBlank() ? "any" : access.strip().toLowerCase(Locale.ROOT);
            if (!List.of("read", "write", "any").contains(access)) {
                throw new IllegalArgumentException("a rule's access is read, write or any");
            }
            project = project == null ? "" : project.strip();
            role = role == null ? "" : role.strip();
            note = note == null ? "" : note.strip();
            for (String p : paths) {
                if (!p.equals(PROJECT) && !p.equals(OUTSIDE)) {
                    FileSystems.getDefault().getPathMatcher("glob:" + expand(p));   // fails on a bad pattern
                }
            }
        }

        /**
         * Reads a rule.
         *
         * @param m its JSON
         * @return the rule
         */
        public static Rule fromJson(Map<String, Object> m) {
            List<Risk> risks = new ArrayList<>();
            for (String r : strings(m.get("risks"))) {
                risks.add(Risk.of(r));
            }
            return new Rule(Json.string(m, "id", ""), Risk.Policy.of(Json.string(m, "effect", "")),
                    strings(m.get("tools")), risks, strings(m.get("paths")), Json.string(m, "access", "any"),
                    Json.string(m, "project", ""), Json.string(m, "role", ""), Json.string(m, "note", ""),
                    !Boolean.FALSE.equals(m.get("enabled")));
        }

        /**
         * Describes the rule.
         *
         * @return its JSON
         */
        public Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("effect", effect.key());
            m.put("tools", tools);
            m.put("risks", risks.stream().map(Risk::key).toList());
            m.put("paths", paths);
            m.put("access", access);
            m.put("project", project);
            m.put("role", role);
            m.put("note", note);
            m.put("enabled", enabled);
            return m;
        }

        /**
         * Says what the rule does, in a line.
         *
         * @return a description
         */
        public String describe() {
            StringBuilder b = new StringBuilder(effect.key());
            b.append(tools.isEmpty() ? " any tool" : " " + String.join(", ", tools));
            if (!risks.isEmpty()) {
                b.append(" that ").append(String.join(" or ", risks.stream().map(Risk::key).toList()));
            }
            if (!paths.isEmpty()) {
                b.append(access.equals("any") ? " reaching " : access.equals("read") ? " reading " : " writing ")
                        .append(String.join(", ", paths));
            }
            if (!project.isEmpty()) {
                b.append(" in project ").append(project);
            }
            if (!role.isEmpty()) {
                b.append(" for role ").append(role);
            }
            return b.toString();
        }

        boolean matches(String tool, Risk risk, String callRole, String callProject, Reach.Access a) {
            if (!enabled || !project.isEmpty() && !project.equals(callProject) || !role.isEmpty() && !role.equals(callRole)) {
                return false;
            }
            if (!risks.isEmpty() && !risks.contains(risk)) {
                return false;
            }
            if (!tools.isEmpty() && tools.stream().noneMatch(t -> name(t, tool))) {
                return false;
            }
            if (paths.isEmpty()) {
                return true;
            }
            if (a == null || access.equals("read") && a.write() || access.equals("write") && !a.write()) {
                return false;
            }
            for (String p : paths) {
                boolean hit = switch (p) {
                    case PROJECT -> Reach.inside(a.path());
                    case OUTSIDE -> !Reach.inside(a.path());
                    default -> FileSystems.getDefault().getPathMatcher("glob:" + expand(p)).matches(a.path());
                };
                if (hit) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * What the rules decided for a call.
     *
     * @param policy the strictest decision over everything the call reaches, or null if no rule matched
     * @param rules the rules that decided
     * @param outside the paths outside the project's folder the call reaches
     */
    public record Verdict(Risk.Policy policy, List<Rule> rules, List<Reach.Access> outside) {
    }

    private final Path file;
    private final List<Rule> rules = new ArrayList<>();

    /**
     * Opens the rules kept in a file; the shipped rules if there is none.
     *
     * @param file where they are kept
     */
    public Rules(Path file) {
        this.file = file;
        if (file != null && Files.isRegularFile(file)) {
            try {
                Object list = Json.object(Json.parse(Files.readString(file, StandardCharsets.UTF_8))).get("rules");
                if (list instanceof List<?> l) {
                    for (Object o : l) {
                        rules.add(Rule.fromJson(Json.object(o)));
                    }
                }
                return;
            } catch (IOException | RuntimeException e) {
                System.getLogger(Rules.class.getName()).log(System.Logger.Level.WARNING,
                        "cannot read " + file + "; using the shipped rules", e);
                rules.clear();
            }
        }
        rules.addAll(shipped());
    }

    /**
     * The rules a new studio starts with.
     *
     * @return deny secrets, ask before reading or writing outside the project's folder
     */
    public static List<Rule> shipped() {
        return List.of(
                new Rule("secrets", Risk.Policy.DENY, List.of(), List.of(), List.of("~/.ssh/**", "~/.gnupg/**",
                        "~/.aws/**", "~/.config/gcloud/**", "~/.kube/**", "~/.docker/config.json", "**/.env",
                        "**/*.pem", "**/id_rsa*", "**/id_ed25519*"), "any", "", "",
                        "keys and passwords stay out of agents' reach", true),
                new Rule("outside-read", Risk.Policy.ASK, List.of(), List.of(), List.of(OUTSIDE), "read", "", "",
                        "reading outside the project's folder asks; one approval covers that folder for the run", true),
                new Rule("outside-write", Risk.Policy.ASK, List.of(), List.of(), List.of(OUTSIDE), "write", "", "",
                        "writing outside the project's folder asks every time", true));
    }

    /**
     * Lists the rules in order.
     *
     * @return the rules
     */
    public synchronized List<Rule> list() {
        return List.copyOf(rules);
    }

    /**
     * Replaces every rule.
     *
     * @param list the new rules, in order
     * @throws IOException if they cannot be kept
     */
    public synchronized void save(List<Rule> list) throws IOException {
        List<String> ids = new ArrayList<>();
        for (Rule r : list) {
            if (ids.contains(r.id())) {
                throw new IllegalArgumentException("two rules have the id " + r.id());
            }
            ids.add(r.id());
        }
        rules.clear();
        rules.addAll(list);
        write();
    }

    /**
     * Adds a rule.
     *
     * @param rule the rule
     * @param at its place, 0 for first; past the end for last
     * @return the rule as kept
     * @throws IOException if it cannot be kept
     */
    public synchronized Rule add(Rule rule, int at) throws IOException {
        Rule r = rule;
        if (rules.stream().anyMatch(x -> x.id().equals(rule.id()))) {
            r = new Rule(rule.id() + "-" + UUID.randomUUID().toString().substring(0, 4), rule.effect(), rule.tools(),
                    rule.risks(), rule.paths(), rule.access(), rule.project(), rule.role(), rule.note(), rule.enabled());
        }
        rules.add(Math.clamp(at, 0, rules.size()), r);
        write();
        return r;
    }

    /**
     * Judges a call.
     *
     * @param tool the tool
     * @param risk its tier for these arguments
     * @param role the agent's role
     * @param project the project's id, or empty
     * @param reach what the call reads and writes
     * @return the verdict
     */
    public synchronized Verdict judge(String tool, Risk risk, String role, String project, List<Reach.Access> reach) {
        List<Reach.Access> outside = reach.stream().filter(a -> !Reach.inside(a.path())).toList();
        Risk.Policy policy = null;
        List<Rule> used = new ArrayList<>();
        List<Reach.Access> each = reach.isEmpty() ? java.util.Collections.singletonList(null) : reach;
        for (Reach.Access a : each) {
            for (Rule r : rules) {
                if (r.matches(tool, risk, role, project == null ? "" : project, a)) {
                    policy = policy == null ? r.effect() : policy.strictest(r.effect());
                    if (!used.contains(r)) {
                        used.add(r);
                    }
                    break;
                }
            }
        }
        return new Verdict(policy, used, outside);
    }

    /**
     * Describes the rules.
     *
     * @return {@code {rules: [...]}}
     */
    public synchronized Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("rules", rules.stream().map(Rule::toJson).toList());
        return m;
    }

    private void write() throws IOException {
        if (file == null) {
            return;
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, Json.write(toJson()), StandardCharsets.UTF_8);
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    static String expand(String pattern) {
        String p = pattern.strip().replace('\\', '/');
        if (p.equals("~") || p.startsWith("~/")) {
            return System.getProperty("user.home").replace('\\', '/') + p.substring(1);
        }
        return p;
    }

    private static boolean name(String pattern, String tool) {
        if (!pattern.contains("*")) {
            return pattern.equals(tool);
        }
        return tool.matches(java.util.regex.Pattern.quote(pattern).replace("*", "\\E.*\\Q"));
    }

    private static List<String> strings(Object o) {
        List<String> out = new ArrayList<>();
        if (o instanceof List<?> l) {
            for (Object x : l) {
                if (x != null && !String.valueOf(x).isBlank()) {
                    out.add(String.valueOf(x).strip());
                }
            }
        } else if (o instanceof String s && !s.isBlank()) {
            for (String x : s.split(",")) {
                if (!x.isBlank()) {
                    out.add(x.strip());
                }
            }
        }
        return out;
    }
}
