package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A preset for agents: what they are told, which tools they get, which models they prefer, how
 * freely they act and how much they may use. A role may {@code extend} another and only add or
 * tighten; the project an agent works in still caps everything a role grants.
 *
 * @param id up to 40 lowercase letters, digits and {@code -}
 * @param name the name shown
 * @param family the group it is listed in: {@code general}, {@code coordination}, {@code build},
 *        {@code ml}, {@code content} or {@code personal}
 * @param description one line on what it does
 * @param parent the role it extends, or null
 * @param instructions what its agents are told, after the studio's own instructions
 * @param tools the tools it gets, or null for its parent's (all tools for a role with no parent)
 * @param addTools tools added to its parent's
 * @param removeTools tools taken from its parent's
 * @param model the chat model its agents think with, or null for the parent's ({@code auto} at the root)
 * @param routes a model per kind of task ({@code image}, {@code speech}, ...) its agents prefer,
 *        used when the project allows it
 * @param mode {@code plan}, {@code ask} or {@code auto}, or null for the parent's
 * @param policy a policy per risk tier that changes what the mode does, such as {@code spend: deny}
 * @param budget limits per run: {@code steps}, {@code tokens}, {@code media}, {@code turnTokens},
 *        {@code context}
 * @param canSpawn roles its agents may start as sub-agents, {@code *} for any; null for the parent's
 * @param builtin whether the studio ships it
 */
public record Role(String id, String name, String family, String description, String parent, String instructions,
        List<String> tools, List<String> addTools, List<String> removeTools, String model, Map<String, String> routes,
        String mode, Map<Risk, Risk.Policy> policy, Map<String, Integer> budget, List<String> canSpawn,
        boolean builtin) {

    /** The families roles are listed in. */
    public static final List<String> FAMILIES = List.of("general", "coordination", "build", "ml", "content", "personal");

    /** The limits a budget may set. */
    public static final List<String> LIMITS = List.of("steps", "tokens", "media", "turnTokens", "context");

    /** Checks the role. */
    public Role {
        if (id == null || !id.matches("[a-z0-9][a-z0-9-]{0,39}")) {
            throw new IllegalArgumentException("a role's id is up to 40 lowercase letters, digits and -");
        }
        name = name == null || name.isBlank() ? id : name.strip();
        family = family == null || family.isBlank() ? "general" : family.strip();
        if (!FAMILIES.contains(family)) {
            throw new IllegalArgumentException("no family " + family + ": use " + String.join(", ", FAMILIES));
        }
        description = description == null ? "" : description.strip();
        parent = parent == null || parent.isBlank() ? null : parent.strip();
        if (id.equals(parent)) {
            throw new IllegalArgumentException("a role cannot extend itself");
        }
        instructions = instructions == null ? "" : instructions.strip();
        tools = tools == null ? null : List.copyOf(tools);
        addTools = addTools == null ? List.of() : List.copyOf(addTools);
        removeTools = removeTools == null ? List.of() : List.copyOf(removeTools);
        model = model == null || model.isBlank() ? null : model.strip();
        routes = routes == null ? Map.of() : Map.copyOf(routes);
        if (mode != null && !mode.isBlank() && !List.of("plan", "ask", "auto").contains(mode)) {
            throw new IllegalArgumentException("the mode is plan, ask or auto, not " + mode);
        }
        mode = mode == null || mode.isBlank() ? null : mode;
        policy = policy == null ? Map.of() : Map.copyOf(policy);
        if (policy.get(Risk.OUTWARD) == Risk.Policy.ALLOW) {
            throw new IllegalArgumentException("what leaves the PC always waits for you: outward cannot be allow");
        }
        budget = budget == null ? Map.of() : Map.copyOf(budget);
        for (Map.Entry<String, Integer> e : budget.entrySet()) {
            if (!LIMITS.contains(e.getKey())) {
                throw new IllegalArgumentException("no limit " + e.getKey() + ": use " + String.join(", ", LIMITS));
            }
            if (e.getValue() == null || e.getValue() < 1) {
                throw new IllegalArgumentException("the limit " + e.getKey() + " must be at least 1");
            }
        }
        canSpawn = canSpawn == null ? null : List.copyOf(canSpawn);
    }

    /**
     * Reads a role.
     *
     * @param m its JSON
     * @param builtin whether the studio ships it
     * @return the role
     */
    public static Role fromJson(Map<String, Object> m, boolean builtin) {
        Map<Risk, Risk.Policy> policy = new LinkedHashMap<>();
        if (m.get("policy") != null) {
            for (Map.Entry<String, Object> e : Json.object(m.get("policy")).entrySet()) {
                if (e.getValue() != null && !String.valueOf(e.getValue()).isBlank()) {
                    policy.put(Risk.of(e.getKey()), Risk.Policy.of(String.valueOf(e.getValue())));
                }
            }
        }
        Map<String, Integer> budget = new LinkedHashMap<>();
        if (m.get("budget") != null) {
            for (Map.Entry<String, Object> e : Json.object(m.get("budget")).entrySet()) {
                if (e.getValue() instanceof Number n) {
                    budget.put(e.getKey(), n.intValue());
                }
            }
        }
        Map<String, String> routes = new LinkedHashMap<>();
        if (m.get("routes") != null) {
            for (Map.Entry<String, Object> e : Json.object(m.get("routes")).entrySet()) {
                if (e.getValue() != null && !String.valueOf(e.getValue()).isBlank()) {
                    routes.put(e.getKey(), String.valueOf(e.getValue()).strip());
                }
            }
        }
        return new Role(Json.string(m, "id", ""), Json.string(m, "name", ""), Json.string(m, "family", "general"),
                Json.string(m, "description", ""), Json.string(m, "extends", null), Json.string(m, "instructions", ""),
                strings(m.get("tools")), strings(m.get("addTools")), strings(m.get("removeTools")),
                Json.string(m, "model", null), routes, Json.string(m, "mode", null), policy, budget,
                strings(m.get("canSpawn")), builtin);
    }

    /**
     * Writes the role.
     *
     * @return its JSON
     */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("name", name);
        m.put("family", family);
        m.put("description", description);
        if (parent != null) {
            m.put("extends", parent);
        }
        m.put("instructions", instructions);
        if (tools != null) {
            m.put("tools", tools);
        }
        if (!addTools.isEmpty()) {
            m.put("addTools", addTools);
        }
        if (!removeTools.isEmpty()) {
            m.put("removeTools", removeTools);
        }
        if (model != null) {
            m.put("model", model);
        }
        if (!routes.isEmpty()) {
            m.put("routes", routes);
        }
        if (mode != null) {
            m.put("mode", mode);
        }
        if (!policy.isEmpty()) {
            Map<String, Object> p = new LinkedHashMap<>();
            for (Risk r : Risk.values()) {
                if (policy.containsKey(r)) {
                    p.put(r.key(), policy.get(r).key());
                }
            }
            m.put("policy", p);
        }
        if (!budget.isEmpty()) {
            m.put("budget", budget);
        }
        if (canSpawn != null) {
            m.put("canSpawn", canSpawn);
        }
        m.put("builtin", builtin);
        return m;
    }

    private static List<String> strings(Object o) {
        if (o == null) {
            return null;
        }
        List<String> out = new ArrayList<>();
        for (Object x : Json.array(o)) {
            String s = String.valueOf(x).strip();
            if (!s.isEmpty() && !out.contains(s)) {
                out.add(s);
            }
        }
        return out;
    }
}
