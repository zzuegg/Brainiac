package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.builder.Graphs;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.items.ItemKind;
import io.github.zzuegg.brainiac.studio.items.Owners;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Tools that edit the builder's networks where the user sees them: a network is read, changed
 * a few blocks at a time and saved under its name, so the builder's canvas shows each change as
 * it is made. A change that would break a block that worked is refused, as on the canvas.
 *
 * <p>A network is the builder's graph: {@code {"nodes": [...], "output": "id"}}, each node with
 * an {@code id}, an {@code op} ({@code input}, {@code layer}, {@code add}, {@code multiply},
 * {@code concat}), its {@code inputs}, and for a layer its catalog {@code entry}, {@code args}
 * and {@code options}; {@code x} and {@code y} place it on the canvas.
 */
public final class NetworkTools {

    private static final List<String> OPS = List.of("input", "layer", "add", "multiply", "concat");

    private final Graphs graphs;
    private final Owners owners;
    private volatile ModelGraph checker;
    private volatile LayerCatalog catalog;

    /**
     * Creates the tools.
     *
     * @param graphs where networks are kept
     * @param owners which project each network belongs to
     */
    public NetworkTools(Graphs graphs, Owners owners) {
        this.graphs = graphs;
        this.owners = owners;
    }

    private ModelGraph checker() {
        if (checker == null) {
            synchronized (this) {
                if (checker == null) {
                    catalog = LayerCatalog.discover();
                    checker = new ModelGraph(catalog);
                }
            }
        }
        return checker;
    }

    private LayerCatalog catalog() {
        checker();
        return catalog;
    }

    private static String project() {
        return ChatService.SCOPE.isBound() && ChatService.SCOPE.get() instanceof Project p ? p.id() : Project.DEFAULT;
    }

    // a network of the caller's project
    private Map<String, Object> network(String name) {
        try {
            Map<String, Object> g = graphs.get(Graphs.name(name)).orElseThrow(() ->
                    new IllegalArgumentException("there is no network " + name + "; list_networks lists them"));
            String owner = owners.of(ItemKind.GRAPH, Graphs.name(name));
            String mine = project();
            if (!mine.equals(owner == null ? Project.DEFAULT : owner)) {
                throw new IllegalArgumentException("the network " + name + " is in another project");
            }
            return normalized(g);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    // options as the checker reads them: a list of {key, args}, also when the page kept them by key
    @SuppressWarnings("unchecked")
    static Map<String, Object> normalized(Map<String, Object> g) {
        Map<String, Object> out = new LinkedHashMap<>(g);
        List<Object> nodes = new ArrayList<>();
        for (Object o : Json.array(g.getOrDefault("nodes", List.of()))) {
            Map<String, Object> n = new LinkedHashMap<>(Json.object(o));
            if (n.get("options") instanceof Map<?, ?> m) {
                List<Object> list = new ArrayList<>();
                ((Map<String, Object>) m).forEach((k, v) -> list.add(new LinkedHashMap<>(Map.of("key", k, "args", v))));
                n.put("options", list);
            }
            nodes.add(n);
        }
        out.put("nodes", nodes);
        return out;
    }

    /**
     * Lists the networks.
     *
     * @return each network of the caller's project
     */
    @Tools.Tool(name = "list_networks", value = "Lists the networks of this project in the builder: their names, how "
            + "many blocks they have, and whether they are built.")
    @Tier(value = Risk.READ)
    @Returns("[{name, blocks, built, updated}]")
    public List<Map<String, Object>> listNetworks() {
        String mine = project();
        List<Map<String, Object>> out = new ArrayList<>();
        try {
            for (Map<String, Object> g : graphs.list()) {
                String name = Json.string(g, "name", "");
                String owner = owners.of(ItemKind.GRAPH, name);
                if (!mine.equals(owner == null ? Project.DEFAULT : owner)) {
                    continue;
                }
                out.add(Map.of("name", name, "blocks", Json.array(g.getOrDefault("nodes", List.of())).size(),
                        "built", Boolean.TRUE.equals(g.get("built")), "updated", Json.integer(g, "updated", 0)));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        return out;
    }

    /**
     * Reads a network.
     *
     * @param name its name
     * @return its blocks and their shapes
     */
    @Tools.Tool(name = "read_network", value = "Reads a network in the builder: every block with its settings and "
            + "inputs, each block's output shape or why it does not fit, and whether the whole network checks out.")
    @Tier(value = Risk.READ, resultChars = 24_000)
    @Returns("{name, output?, blocks: [{id, op, entry?, args?, options?, inputs, shape?, error?}], ok, problem?, "
            + "outputShape?}")
    public Map<String, Object> readNetwork(@Tools.Param("the network's name") String name) {
        Map<String, Object> g = network(name);
        return describe(Graphs.name(name), g, checker().check(g));
    }

    private static Map<String, Object> describe(String name, Map<String, Object> g, ModelGraph.Check c) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("name", name);
        if (g.get("output") != null) {
            out.put("output", g.get("output"));
        }
        List<Map<String, Object>> blocks = new ArrayList<>();
        Map<String, Object> nodes = Json.object(c.toJson().get("nodes"));
        for (Object o : Json.array(g.get("nodes"))) {
            Map<String, Object> n = Json.object(o);
            Map<String, Object> b = new LinkedHashMap<>();
            for (String k : List.of("id", "op", "entry", "shape", "args", "options", "inputs")) {
                if (n.get(k) != null) {
                    b.put(k, n.get(k));
                }
            }
            Object state = nodes.get(Json.string(n, "id", ""));
            if (state instanceof Map<?, ?> s) {
                if (s.get("shape") != null) {
                    b.put("outputShape", s.get("shape"));
                }
                if (s.get("error") != null) {
                    b.put("error", s.get("error"));
                }
            }
            blocks.add(b);
        }
        out.put("blocks", blocks);
        Map<String, Object> cj = c.toJson();
        out.put("ok", cj.get("ok"));
        for (String k : List.of("problem", "outputShape")) {
            if (cj.get(k) != null) {
                out.put(k, cj.get(k));
            }
        }
        return out;
    }

    /**
     * Makes a network.
     *
     * @param name its name
     * @param shape the input's shape
     * @return the network
     */
    @Tools.Tool(name = "create_network", value = "Makes a new network in the builder with one Input block, which "
            + "edit_network then grows. It belongs to this project and shows in the builder at once.")
    @Tier(value = Risk.CREATE, what = "makes a network in the builder")
    @Returns("{name, blocks, ok, ...}: as read_network")
    public Map<String, Object> createNetwork(@Tools.Param("1 to 64 letters, digits, spaces, dots, dashes or underscores") String name,
            @Tools.Param("the shape of one input sample, channels first, such as [3, 32, 32] or [784]") List<Integer> shape) {
        String n = Graphs.name(name);
        try {
            if (graphs.get(n).isPresent()) {
                throw new IllegalArgumentException("there is a network named " + n + " already; edit it or pick another name");
            }
            Map<String, Object> input = new LinkedHashMap<>();
            input.put("id", "in");
            input.put("op", "input");
            input.put("shape", shape == null || shape.isEmpty() ? List.of(16) : shape);
            input.put("x", 40);
            input.put("y", 120);
            Map<String, Object> g = new LinkedHashMap<>(Map.of("nodes", List.of(input)));
            graphs.save(n, g);
            owners.claim(ItemKind.GRAPH, n, project());
            return describe(n, g, checker().check(g));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    /**
     * Changes a network.
     *
     * @param name the network
     * @param changes what to change, in order
     * @return the network afterwards
     */
    @Tools.Tool(name = "edit_network", value = "Changes a network in the builder, a few steps at a time; the user sees "
            + "each change on the canvas and can undo it. Steps, applied in order: "
            + "{\"do\": \"add\", \"id\"?, \"layer\": a key from list_layers, \"args\"?: [..] or {name: value}, "
            + "\"options\"?: {optionKey: [args]}, \"inputs\"?: [ids]} or {\"do\": \"add\", \"op\": \"add\"|\"multiply\"|"
            + "\"concat\"|\"input\", \"inputs\"?, \"shape\"?}; {\"do\": \"set\", \"id\", \"args\"?, \"options\"? (an option set "
            + "to null is switched off), \"shape\"?}; {\"do\": \"connect\", \"from\", \"to\"}; {\"do\": \"disconnect\", "
            + "\"from\", \"to\"}; {\"do\": \"remove\", \"id\"}; {\"do\": \"output\", \"id\"}. Args left out get a first value "
            + "that works. The change is kept only if no block that worked stops working.")
    @Tier(value = Risk.CREATE, what = "changes a network in the builder")
    @Returns("{name, blocks, ok, problem?, outputShape?, added?: [ids]}: as read_network, after the change")
    public Map<String, Object> editNetwork(@Tools.Param("the network's name") String name,
            @Tools.Param("the steps") List<Map<String, Object>> changes) {
        if (changes == null || changes.isEmpty()) {
            throw new IllegalArgumentException("say what to change");
        }
        String n = Graphs.name(name);
        Map<String, Object> before = network(n);
        ModelGraph.Check was = checker().check(before);
        Map<String, Object> g = Json.object(Json.parse(Json.write(before)));
        List<String> added = new ArrayList<>();
        int i = 0;
        for (Map<String, Object> c : changes) {
            i++;
            try {
                apply(g, c, added);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("step " + i + " (" + Json.string(c, "do", "?") + "): " + e.getMessage()
                        + "; nothing was changed");
            }
        }
        ModelGraph.Check now = checker().check(g);
        List<String> broken = new ArrayList<>();
        now.errors().forEach((id, err) -> {
            if (was.shapes().containsKey(id)) {
                broken.add(id + ": " + err);
            }
        });
        if (!broken.isEmpty()) {
            throw new IllegalArgumentException("refused, it would break blocks that worked: " + String.join("; ", broken)
                    + ". Nothing was changed");
        }
        try {
            graphs.save(n, g);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
        Map<String, Object> out = describe(n, g, now);
        if (!added.isEmpty()) {
            out.put("added", added);
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private void apply(Map<String, Object> g, Map<String, Object> c, List<String> added) {
        List<Object> nodes = (List<Object>) g.get("nodes");
        String what = Json.string(c, "do", "");
        switch (what) {
            case "add" -> {
                String op = c.get("layer") != null ? "layer" : Json.string(c, "op", "");
                if (!OPS.contains(op)) {
                    throw new IllegalArgumentException("add a \"layer\" (a key from list_layers) or an \"op\": input, add, "
                            + "multiply or concat");
                }
                Map<String, Object> node = new LinkedHashMap<>();
                String base = op.equals("layer") ? group(Json.string(c, "layer", "")) : op;
                String id = Json.string(c, "id", null);
                if (id == null || id.isBlank()) {
                    id = freeId(nodes, base);
                } else if (find(nodes, id) != null) {
                    throw new IllegalArgumentException("there is a block " + id + " already");
                }
                node.put("id", id);
                node.put("op", op);
                if (op.equals("layer")) {
                    LayerCatalog.Entry e = entry(Json.string(c, "layer", ""));
                    node.put("entry", e.key());
                    node.put("args", args(e, c.get("args"), null));
                    node.put("options", options(e, c.get("options"), new ArrayList<>()));
                }
                if (op.equals("input")) {
                    node.put("shape", c.get("shape") == null ? List.of(16) : Json.array(c.get("shape")));
                }
                List<String> inputs = strings(c.get("inputs"));
                for (String from : inputs) {
                    if (find(nodes, from) == null) {
                        throw new IllegalArgumentException("there is no block " + from + " to take input from");
                    }
                }
                node.put("inputs", inputs);
                place(nodes, node, inputs);
                nodes.add(node);
                added.add(id);
            }
            case "set" -> {
                Map<String, Object> node = need(nodes, Json.string(c, "id", ""));
                if (c.get("shape") != null) {
                    if (!"input".equals(node.get("op"))) {
                        throw new IllegalArgumentException("only an Input block has a shape; " + node.get("id") + " is not one");
                    }
                    node.put("shape", Json.array(c.get("shape")));
                }
                if ("layer".equals(node.get("op"))) {
                    LayerCatalog.Entry e = entry(Json.string(node, "entry", ""));
                    if (c.containsKey("args")) {
                        node.put("args", args(e, c.get("args"), Json.array(node.getOrDefault("args", List.of()))));
                    }
                    if (c.containsKey("options")) {
                        List<Object> current = new ArrayList<>(Json.array(node.getOrDefault("options", List.of())));
                        node.put("options", options(e, c.get("options"), current));
                    }
                } else if (c.containsKey("args") || c.containsKey("options")) {
                    throw new IllegalArgumentException(node.get("id") + " has no settings");
                }
            }
            case "connect" -> {
                Map<String, Object> to = need(nodes, Json.string(c, "to", ""));
                String from = Json.string(c, "from", "");
                need(nodes, from);
                if (from.equals(to.get("id")) || "input".equals(to.get("op"))) {
                    throw new IllegalArgumentException("an Input block, or a block itself, takes no wire from " + from);
                }
                List<String> ins = new ArrayList<>(strings(to.get("inputs")));
                if (!ins.contains(from)) {
                    if (multi(to)) {
                        ins.add(from);
                    } else {
                        ins = new ArrayList<>(List.of(from));
                    }
                }
                to.put("inputs", ins);
            }
            case "disconnect" -> {
                Map<String, Object> to = need(nodes, Json.string(c, "to", ""));
                List<String> ins = new ArrayList<>(strings(to.get("inputs")));
                if (!ins.remove(Json.string(c, "from", ""))) {
                    throw new IllegalArgumentException(c.get("from") + " is not wired into " + to.get("id"));
                }
                to.put("inputs", ins);
            }
            case "remove" -> {
                String id = Json.string(c, "id", "");
                Map<String, Object> node = need(nodes, id);
                nodes.remove(node);
                for (Object o : nodes) {
                    Map<String, Object> m = (Map<String, Object>) o;
                    m.put("inputs", new ArrayList<>(strings(m.get("inputs")).stream().filter(x -> !x.equals(id)).toList()));
                }
                if (id.equals(g.get("output"))) {
                    g.remove("output");
                }
            }
            case "output" -> {
                String id = Json.string(c, "id", "");
                need(nodes, id);
                g.put("output", id);
            }
            default -> throw new IllegalArgumentException("a step does add, set, connect, disconnect, remove or output, not "
                    + (what.isBlank() ? "nothing" : what));
        }
    }

    private LayerCatalog.Entry entry(String key) {
        Optional<LayerCatalog.Entry> e = catalog().entry(key);
        if (e.isPresent()) {
            return e.get();
        }
        // a name the model remembers loosely: one entry whose key ends with it
        String k = key.toLowerCase(Locale.ROOT);
        List<LayerCatalog.Entry> like = catalog().entries().stream()
                .filter(x -> x.key().toLowerCase(Locale.ROOT).endsWith(k) || x.key().toLowerCase(Locale.ROOT).endsWith("." + k)).toList();
        if (like.size() == 1) {
            return like.getFirst();
        }
        throw new IllegalArgumentException("there is no layer " + key + "; use a key from list_layers");
    }

    private static String group(String key) {
        String s = key.replaceFirst("[#(].*$", "");
        s = s.substring(s.lastIndexOf('.') + 1);
        return s.isBlank() ? "block" : s;
    }

    private static boolean multi(Map<String, Object> n) {
        String op = Json.string(n, "op", "");
        return op.equals("add") || op.equals("multiply") || op.equals("concat");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> find(List<Object> nodes, String id) {
        for (Object o : nodes) {
            if (id.equals(((Map<String, Object>) o).get("id"))) {
                return (Map<String, Object>) o;
            }
        }
        return null;
    }

    private static Map<String, Object> need(List<Object> nodes, String id) {
        Map<String, Object> n = find(nodes, id);
        if (n == null) {
            throw new IllegalArgumentException("there is no block " + id);
        }
        return n;
    }

    private static String freeId(List<Object> nodes, String base) {
        String b = base.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "");
        b = b.isEmpty() ? "block" : b.length() > 12 ? b.substring(0, 12) : b;
        for (int i = 1; ; i++) {
            if (find(nodes, b + i) == null) {
                return b + i;
            }
        }
    }

    // to the right of what feeds it, clear of the blocks already there
    @SuppressWarnings("unchecked")
    private static void place(List<Object> nodes, Map<String, Object> node, List<String> inputs) {
        double x = 40;
        double y = 120;
        if (!inputs.isEmpty()) {
            for (String in : inputs) {
                Map<String, Object> s = find(nodes, in);
                x = Math.max(x, num(s.get("x"), 40) + 250);
            }
            y = num(find(nodes, inputs.getFirst()).get("y"), 120);
        } else if (!nodes.isEmpty()) {
            for (Object o : nodes) {
                x = Math.max(x, num(((Map<String, Object>) o).get("x"), 40) + 250);
            }
        }
        Set<String> taken = new HashSet<>();
        for (Object o : nodes) {
            Map<String, Object> m = (Map<String, Object>) o;
            taken.add(Math.round(num(m.get("x"), 0) / 60) + ":" + Math.round(num(m.get("y"), 0) / 60));
        }
        while (taken.contains(Math.round(x / 60) + ":" + Math.round(y / 60))) {
            y += 210;
        }
        node.put("x", (int) x);
        node.put("y", (int) y);
    }

    private static double num(Object v, double otherwise) {
        return v instanceof Number n ? n.doubleValue() : otherwise;
    }

    private static List<String> strings(Object v) {
        List<String> out = new ArrayList<>();
        if (v != null) {
            for (Object o : Json.array(v)) {
                out.add(String.valueOf(o));
            }
        }
        return out;
    }

    // the arguments in order: a list as given, or values by name over the current ones; gaps get a first value
    private static List<Object> args(LayerCatalog.Entry e, Object given, List<Object> current) {
        List<LayerCatalog.Param> ps = e.params();
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < ps.size(); i++) {
            out.add(current != null && i < current.size() ? current.get(i) : guess(ps.get(i)));
        }
        if (given instanceof List<?> l) {
            for (int i = 0; i < Math.min(l.size(), ps.size()); i++) {
                out.set(i, l.get(i));
            }
        } else if (given instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> kv : m.entrySet()) {
                int at = -1;
                for (int i = 0; i < ps.size(); i++) {
                    if (ps.get(i).name().equalsIgnoreCase(String.valueOf(kv.getKey()))) {
                        at = i;
                    }
                }
                if (at < 0) {
                    throw new IllegalArgumentException(group(e.key()) + " has no setting " + kv.getKey() + "; it has "
                            + String.join(", ", ps.stream().map(LayerCatalog.Param::name).toList()));
                }
                out.set(at, kv.getValue());
            }
        } else if (given != null) {
            throw new IllegalArgumentException("args is a list in order, or {name: value}");
        }
        return out;
    }

    private static List<Object> options(LayerCatalog.Entry e, Object given, List<Object> current) {
        List<Object> out = new ArrayList<>(current);
        if (given == null) {
            return out;
        }
        if (!(given instanceof Map<?, ?> m)) {
            throw new IllegalArgumentException("options is {optionKey: [args]}, with null to switch one off");
        }
        for (Map.Entry<?, ?> kv : m.entrySet()) {
            String key = String.valueOf(kv.getKey());
            LayerCatalog.Option o = e.options().stream().filter(x -> x.key().equals(key)).findFirst()
                    .or(() -> e.options().stream().filter(x -> x.name().equals(key)).findFirst())
                    .orElseThrow(() -> new IllegalArgumentException(group(e.key()) + " has no option " + key + "; it has "
                            + String.join(", ", e.options().stream().map(LayerCatalog.Option::key).toList())));
            out.removeIf(x -> o.key().equals(Json.object(x).get("key")));
            if (kv.getValue() != null) {
                List<Object> a = new ArrayList<>();
                List<?> vals = kv.getValue() instanceof List<?> l ? l : List.of(kv.getValue());
                for (int i = 0; i < o.params().size(); i++) {
                    a.add(i < vals.size() ? vals.get(i) : guess(o.params().get(i)));
                }
                out.add(new LinkedHashMap<>(Map.of("key", o.key(), "args", a)));
            }
        }
        return out;
    }

    // a first value that works, as the canvas gives a new block
    static Object guess(LayerCatalog.Param p) {
        String n = p.name().toLowerCase(Locale.ROOT);
        Object v;
        switch (p.kind()) {
            case RANDOM -> {
                return null;
            }
            case CHOICE -> {
                return p.choices().contains("RELU") ? "RELU" : p.choices().contains("HE") ? "HE" : p.choices().getFirst();
            }
            case BOOLEAN -> {
                return false;
            }
            case STRING -> {
                return "";
            }
            case FLOAT, DOUBLE -> v = n.matches(".*(rate|drop|prob|eps|alpha|slope|momentum).*") ? 0.1 : 1;
            default -> {
                if (n.matches(".*(kernel|window).*")) {
                    v = 3;
                } else if (n.contains("head")) {
                    v = 4;
                } else if (n.matches(".*(units|filters|features|dim|channels|hidden|width|out|size|vocab|capacity|experts|slots).*")) {
                    v = 64;
                } else if (n.matches(".*(factor|scale).*")) {
                    v = 2;
                } else {
                    v = 1;
                }
            }
        }
        if (!p.array()) {
            return v;
        }
        return n.matches(".*(kernel|window).*") ? List.of(v, v) : List.of(v);
    }
}
