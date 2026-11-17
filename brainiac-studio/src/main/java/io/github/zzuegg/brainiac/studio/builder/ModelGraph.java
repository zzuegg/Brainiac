package io.github.zzuegg.brainiac.studio.builder;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.Node;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog.Entry;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog.Kind;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog.Option;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog.Param;
import io.github.zzuegg.brainiac.tensor.Shape;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A model drawn in the builder, checked with the core's own shape rules: each block is made
 * from the {@link LayerCatalog} and wired with {@link Node}, so a block that cannot take its
 * input is refused with the core's reason, and a graph that checks out builds as it is.
 *
 * <p>The graph is JSON: {@code {"nodes": [...], "output": "id"}}, where a node is one of
 * <pre>{@code
 * {"id": "in",  "op": "input", "shape": [3, 32, 32]}
 * {"id": "c1",  "op": "layer", "entry": "io.github.zzuegg.brainiac.nn.Conv#of(int,int[])",
 *  "args": [32, [3, 3]], "options": [{"key": "stride(int[])", "args": [[2, 2]]}], "inputs": ["in"]}
 * {"id": "sum", "op": "add", "inputs": ["c1", "c2"]}      // also "multiply" and "concat"
 * }</pre>
 */
public final class ModelGraph {

    /** At most this many blocks in one graph. */
    static final int MAX_NODES = 2000;

    /**
     * What checking a graph found.
     *
     * @param shapes the output shape of every block that checks out
     * @param errors why each other block does not
     * @param input the input block's id, if there is exactly one
     * @param output the output block's id: the one asked for, or the only block nothing reads
     */
    public record Check(Map<String, Shape> shapes, Map<String, String> errors, String input, String output) {

        /** @return true if every block checks out and there is one input and one output */
        public boolean ok() {
            return errors.isEmpty() && input != null && output != null;
        }

        /**
         * Describes the check for the page.
         *
         * @return {@code {ok, nodes: {id: {shape} or {error}}, input, output, problem}}
         */
        public Map<String, Object> toJson() {
            Map<String, Object> nodes = new LinkedHashMap<>();
            shapes.forEach((id, s) -> nodes.put(id, Map.of("shape", dims(s))));
            errors.forEach((id, e) -> nodes.put(id, Map.of("error", e)));
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("ok", ok());
            out.put("nodes", nodes);
            if (input != null) {
                out.put("input", input);
            }
            if (output != null) {
                out.put("output", output);
                if (shapes.containsKey(output)) {
                    out.put("outputShape", dims(shapes.get(output)));
                }
            }
            if (input == null) {
                out.put("problem", "the model needs exactly one Input block");
            } else if (output == null) {
                out.put("problem", "the model needs one output: a block nothing else reads, or one marked as the output");
            }
            return out;
        }
    }

    private final LayerCatalog catalog;

    /**
     * Uses the layers of a catalog.
     *
     * @param catalog the catalog
     */
    public ModelGraph(LayerCatalog catalog) {
        this.catalog = catalog;
    }

    /**
     * Checks every block's shape.
     *
     * @param graph the graph as parsed JSON
     * @return what was found
     */
    public Check check(Map<String, Object> graph) {
        return evaluate(graph, new SplittableRandom(0), new LinkedHashMap<>());
    }

    /**
     * Builds the model with fresh weights.
     *
     * @param graph the graph as parsed JSON
     * @param seed the seed for the initial weights
     * @return the model
     * @throws IllegalArgumentException if the graph does not check out
     */
    public Model build(Map<String, Object> graph, long seed) {
        Map<String, Node> nodes = new LinkedHashMap<>();
        Check c = evaluate(graph, new SplittableRandom(seed), nodes);
        if (!c.ok()) {
            String why = c.errors().isEmpty() ? (String) c.toJson().get("problem")
                    : c.errors().entrySet().iterator().next().getKey() + ": " + c.errors().values().iterator().next();
            throw new IllegalArgumentException("the model does not check out: " + why);
        }
        return Model.graph(nodes.get(c.input()), nodes.get(c.output())).seed(seed).build();
    }

    private Check evaluate(Map<String, Object> graph, SplittableRandom random, Map<String, Node> nodes) {
        List<Object> list = Json.array(graph.get("nodes"));
        if (list.size() > MAX_NODES) {
            throw new IllegalArgumentException("at most " + MAX_NODES + " blocks");
        }
        Map<String, Map<String, Object>> byId = new LinkedHashMap<>();
        for (Object o : list) {
            Map<String, Object> n = Json.object(o);
            if (!(n.get("id") instanceof String id) || id.isEmpty() || id.length() > 64) {
                throw new IllegalArgumentException("every block needs an id of 1 to 64 characters");
            }
            if (byId.put(id, n) != null) {
                throw new IllegalArgumentException("two blocks are called " + id);
            }
        }
        Map<String, Shape> shapes = new LinkedHashMap<>();
        Map<String, String> errors = new LinkedHashMap<>();
        List<String> inputs = new ArrayList<>();
        boolean progress = true;
        while (progress) {
            progress = false;
            for (Map.Entry<String, Map<String, Object>> e : byId.entrySet()) {
                String id = e.getKey();
                if (nodes.containsKey(id) || errors.containsKey(id)) {
                    continue;
                }
                Map<String, Object> n = e.getValue();
                List<String> from = ids(n.get("inputs"));
                String waiting = null;
                boolean ready = true;
                for (String f : from) {
                    if (!byId.containsKey(f)) {
                        waiting = "reads " + f + ", which is not in the graph";
                    } else if (errors.containsKey(f)) {
                        waiting = "waits for " + f + ", which does not check out";
                    } else if (!nodes.containsKey(f)) {
                        ready = false;
                    }
                }
                if (waiting != null) {
                    errors.put(id, waiting);
                    progress = true;
                    continue;
                }
                if (!ready) {
                    continue;
                }
                try {
                    Node made = make(n, from.stream().map(nodes::get).toList(), random);
                    nodes.put(id, made);
                    shapes.put(id, made.shape());
                    if ("input".equals(n.get("op"))) {
                        inputs.add(id);
                    }
                } catch (IllegalArgumentException | IllegalStateException | ArithmeticException bad) {
                    errors.put(id, message(bad));
                }
                progress = true;
            }
        }
        for (String id : byId.keySet()) {
            if (!nodes.containsKey(id) && !errors.containsKey(id)) {
                errors.put(id, "is part of a loop; a model's blocks only feed forward");
            }
        }
        String output = graph.get("output") instanceof String o && byId.containsKey(o) ? o : null;
        if (output == null) {
            List<String> sinks = new ArrayList<>(byId.keySet());
            byId.values().forEach(n -> sinks.removeAll(ids(n.get("inputs"))));
            sinks.removeAll(inputs);
            output = sinks.size() == 1 ? sinks.getFirst() : null;
        }
        return new Check(shapes, errors, inputs.size() == 1 ? inputs.getFirst() : null, output);
    }

    private Node make(Map<String, Object> n, List<Node> from, SplittableRandom random) {
        String op = String.valueOf(n.get("op"));
        switch (op) {
            case "input" -> {
                if (!from.isEmpty()) {
                    throw new IllegalArgumentException("an Input block reads nothing");
                }
                List<Object> dims = Json.array(n.get("shape"));
                return Node.input(dims.stream().mapToInt(d -> integer(d, "a dimension")).toArray());
            }
            case "add", "multiply", "concat" -> {
                if (from.size() < 2) {
                    throw new IllegalArgumentException(op + " needs at least two inputs");
                }
                Node out = from.getFirst();
                if (op.equals("concat")) {
                    return out.concat(from.subList(1, from.size()).toArray(Node[]::new));
                }
                for (Node other : from.subList(1, from.size())) {
                    out = op.equals("add") ? out.add(other) : out.multiply(other);
                }
                return out;
            }
            case "layer" -> {
                if (from.isEmpty()) {
                    throw new IllegalArgumentException("is not connected to an input yet");
                }
                Entry entry = catalog.entry(String.valueOf(n.get("entry")))
                        .orElseThrow(() -> new IllegalArgumentException("no layer " + n.get("entry")));
                return attach(entry, make(entry, n.get("args"), n.get("options"), random), from);
            }
            default -> throw new IllegalArgumentException("unknown block type " + op);
        }
    }

    /**
     * Makes what a catalog entry describes, with its options applied in order.
     *
     * @param entry the factory, constructor or constant
     * @param args its arguments, a list aligned with its parameters
     * @param options a list of {@code {"key": ..., "args": [...]}}, or null
     * @param random fills in random sources
     * @return the made object
     */
    public static Object make(Entry entry, Object args, Object options, SplittableRandom random) {
        Object made = create(entry, args, random);
        for (Object o : options == null ? List.of() : Json.array(options)) {
            Map<String, Object> opt = Json.object(o);
            Option option = entry.option(String.valueOf(opt.get("key")))
                    .orElseThrow(() -> new IllegalArgumentException(entry.group() + " has no option " + opt.get("key")));
            made = invoke(option.method(), made, values(option.params(), opt.get("args"), random));
        }
        return made;
    }

    private static Node attach(Entry entry, Object made, List<Node> from) {
        Node first = from.getFirst();
        if (from.size() > 1) {
            if (made instanceof MultiInputLayer.Spec spec) {
                return first.merge(spec, from.subList(1, from.size()).toArray(Node[]::new));
            }
            throw new IllegalArgumentException(entry.group() + " takes one input; join several with Add or Concat first");
        }
        if (made instanceof LayerSpec spec) {
            return first.add(spec);
        }
        if (made instanceof MultiInputLayer.Spec) {
            throw new IllegalArgumentException(entry.group() + " needs at least two inputs");
        }
        if (made instanceof MultiInputLayer) {
            throw new IllegalArgumentException(entry.group() + " takes several inputs, which the builder cannot wire yet");
        }
        return first.add(LayerSpec.of((Layer) made));
    }

    private static Object create(Entry entry, Object args, SplittableRandom random) {
        try {
            return switch (entry.member()) {
                case Method m -> invoke(m, null, values(entry.params(), args, random));
                case Constructor<?> k -> k.newInstance(values(entry.params(), args, random));
                case Field f -> f.get(null);
                default -> throw new IllegalStateException("cannot make " + entry.key());
            };
        } catch (InvocationTargetException e) {
            throw rethrow(e);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot make " + entry.key() + ": " + e, e);
        }
    }

    private static Object invoke(Method m, Object target, Object[] args) {
        try {
            return m.invoke(target, args);
        } catch (InvocationTargetException e) {
            throw rethrow(e);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot call " + m, e);
        }
    }

    private static RuntimeException rethrow(InvocationTargetException e) {
        Throwable cause = e.getCause();
        if (cause instanceof IllegalArgumentException || cause instanceof IllegalStateException
                || cause instanceof ArithmeticException || cause instanceof NullPointerException) {
            return new IllegalArgumentException(message(cause), cause);
        }
        return cause instanceof RuntimeException r ? r : new IllegalStateException(cause);
    }

    private static Object[] values(List<Param> params, Object args, SplittableRandom random) {
        List<Object> given = args == null ? List.of() : Json.array(args);
        Object[] out = new Object[params.size()];
        int at = 0;
        for (int i = 0; i < params.size(); i++) {
            Param p = params.get(i);
            if (p.kind() == Kind.RANDOM) {
                out[i] = p.type() == java.util.Random.class ? new java.util.Random(random.nextLong()) : random.split();
                if (given.size() == params.size()) {
                    at++;   // the page may send a placeholder for it
                }
                continue;
            }
            if (at >= given.size()) {
                throw new IllegalArgumentException(p.name() + " is not set");
            }
            out[i] = value(p, given.get(at++));
        }
        return out;
    }

    private static Object value(Param p, Object v) {
        if (v == null) {
            throw new IllegalArgumentException(p.name() + " is not set");
        }
        if (!p.array()) {
            return one(p, p.type(), v);
        }
        List<Object> items = v instanceof List<?> l ? new ArrayList<>(l) : List.of(v);
        Class<?> c = p.type().getComponentType();
        Object arr = Array.newInstance(c, items.size());
        for (int i = 0; i < items.size(); i++) {
            Array.set(arr, i, one(p, c, items.get(i)));
        }
        return arr;
    }

    private static Object one(Param p, Class<?> t, Object v) {
        return switch (p.kind()) {
            case INT -> {
                int i = integer(v, p.name());
                yield t == short.class ? (Object) (short) i : t == byte.class ? (Object) (byte) i : (Object) i;
            }
            case LONG -> (long) number(v, p.name());
            case FLOAT -> (float) number(v, p.name());
            case DOUBLE -> number(v, p.name());
            case BOOLEAN -> {
                if (!(v instanceof Boolean b)) {
                    throw new IllegalArgumentException(p.name() + " is true or false");
                }
                yield b;
            }
            case STRING -> String.valueOf(v);
            case CHOICE -> t.isEnum() ? enumValue(t, String.valueOf(v)) : LayerCatalog.constant(t, String.valueOf(v));
            case RANDOM -> throw new IllegalStateException("filled in by the studio");
        };
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Object enumValue(Class<?> t, String name) {
        try {
            return Enum.valueOf((Class) t, name);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(t.getSimpleName() + " has no " + name + "; it has "
                    + String.join(", ", LayerCatalog.choices(t)));
        }
    }

    private static double number(Object v, String what) {
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        if (v instanceof String s) {
            try {
                return Double.parseDouble(s.strip());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(what + " is a number, not " + s);
            }
        }
        throw new IllegalArgumentException(what + " is a number");
    }

    private static int integer(Object v, String what) {
        double d = number(v, what);
        if (d != Math.rint(d) || Math.abs(d) > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " is a whole number");
        }
        return (int) d;
    }

    private static List<String> ids(Object o) {
        if (o == null) {
            return List.of();
        }
        return Json.array(o).stream().map(String::valueOf).toList();
    }

    private static String message(Throwable t) {
        String m = t.getMessage();
        return m == null || m.isBlank() ? t.getClass().getSimpleName() : m;
    }

    private static List<Integer> dims(Shape s) {
        List<Integer> out = new ArrayList<>();
        for (int d : s.toArray()) {
            out.add(d);
        }
        return out;
    }
}
