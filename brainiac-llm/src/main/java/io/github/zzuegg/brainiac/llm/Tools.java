package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.AnnotatedElement;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;

/**
 * Makes {@linkplain Agent.Tool tools} from ordinary Java methods and records, with the JSON
 * schema the model sees derived from the parameter types, and the model's JSON arguments
 * converted back to those types.
 *
 * <pre>{@code
 * class Shop {
 *     @Tools.Tool("Looks up the price of a product, in euros")
 *     double price(@Tools.Param("the product's name") String product, Optional<Integer> quantity) { ... }
 *
 *     @Tools.Tool("Places an order and returns its number")
 *     Order order(Order.Request request) { ... }       // records become nested objects
 * }
 * Agent agent = new Agent(generator, tokenizer, Tools.from(new Shop()), options);
 *
 * // or one tool whose arguments are a record:
 * record Add(@Tools.Param("first addend") long a, long b) {}
 * Agent.Tool add = Tools.of("add", "Adds two numbers", Add.class, x -> x.a() + x.b());
 * }</pre>
 *
 * <p>Types: strings, booleans, every primitive number type and its box, enums (by constant
 * name), records, lists, sets, arrays and string-keyed maps of these, and {@code Optional} for
 * an argument the model may leave out (also nullable boxes marked
 * {@linkplain Param#optional optional}). Results: a string as it is, {@code void} or
 * {@code null} as {@code "ok"}, anything else as JSON. An exception becomes the tool's error
 * text, which the model reads and can correct.
 *
 * <p>Methods and records are reached by reflection: make them public in an exported package,
 * or open the package to this module. Parameter names come from {@link Param#name}, else from
 * the compiled names ({@code javac -parameters}).
 */
public final class Tools {

    private Tools() {
    }

    /** Marks a method as a tool. */
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    public @interface Tool {

        /** @return what the tool does, for the model */
        String value();

        /** @return the tool's name; empty for the method's name */
        String name() default "";
    }

    /** Describes a tool's parameter or a record component. */
    @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.PARAMETER, ElementType.RECORD_COMPONENT, ElementType.FIELD})
    public @interface Param {

        /** @return what the value means, for the model */
        String value() default "";

        /** @return the name the model uses; empty for the Java name */
        String name() default "";

        /** @return whether the model may leave it out (then {@code null}, or empty for {@code Optional}) */
        boolean optional() default false;
    }

    /**
     * The tools of an object's methods marked {@link Tool}, or of a class's static ones.
     *
     * @param target an object, or a {@link Class} for its static methods
     * @return one tool per marked method, ordered by name
     * @throws IllegalArgumentException if none is marked or a type cannot be described
     */
    public static List<Agent.Tool> from(Object target) {
        Objects.requireNonNull(target, "target");
        Class<?> type = target instanceof Class<?> c ? c : target.getClass();
        Object receiver = target instanceof Class<?> ? null : target;
        List<Method> methods = new ArrayList<>();
        for (Method m : type.getMethods()) {
            if (m.isAnnotationPresent(Tool.class) && (receiver != null || Modifier.isStatic(m.getModifiers()))) {
                methods.add(m);
            }
        }
        for (Method m : type.getDeclaredMethods()) {
            if (m.isAnnotationPresent(Tool.class) && !methods.contains(m)
                    && (receiver != null || Modifier.isStatic(m.getModifiers()))) {
                methods.add(m);
            }
        }
        if (methods.isEmpty()) {
            throw new IllegalArgumentException(type.getName() + " has no method marked @Tools.Tool");
        }
        methods.sort(java.util.Comparator.comparing(Tools::toolName));
        List<Agent.Tool> tools = new ArrayList<>();
        for (Method m : methods) {
            tools.add(tool(m, receiver));
        }
        return tools;
    }

    /**
     * A tool whose arguments are the components of a record.
     *
     * @param name the tool's name
     * @param description what it does, for the model
     * @param arguments the record type of its arguments
     * @param run computes the result
     * @param <R> the record type
     * @return the tool
     */
    public static <R extends Record> Agent.Tool of(String name, String description, Class<R> arguments,
            Function<? super R, ?> run) {
        Objects.requireNonNull(run, "run");
        Map<String, Object> schema = schema(arguments);
        return new Agent.Tool(name, description, schema, args -> result(run.apply(record(arguments, args, ""))));
    }

    private static String toolName(Method m) {
        String n = m.getAnnotation(Tool.class).name();
        return n.isEmpty() ? m.getName() : n;
    }

    private static Agent.Tool tool(Method m, Object receiver) {
        m.trySetAccessible();
        Parameter[] params = m.getParameters();
        Map<String, Object> properties = new LinkedHashMap<>();
        List<String> required = new ArrayList<>();
        String[] names = new String[params.length];
        for (int i = 0; i < params.length; i++) {
            Parameter p = params[i];
            names[i] = name(p, p.getName());
            properties.put(names[i], describe(p.getParameterizedType(), p));
            if (!optional(p.getParameterizedType(), p)) {
                required.add(names[i]);
            }
        }
        Map<String, Object> schema = object(properties, required);
        Tool t = m.getAnnotation(Tool.class);
        return new Agent.Tool(toolName(m), t.value(), schema, args -> {
            Object[] values = new Object[params.length];
            for (int i = 0; i < params.length; i++) {
                values[i] = argument(args, names[i], params[i].getParameterizedType(), params[i], names[i]);
            }
            try {
                Object out = m.invoke(receiver, values);
                return m.getReturnType() == void.class ? "ok" : result(out);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException("cannot call " + m + ": make it public or open its package", e);
            } catch (InvocationTargetException e) {
                Throwable cause = e.getCause();
                if (cause instanceof RuntimeException r) {
                    throw r;
                }
                throw new RuntimeException(cause.getMessage(), cause);
            }
        });
    }

    // ---------------------------------------------------------------- schemas

    /**
     * The JSON schema of a Java type, as tools describe their arguments.
     *
     * @param type a supported type (see the class description)
     * @return the schema, as nested maps and lists
     * @throws IllegalArgumentException for an unsupported type
     */
    public static Map<String, Object> schema(Type type) {
        return describe(type, (AnnotatedElement) null);
    }

    private static Map<String, Object> describe(Type type, AnnotatedElement where) {
        Map<String, Object> s = describeType(type, new LinkedHashSet<>());
        Param p = where == null ? null : where.getAnnotation(Param.class);
        if (p != null && !p.value().isEmpty()) {
            s = new LinkedHashMap<>(s);
            s.put("description", p.value());
        }
        return s;
    }

    private static Map<String, Object> describeType(Type type, Set<Type> open) {
        Class<?> raw = raw(type);
        Map<String, Object> s = new LinkedHashMap<>();
        if (raw == String.class || raw == char.class || raw == Character.class) {
            s.put("type", "string");
        } else if (raw == boolean.class || raw == Boolean.class) {
            s.put("type", "boolean");
        } else if (raw == int.class || raw == long.class || raw == short.class || raw == byte.class
                || raw == Integer.class || raw == Long.class || raw == Short.class || raw == Byte.class
                || raw == java.math.BigInteger.class) {
            s.put("type", "integer");
        } else if (raw == double.class || raw == float.class || raw == Double.class || raw == Float.class
                || raw == java.math.BigDecimal.class) {
            s.put("type", "number");
        } else if (raw.isEnum()) {
            s.put("type", "string");
            List<String> names = new ArrayList<>();
            for (Object c : raw.getEnumConstants()) {
                names.add(((Enum<?>) c).name());
            }
            s.put("enum", names);
        } else if (raw == Optional.class) {
            return describeType(typeArgument(type, 0), open);
        } else if (raw.isArray() || Collection.class.isAssignableFrom(raw)) {
            s.put("type", "array");
            s.put("items", describeType(element(type), open));
            if (Set.class.isAssignableFrom(raw)) {
                s.put("uniqueItems", true);
            }
        } else if (Map.class.isAssignableFrom(raw)) {
            if (raw(typeArgument(type, 0)) != String.class) {
                throw new IllegalArgumentException("map keys must be strings: " + type);
            }
            s.put("type", "object");
            s.put("additionalProperties", describeType(typeArgument(type, 1), open));
        } else if (raw.isRecord()) {
            if (!open.add(raw)) {
                throw new IllegalArgumentException("a record that contains itself cannot be described: " + raw);
            }
            Map<String, Object> properties = new LinkedHashMap<>();
            List<String> required = new ArrayList<>();
            for (RecordComponent c : raw.getRecordComponents()) {
                String n = name(c, c.getName());
                Map<String, Object> d = describeType(c.getGenericType(), open);
                Param p = c.getAnnotation(Param.class);
                if (p != null && !p.value().isEmpty()) {
                    d = new LinkedHashMap<>(d);
                    d.put("description", p.value());
                }
                properties.put(n, d);
                if (!optional(c.getGenericType(), c)) {
                    required.add(n);
                }
            }
            open.remove(raw);
            return object(properties, required);
        } else if (raw == Object.class) {
            return s;   // anything
        } else {
            throw new IllegalArgumentException("cannot describe " + type + " as JSON; use a record, enum, list or map");
        }
        return s;
    }

    private static Map<String, Object> object(Map<String, Object> properties, List<String> required) {
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("type", "object");
        s.put("properties", properties);
        if (!required.isEmpty()) {
            s.put("required", required);
        }
        return s;
    }

    private static String name(AnnotatedElement e, String javaName) {
        Param p = e.getAnnotation(Param.class);
        return p != null && !p.name().isEmpty() ? p.name() : javaName;
    }

    private static boolean optional(Type type, AnnotatedElement e) {
        Param p = e.getAnnotation(Param.class);
        return raw(type) == Optional.class || (p != null && p.optional());
    }

    // ---------------------------------------------------------------- arguments

    private static Object argument(Map<String, Object> args, String name, Type type, AnnotatedElement where,
            String path) {
        Object v = args == null ? null : args.get(name);
        if (v == null) {
            if (raw(type) == Optional.class) {
                return Optional.empty();
            }
            if (optional(type, where) && !raw(type).isPrimitive()) {
                return null;
            }
            throw new IllegalArgumentException("missing argument " + path);
        }
        return convert(v, type, path);
    }

    private static Object convert(Object v, Type type, String path) {
        Class<?> raw = raw(type);
        if (raw == Optional.class) {
            return v == null ? Optional.empty() : Optional.of(convert(v, typeArgument(type, 0), path));
        }
        if (v == null) {
            if (raw.isPrimitive()) {
                throw new IllegalArgumentException(path + " must not be null");
            }
            return null;
        }
        if (raw == Object.class) {
            return v;
        }
        if (raw == String.class) {
            return v instanceof String s ? s : Json.write(v);
        }
        if (raw == char.class || raw == Character.class) {
            String s = String.valueOf(v);
            if (s.length() != 1) {
                throw new IllegalArgumentException(path + " must be one character, got " + Json.write(v));
            }
            return s.charAt(0);
        }
        if (raw == boolean.class || raw == Boolean.class) {
            if (v instanceof Boolean b) {
                return b;
            }
            if ("true".equals(v) || "false".equals(v)) {
                return Boolean.parseBoolean((String) v);
            }
            throw new IllegalArgumentException(path + " must be true or false, got " + Json.write(v));
        }
        if (raw.isPrimitive() || Number.class.isAssignableFrom(raw)) {
            return number(v, raw, path);
        }
        if (raw.isEnum()) {
            String s = String.valueOf(v);
            for (Object c : raw.getEnumConstants()) {
                if (((Enum<?>) c).name().equalsIgnoreCase(s)) {
                    return c;
                }
            }
            throw new IllegalArgumentException(path + " must be one of " + describeType(raw, new LinkedHashSet<>()).get("enum")
                    + ", got " + Json.write(v));
        }
        if (raw.isArray() || Collection.class.isAssignableFrom(raw)) {
            if (!(v instanceof List<?> list)) {
                throw new IllegalArgumentException(path + " must be an array, got " + Json.write(v));
            }
            Type element = element(type);
            List<Object> items = new ArrayList<>(list.size());
            for (int i = 0; i < list.size(); i++) {
                items.add(convert(list.get(i), element, path + "[" + i + "]"));
            }
            if (raw.isArray()) {
                Object array = Array.newInstance(raw.getComponentType(), items.size());
                for (int i = 0; i < items.size(); i++) {
                    Array.set(array, i, items.get(i));
                }
                return array;
            }
            return Set.class.isAssignableFrom(raw) ? new LinkedHashSet<>(items) : items;
        }
        if (Map.class.isAssignableFrom(raw)) {
            Map<String, Object> m = object(v, path);
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : m.entrySet()) {
                out.put(e.getKey(), convert(e.getValue(), typeArgument(type, 1), path + "." + e.getKey()));
            }
            return out;
        }
        if (raw.isRecord()) {
            return record(raw, object(v, path), path);
        }
        throw new IllegalArgumentException("cannot convert JSON to " + type);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> object(Object v, String path) {
        if (v instanceof String s) {   // some models quote nested objects
            try {
                v = Json.parse(s);
            } catch (RuntimeException e) {
                // reported below
            }
        }
        if (!(v instanceof Map<?, ?> m)) {
            throw new IllegalArgumentException((path.isEmpty() ? "the arguments" : path) + " must be an object, got "
                    + Json.write(v));
        }
        return (Map<String, Object>) m;
    }

    private static <R> R record(Class<R> type, Map<String, Object> args, String path) {
        RecordComponent[] parts = type.getRecordComponents();
        Class<?>[] types = new Class<?>[parts.length];
        Object[] values = new Object[parts.length];
        for (int i = 0; i < parts.length; i++) {
            types[i] = parts[i].getType();
            String n = name(parts[i], parts[i].getName());
            values[i] = argument(args, n, parts[i].getGenericType(), parts[i], path.isEmpty() ? n : path + "." + n);
        }
        try {
            Constructor<R> c = type.getDeclaredConstructor(types);
            c.trySetAccessible();
            return c.newInstance(values);
        } catch (InvocationTargetException e) {
            Throwable cause = e.getCause();
            throw cause instanceof RuntimeException r ? r : new RuntimeException(cause.getMessage(), cause);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot create " + type.getName() + ": make it public or open its package", e);
        }
    }

    private static Object number(Object v, Class<?> raw, String path) {
        Number n;
        if (v instanceof Number x) {
            n = x;
        } else if (v instanceof String s) {
            try {
                n = new java.math.BigDecimal(s.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(path + " must be a number, got " + Json.write(v));
            }
        } else {
            throw new IllegalArgumentException(path + " must be a number, got " + Json.write(v));
        }
        double d = n.doubleValue();
        boolean whole = d == Math.rint(d) && !Double.isInfinite(d);
        if (raw == double.class || raw == Double.class) {
            return d;
        }
        if (raw == float.class || raw == Float.class) {
            return (float) d;
        }
        if (raw == java.math.BigDecimal.class) {
            return n instanceof java.math.BigDecimal b ? b : new java.math.BigDecimal(n.toString());
        }
        if (!whole) {
            throw new IllegalArgumentException(path + " must be a whole number, got " + Json.write(v));
        }
        if (raw == java.math.BigInteger.class) {
            return new java.math.BigDecimal(n.toString()).toBigIntegerExact();
        }
        long l = n instanceof Long x ? x : (long) d;
        long min = raw == int.class || raw == Integer.class ? Integer.MIN_VALUE
                : raw == short.class || raw == Short.class ? Short.MIN_VALUE
                : raw == byte.class || raw == Byte.class ? Byte.MIN_VALUE : Long.MIN_VALUE;
        long max = raw == int.class || raw == Integer.class ? Integer.MAX_VALUE
                : raw == short.class || raw == Short.class ? Short.MAX_VALUE
                : raw == byte.class || raw == Byte.class ? Byte.MAX_VALUE : Long.MAX_VALUE;
        if (l < min || l > max) {
            throw new IllegalArgumentException(path + " is out of range: " + Json.write(v));
        }
        if (raw == int.class || raw == Integer.class) {
            return (int) l;
        }
        if (raw == short.class || raw == Short.class) {
            return (short) l;
        }
        if (raw == byte.class || raw == Byte.class) {
            return (byte) l;
        }
        return l;
    }

    // ---------------------------------------------------------------- results

    /**
     * The text a tool returns to the model for a Java value: a string as it is, {@code null}
     * as {@code "ok"}, anything else as JSON (records as objects, enums by name).
     *
     * @param value the value
     * @return the text
     */
    public static String result(Object value) {
        if (value == null) {
            return "ok";
        }
        if (value instanceof CharSequence s) {
            return s.toString();
        }
        return Json.write(jsonable(value));
    }

    private static Object jsonable(Object v) {
        return switch (v) {
            case null -> null;
            case String s -> s;
            case Boolean b -> b;
            case Character c -> String.valueOf(c);
            case java.math.BigDecimal b -> b.doubleValue();
            case java.math.BigInteger b -> b.longValue();
            case Number n -> n;
            case Enum<?> e -> e.name();
            case Optional<?> o -> o.map(Tools::jsonable).orElse(null);
            case Map<?, ?> m -> {
                Map<String, Object> out = new LinkedHashMap<>();
                m.forEach((k, x) -> out.put(String.valueOf(k), jsonable(x)));
                yield out;
            }
            case Iterable<?> it -> {
                List<Object> out = new ArrayList<>();
                it.forEach(x -> out.add(jsonable(x)));
                yield out;
            }
            case Record r -> {
                Map<String, Object> out = new LinkedHashMap<>();
                for (RecordComponent c : r.getClass().getRecordComponents()) {
                    try {
                        Method a = c.getAccessor();
                        a.trySetAccessible();
                        out.put(name(c, c.getName()), jsonable(a.invoke(r)));
                    } catch (ReflectiveOperationException e) {
                        throw new IllegalStateException("cannot read " + c, e);
                    }
                }
                yield out;
            }
            default -> {
                if (v.getClass().isArray()) {
                    List<Object> out = new ArrayList<>();
                    for (int i = 0; i < Array.getLength(v); i++) {
                        out.add(jsonable(Array.get(v, i)));
                    }
                    yield out;
                }
                yield v.toString();
            }
        };
    }

    // ---------------------------------------------------------------- types

    private static Class<?> raw(Type t) {
        return switch (t) {
            case Class<?> c -> c;
            case ParameterizedType p -> (Class<?>) p.getRawType();
            case GenericArrayType g -> Array.newInstance(raw(g.getGenericComponentType()), 0).getClass();
            default -> Object.class;   // wildcards and type variables: anything
        };
    }

    private static Type typeArgument(Type t, int i) {
        if (t instanceof ParameterizedType p) {
            Type a = p.getActualTypeArguments()[i];
            return a instanceof java.lang.reflect.WildcardType w ? w.getUpperBounds()[0] : a;
        }
        return Object.class;
    }

    private static Type element(Type t) {
        if (t instanceof GenericArrayType g) {
            return g.getGenericComponentType();
        }
        Class<?> raw = raw(t);
        return raw.isArray() ? raw.getComponentType() : typeArgument(t, 0);
    }
}
