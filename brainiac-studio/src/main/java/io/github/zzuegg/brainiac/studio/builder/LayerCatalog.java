package io.github.zzuegg.brainiac.studio.builder;

import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.module.ModuleReader;
import java.lang.module.ModuleReference;
import java.lang.module.ResolvedModule;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.TreeSet;
import java.util.jar.JarEntry;
import java.util.function.Function;
import java.util.jar.JarFile;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

/**
 * Every layer the installed Brainiac modules offer, found by looking at their public API
 * instead of from a hand-written list: public static factories, constructors and constants
 * that produce a {@link LayerSpec}, a {@link Layer} or a {@link MultiInputLayer.Spec}, and the
 * chainable options on what they produce (such as {@code Conv.of(32, 3, 3).stride(2)}).
 *
 * <p>Only methods whose every parameter the builder can fill in are offered: numbers, flags,
 * text, arrays of those, enums, and types with named constants such as {@code Activation.RELU}
 * or {@code Padding.SAME}. A random source is filled in by the studio. Parameter names come from
 * the class files when they were compiled with {@code -parameters}.
 */
public final class LayerCatalog {

    private static final String PREFIX = "io.github.zzuegg.brainiac";

    /** What a value in the builder is. */
    public enum Kind { INT, LONG, FLOAT, DOUBLE, BOOLEAN, STRING, CHOICE, RANDOM }

    /**
     * One parameter.
     *
     * @param name its name, or {@code argN} when the class was compiled without names
     * @param kind what it holds
     * @param array true for an array or varargs
     * @param choices the names to pick from for {@link Kind#CHOICE}
     * @param type the Java type, for showing
     */
    public record Param(String name, Kind kind, boolean array, List<String> choices, Class<?> type) {
    }

    /**
     * One way to change a layer after it is made, such as {@code stride(int...)}.
     *
     * @param key its name and parameter types, unique within the entry
     * @param name the method name
     * @param params what it takes
     * @param method the method
     */
    public record Option(String key, String name, List<Param> params, Method method) {
    }

    /**
     * One way to make a layer.
     *
     * @param key the class and member, such as {@code io.github.zzuegg.brainiac.nn.Conv#of(int,int[])}
     * @param group the class it is in, such as {@code Conv}
     * @param name the member, such as {@code of}, {@code new} or {@code RELU}
     * @param module the module it comes from
     * @param produces {@code spec}, {@code layer} or {@code multi} (takes several inputs)
     * @param multiInput true if it can take several inputs
     * @param params what it takes
     * @param options the options of what it makes
     * @param member the factory method, constructor or constant
     */
    public record Entry(String key, String group, String name, String module, String produces, boolean multiInput,
            List<Param> params, List<Option> options, Member member) {

        /**
         * Finds an option.
         *
         * @param key its key
         * @return the option
         */
        public Optional<Option> option(String key) {
            return options.stream().filter(o -> o.key().equals(key)).findFirst();
        }
    }

    private final Map<String, Entry> entries;
    private final List<String> skipped;

    private LayerCatalog(Map<String, Entry> entries, List<String> skipped) {
        this.entries = entries;
        this.skipped = skipped;
    }

    /**
     * Looks through the Brainiac modules that are present for layers.
     *
     * @return the catalog
     */
    public static LayerCatalog discover() {
        return discover(LayerCatalog::produces);
    }

    /**
     * Looks through the Brainiac modules that are present for anything a function recognizes,
     * such as optimizers or losses, the same way as for layers.
     *
     * @param kinds says what a type makes, as a short word, or null if it is not wanted
     * @return the catalog
     */
    public static LayerCatalog discover(Function<Class<?>, String> kinds) {
        Scan scan = scan();
        return of(scan.classes(), scan.moduleOf(), kinds);
    }

    private record Scan(List<Class<?>> classes, Map<String, String> moduleOf) {
    }

    private static volatile Scan scanned;

    private static Scan scan() {
        Scan s = scanned;
        if (s != null) {
            return s;
        }
        Set<String> names = new TreeSet<>();
        Map<String, String> moduleOf = new LinkedHashMap<>();
        for (Module m : ModuleLayer.boot().modules()) {
            if (m.getName().startsWith(PREFIX)) {
                ModuleLayer.boot().configuration().findModule(m.getName()).map(ResolvedModule::reference)
                        .ifPresent(ref -> classesIn(ref, m, names, moduleOf));
            }
        }
        if (!LayerSpec.class.getModule().isNamed()) {
            // on the class path (as in some test runs): look in every Brainiac folder or jar there
            for (String entry : System.getProperty("java.class.path", "").split(File.pathSeparator)) {
                if (entry.contains("brainiac") && !entry.contains("brainiac-studio")) {
                    classesIn(Path.of(entry), names, moduleOf);
                }
            }
        }
        s = new Scan(names.stream().<Class<?>>map(LayerCatalog::load).filter(c -> c != null).toList(), Map.copyOf(moduleOf));
        scanned = s;
        return s;
    }

    /**
     * Builds a catalog from given classes, for tests.
     *
     * @param classes the classes to look at
     * @param moduleOf the module each class comes from, by class name
     * @param kinds says what a type makes, or null if it is not wanted
     * @return the catalog
     */
    static LayerCatalog of(List<Class<?>> classes, Map<String, String> moduleOf, Function<Class<?>, String> kinds) {
        Map<String, Entry> out = new LinkedHashMap<>();
        List<String> skipped = new ArrayList<>();
        for (Class<?> c : classes) {
            if (!Modifier.isPublic(c.getModifiers()) || c.isAnonymousClass() || c.isSynthetic()
                    || !c.getModule().isExported(c.getPackageName())) {
                continue;
            }
            String module = moduleOf.getOrDefault(c.getName(), c.getModule().isNamed() ? c.getModule().getName() : "");
            List<Executable> makers = new ArrayList<>();
            for (Method m : c.getDeclaredMethods()) {
                if (Modifier.isPublic(m.getModifiers()) && Modifier.isStatic(m.getModifiers()) && !m.isSynthetic()
                        && kinds.apply(m.getReturnType()) != null) {
                    makers.add(m);
                }
            }
            if (!Modifier.isAbstract(c.getModifiers()) && kinds.apply(c) != null) {
                for (Constructor<?> k : c.getConstructors()) {
                    makers.add(k);
                }
            }
            for (Executable e : makers) {
                Class<?> made = e instanceof Method m ? m.getReturnType() : c;
                List<Param> params = params(e);
                String key = c.getName() + "#" + (e instanceof Method ? e.getName() : "new") + signature(e);
                if (params == null) {
                    skipped.add(key);
                    continue;
                }
                out.put(key, new Entry(key, group(c), e instanceof Method ? e.getName() : "new", module,
                        kinds.apply(made),
                        MultiInputLayer.Spec.class.isAssignableFrom(made), params, options(made), (Member) e));
            }
            for (Field f : c.getDeclaredFields()) {
                int mod = f.getModifiers();
                if (Modifier.isPublic(mod) && Modifier.isStatic(mod) && Modifier.isFinal(mod)
                        && kinds.apply(f.getType()) != null) {
                    String key = c.getName() + "#" + f.getName();
                    out.put(key, new Entry(key, group(c), f.getName(), module, kinds.apply(f.getType()),
                            MultiInputLayer.Spec.class.isAssignableFrom(f.getType()), List.of(), options(f.getType()), f));
                }
            }
        }
        Map<String, Entry> sorted = new LinkedHashMap<>();
        out.values().stream().sorted(Comparator.comparing(Entry::group).thenComparing(Entry::name).thenComparing(Entry::key))
                .forEach(e -> sorted.put(e.key(), e));
        return new LayerCatalog(sorted, List.copyOf(skipped));
    }

    /**
     * Returns every entry, grouped by class.
     *
     * @return the entries
     */
    public List<Entry> entries() {
        return List.copyOf(entries.values());
    }

    /**
     * Finds an entry.
     *
     * @param key its key
     * @return the entry
     */
    public Optional<Entry> entry(String key) {
        return Optional.ofNullable(entries.get(key));
    }

    /**
     * Returns the factories that were left out because the builder cannot fill in one of
     * their parameters (such as a tensor of weights).
     *
     * @return their keys
     */
    public List<String> skipped() {
        return skipped;
    }

    /**
     * Describes the catalog for the page.
     *
     * @return {@code {entries: [...], skipped: n}}
     */
    public Map<String, Object> toJson() {
        List<Object> list = new ArrayList<>();
        for (Entry e : entries.values()) {
            list.add(describe(e));
        }
        return Map.of("entries", list, "skipped", skipped.size());
    }

    /**
     * Describes one entry for the page.
     *
     * @param e the entry
     * @return its key, group, name, module, parameters and options
     */
    public static Map<String, Object> describe(Entry e) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("key", e.key());
        m.put("group", e.group());
        m.put("name", e.name());
        m.put("module", e.module());
        m.put("experimental", e.module().endsWith(".experiments") || e.key().contains(".experimental.")
                || e.key().contains(".experiments."));
        m.put("produces", e.produces());
        m.put("multiInput", e.multiInput());
        m.put("params", e.params().stream().map(LayerCatalog::paramJson).toList());
        m.put("options", e.options().stream().map(o -> Map.of("key", o.key(), "name", o.name(),
                "params", o.params().stream().map(LayerCatalog::paramJson).toList())).toList());
        return m;
    }

    private static Map<String, Object> paramJson(Param p) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", p.name());
        m.put("kind", p.kind().name().toLowerCase(java.util.Locale.ROOT));
        m.put("array", p.array());
        m.put("type", p.type().getSimpleName());
        if (!p.choices().isEmpty()) {
            m.put("choices", p.choices());
        }
        return m;
    }

    /** What a type makes: {@code spec}, {@code layer}, {@code multi}, or null if it is no layer. */
    static String produces(Class<?> t) {
        if (LayerSpec.class.isAssignableFrom(t)) {
            return "spec";
        }
        if (MultiInputLayer.Spec.class.isAssignableFrom(t) || MultiInputLayer.class.isAssignableFrom(t)) {
            return "multi";
        }
        if (Layer.class.isAssignableFrom(t)) {
            return "layer";
        }
        return null;
    }

    private static List<Option> options(Class<?> made) {
        Map<String, Option> out = new LinkedHashMap<>();
        for (Method m : made.getMethods()) {
            // a chainable option returns the same type, like Spec stride(int...)
            if (Modifier.isStatic(m.getModifiers()) || m.isSynthetic() || m.isBridge()
                    || !m.getDeclaringClass().getName().startsWith(PREFIX) || !m.getReturnType().equals(made)) {
                continue;
            }
            List<Param> params = params(m);
            if (params == null || params.stream().anyMatch(p -> p.kind() == Kind.RANDOM)) {
                continue;
            }
            String key = m.getName() + signature(m);
            out.putIfAbsent(key, new Option(key, m.getName(), params, m));
        }
        List<Option> list = new ArrayList<>(out.values());
        list.sort(Comparator.comparing(Option::name).thenComparing(Option::key));
        return List.copyOf(list);
    }

    /** The parameters, or null if one cannot be filled in. */
    private static List<Param> params(Executable e) {
        List<Param> out = new ArrayList<>();
        for (Parameter p : e.getParameters()) {
            Class<?> t = p.getType();
            boolean array = t.isArray();
            Class<?> one = array ? t.getComponentType() : t;
            Kind k = kind(one);
            if (k == null || array && (one.isArray() || k == Kind.RANDOM)) {
                return null;
            }
            out.add(new Param(p.getName(), k, array, k == Kind.CHOICE ? choices(one) : List.of(), t));
        }
        return List.copyOf(out);
    }

    static Kind kind(Class<?> t) {
        if (t == int.class || t == Integer.class || t == short.class || t == byte.class) {
            return Kind.INT;
        }
        if (t == long.class || t == Long.class) {
            return Kind.LONG;
        }
        if (t == float.class || t == Float.class) {
            return Kind.FLOAT;
        }
        if (t == double.class || t == Double.class) {
            return Kind.DOUBLE;
        }
        if (t == boolean.class || t == Boolean.class) {
            return Kind.BOOLEAN;
        }
        if (t == String.class) {
            return Kind.STRING;
        }
        if (t == RandomGenerator.class || t == SplittableRandom.class || t == Random.class) {
            return Kind.RANDOM;
        }
        if (t.isEnum() && Modifier.isPublic(t.getModifiers()) || !choices(t).isEmpty()) {
            return Kind.CHOICE;
        }
        return null;
    }

    /** The named constants of a type: enum values, or public static final fields of that type. */
    static List<String> choices(Class<?> t) {
        if (t.isPrimitive() || t.isArray() || !t.getName().startsWith(PREFIX) || !Modifier.isPublic(t.getModifiers())) {
            return List.of();
        }
        if (t.isEnum()) {
            return Arrays.stream(t.getEnumConstants()).map(c -> ((Enum<?>) c).name()).toList();
        }
        List<String> out = new ArrayList<>();
        for (Field f : t.getFields()) {
            int mod = f.getModifiers();
            if (Modifier.isStatic(mod) && Modifier.isFinal(mod) && t.isAssignableFrom(f.getType())) {
                out.add(f.getName());
            }
        }
        return List.copyOf(out);
    }

    /**
     * Returns the value of a named constant.
     *
     * @param type the type
     * @param name the constant
     * @return its value
     */
    static Object constant(Class<?> type, String name) {
        if (!choices(type).contains(name)) {
            throw new IllegalArgumentException(type.getSimpleName() + " has no " + name + "; it has "
                    + String.join(", ", choices(type)));
        }
        try {
            return type.getField(name).get(null);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot read " + type.getSimpleName() + "." + name, e);
        }
    }

    private static String signature(Executable e) {
        return "(" + String.join(",", Arrays.stream(e.getParameterTypes()).map(Class::getSimpleName).toList()) + ")";
    }

    private static String group(Class<?> c) {
        return c.getEnclosingClass() == null ? c.getSimpleName()
                : group(c.getEnclosingClass()) + "." + c.getSimpleName();
    }

    private static void classesIn(ModuleReference ref, Module module, Set<String> names, Map<String, String> moduleOf) {
        try (ModuleReader reader = ref.open(); Stream<String> list = reader.list()) {
            list.filter(n -> n.endsWith(".class") && !n.endsWith("module-info.class") && !n.contains("$"))
                    .map(n -> n.substring(0, n.length() - 6).replace('/', '.'))
                    .filter(n -> module.isExported(n.substring(0, Math.max(0, n.lastIndexOf('.')))))
                    .forEach(n -> {
                        names.add(n);
                        moduleOf.put(n, module.getName());
                    });
        } catch (IOException e) {
            throw new UncheckedIOException("cannot list " + module.getName(), e);
        }
    }

    private static void classesIn(Path entry, Set<String> names, Map<String, String> moduleOf) {
        String root = PREFIX.replace('.', '/');
        if (Files.isRegularFile(entry) && entry.toString().endsWith(".jar")) {
            try (JarFile jar = new JarFile(entry.toFile())) {
                jar.stream().map(JarEntry::getName)
                        .filter(n -> n.endsWith(".class") && !n.contains("$") && n.startsWith(root))
                        .forEach(n -> names.add(n.substring(0, n.length() - 6).replace('/', '.')));
            } catch (IOException e) {
                throw new UncheckedIOException("cannot list " + entry, e);
            }
            return;
        }
        if (!Files.isDirectory(entry)) {
            return;
        }
        try (Stream<Path> files = Files.walk(entry)) {
            files.map(p -> entry.relativize(p).toString().replace(File.separatorChar, '/'))
                    .filter(n -> n.endsWith(".class") && !n.contains("$") && !n.endsWith("module-info.class")
                            && n.startsWith(PREFIX.replace('.', '/')))
                    .forEach(n -> names.add(n.substring(0, n.length() - 6).replace('/', '.')));
        } catch (IOException e) {
            throw new UncheckedIOException("cannot list " + entry, e);
        }
    }

    private static Class<?> load(String name) {
        try {
            return Class.forName(name, false, LayerCatalog.class.getClassLoader());
        } catch (ClassNotFoundException | LinkageError e) {
            return null;
        }
    }
}
