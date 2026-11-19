package io.github.zzuegg.brainiac.backend.hip;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Finds a native library and makes downcall handles for its functions with the Foreign
 * Function and Memory API.
 *
 * <p>Two linking modes. By default every function is linked normally, so the JVM can stop
 * the calling thread for garbage collection while it runs. With
 * {@code -Dbrainiac.hip.critical=true} the short, non-blocking functions called once per
 * operation (kernel launches, asynchronous copies and fills, rocBLAS products) are linked as
 * {@linkplain Linker.Option#critical(boolean) critical}, which drops the thread state
 * transition and makes each call cheaper; they then must not block for long.
 *
 * <p>Pointers into device memory and opaque handles (streams, functions, rocBLAS handles) are
 * passed as {@code long}: on the 64-bit platforms HIP supports they travel in the same
 * registers as pointers, and no {@link MemorySegment} has to be made for each call.
 */
@SuppressWarnings("restricted")
final class Binder {

    /** Whether hot functions are linked as critical ({@code -Dbrainiac.hip.critical=true}). */
    static final boolean CRITICAL = Boolean.getBoolean("brainiac.hip.critical");

    static final boolean WINDOWS = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).startsWith("windows");

    private static final Linker LINKER = Linker.nativeLinker();

    private final SymbolLookup lookup;
    private final String library;

    private Binder(SymbolLookup lookup, String library) {
        this.lookup = lookup;
        this.library = library;
    }

    /** The library's file, or the name the system loader found it by. */
    String library() {
        return library;
    }

    /**
     * A handle for a function the library must have.
     *
     * @param name the symbol
     * @param descriptor its signature
     * @param hot whether it is called once per operation, so linked as critical in that mode
     */
    MethodHandle bind(String name, FunctionDescriptor descriptor, boolean hot) {
        MemorySegment symbol = lookup.find(name)
                .orElseThrow(() -> new HipException(library + " has no function " + name, 0));
        return hot && CRITICAL
                ? LINKER.downcallHandle(symbol, descriptor, Linker.Option.critical(false))
                : LINKER.downcallHandle(symbol, descriptor);
    }

    /** A handle for a function that older versions of the library lack, or {@code null}. */
    MethodHandle optional(String name, FunctionDescriptor descriptor) {
        return lookup.find(name).map(s -> LINKER.downcallHandle(s, descriptor)).orElse(null);
    }

    /** Wraps a failure from {@link MethodHandle#invokeExact}. */
    static RuntimeException fail(String call, Throwable t) {
        if (t instanceof RuntimeException r) {
            return r;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new HipException(call + " failed: " + t, t);
    }

    // ------------------------------------------------------------------ finding libraries

    /**
     * Loads the first library that can be found: the file named by the system property
     * {@code property} if it is set (and nothing else), otherwise each of {@code names} through
     * the system's loader (which searches {@code PATH} or {@code LD_LIBRARY_PATH} and the
     * linker cache), otherwise the newest file matching {@code pattern} in the HIP SDK's
     * folders.
     *
     * @param what a description for the error message, such as "hipRTC"
     * @param property the system property naming the file
     * @param names file names to try in order, newest version first
     * @param pattern a file name pattern for the toolkit folders, group 1 being the version, or
     *        {@code null} to rely on the system's loader alone
     * @param companion when not {@code null}, maps a found file name to a library in the same
     *        folder that is loaded first (hipRTC needs its builtins library, which Windows only
     *        looks for on the {@code PATH})
     * @return the library
     * @throws HipException if none was found
     */
    static Binder load(String what, String property, List<String> names, Pattern pattern,
            java.util.function.UnaryOperator<String> companion) {
        Arena arena = Arena.global();
        String explicit = System.getProperty(property);
        if (explicit != null && !explicit.isBlank()) {
            Path p = Path.of(explicit);
            try {
                loadCompanion(p, companion, arena);
                return new Binder(SymbolLookup.libraryLookup(p, arena), p.toString());
            } catch (IllegalArgumentException e) {
                throw new HipException("cannot load " + what + " from " + p + " (-D" + property + "): "
                        + e.getMessage(), e);
            }
        }
        List<String> tried = new ArrayList<>();
        // the toolkit's own folders first when they hold a match: a library the system loader
        // finds by its plain name may be an older one left on the PATH
        for (Path dir : pattern == null ? List.<Path>of() : toolkitFolders()) {
            Path found = newest(dir, pattern);
            if (found != null) {
                try {
                    loadCompanion(found, companion, arena);
                    return new Binder(SymbolLookup.libraryLookup(found, arena), found.toString());
                } catch (IllegalArgumentException e) {
                    tried.add(found + " (" + e.getMessage() + ")");
                }
            }
        }
        for (String name : names) {
            try {
                if (companion != null) {
                    try {
                        SymbolLookup.libraryLookup(companion.apply(name), arena);
                    } catch (IllegalArgumentException ignored) {
                        // the library may still find it
                    }
                }
                return new Binder(SymbolLookup.libraryLookup(name, arena), name);
            } catch (IllegalArgumentException e) {
                tried.add(name);
            }
        }
        throw new HipException(what + " not found (tried " + String.join(", ", tried)
                + "); install AMD's HIP SDK (ROCm on Linux) or name the file with -D" + property, 0);
    }

    private static void loadCompanion(Path file, java.util.function.UnaryOperator<String> companion, Arena arena) {
        if (companion == null || file.getFileName() == null) {
            return;
        }
        Path other = file.resolveSibling(companion.apply(file.getFileName().toString()));
        if (Files.isRegularFile(other)) {
            try {
                SymbolLookup.libraryLookup(other, arena);
            } catch (IllegalArgumentException ignored) {
                // the library may still find it
            }
        }
    }

    /** The newest file in {@code dir} whose name matches {@code pattern}, by the version in group 1. */
    static Path newest(Path dir, Pattern pattern) {
        if (!Files.isDirectory(dir)) {
            return null;
        }
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(f -> f.getFileName() != null && pattern.matcher(f.getFileName().toString()).matches())
                    .max(Comparator.comparing((Path f) -> version(pattern, f.getFileName().toString())))
                    .orElse(null);
        } catch (java.io.IOException | SecurityException e) {
            return null;
        }
    }

    /** Group 1 of the pattern as a number with up to three parts, for ordering: "6.4" before "7". */
    static Long version(Pattern pattern, String name) {
        Matcher m = pattern.matcher(name);
        if (!m.matches() || m.groupCount() < 1 || m.group(1) == null) {
            return 0L;
        }
        String[] parts = m.group(1).split("[._]");
        long v = 0;
        for (int i = 0; i < 3; i++) {
            long part = 0;
            if (i < parts.length && !parts[i].isEmpty()) {
                try {
                    part = Long.parseLong(parts[i]);
                } catch (NumberFormatException e) {
                    part = 0;
                }
            }
            v = v * 100_000 + part;
        }
        return v;
    }

    /** Where the HIP SDK (or ROCm) puts its libraries, from the environment and the usual install places. */
    static List<Path> toolkitFolders() {
        Set<Path> roots = new LinkedHashSet<>();
        for (String env : List.of("HIP_PATH", "ROCM_PATH")) {
            String v = System.getenv(env);
            if (v != null && !v.isBlank()) {
                roots.add(Path.of(v.strip()));
            }
        }
        // the Windows installer also sets HIP_PATH_70, HIP_PATH_72, ...: newest first
        System.getenv().entrySet().stream()
                .filter(e -> e.getKey().toUpperCase(Locale.ROOT).startsWith("HIP_PATH_") && !e.getValue().isBlank())
                .sorted(java.util.Map.Entry.<String, String>comparingByKey().reversed())
                .forEach(e -> roots.add(Path.of(e.getValue().strip())));
        if (WINDOWS) {
            Path base = Path.of(System.getenv().getOrDefault("ProgramFiles", "C:\\Program Files"), "AMD", "ROCm");
            roots.addAll(subfolders(base));
        } else {
            roots.add(Path.of("/opt/rocm"));
            roots.addAll(subfolders(Path.of("/opt")).stream()
                    .filter(p -> p.getFileName() != null && p.getFileName().toString().startsWith("rocm-")).toList());
        }
        Set<Path> folders = new LinkedHashSet<>();
        for (Path root : roots) {
            if (WINDOWS) {
                folders.add(root.resolve("bin"));
            } else {
                folders.add(root.resolve("lib"));
                folders.add(root.resolve("lib64"));
            }
        }
        if (!WINDOWS) {
            folders.add(Path.of("/usr/lib/x86_64-linux-gnu"));
            folders.add(Path.of("/usr/lib64"));
        }
        return List.copyOf(folders);
    }

    /** Subfolders, newest name first, so the latest toolkit wins. */
    private static List<Path> subfolders(Path dir) {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(Files::isDirectory)
                    .sorted(Comparator.comparing((Path p) -> p.getFileName().toString()).reversed())
                    .toList();
        } catch (java.io.IOException | SecurityException e) {
            return List.of();
        }
    }
}
