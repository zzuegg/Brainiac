package io.github.zzuegg.brainiac.studio.agent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Finds the declarations in source files: classes, methods, functions and fields, with the
 * lines they span. It reads Java, Kotlin, JavaScript, TypeScript, C-like languages and Python
 * line by line, by their braces or indentation, without compiling anything, so it works on code
 * that does not build yet. What it finds is kept per file until the file changes.
 */
final class Symbols {

    /**
     * One declaration.
     *
     * @param name its name
     * @param kind class, interface, enum, record, method, function, field, constructor or object
     * @param line its first line, from 1
     * @param end its last line
     * @param signature the declaring line, trimmed
     * @param container the declaration it sits in, or empty
     * @param depth how deep it is nested, 0 at the top
     */
    record Symbol(String name, String kind, int line, int end, String signature, String container, int depth) {
    }

    /**
     * What a file declares, and the words it uses.
     *
     * @param modified when the file was read
     * @param size its size then
     * @param symbols its declarations, in order
     * @param words the identifiers in it
     */
    record FileSymbols(long modified, long size, List<Symbol> symbols, Set<String> words) {
    }

    private static final Map<Path, FileSymbols> CACHE = new ConcurrentHashMap<>();
    private static final int MAX_CACHED = 50_000;
    private static final long MAX_BYTES = 2L << 20;

    private static final Set<String> NOT_NAMES = Set.of("if", "for", "while", "switch", "catch", "return", "new",
            "else", "throw", "synchronized", "try", "do", "case", "assert", "super", "this", "yield", "when", "sizeof",
            "typeof", "await", "function", "func", "foreach", "using", "lock", "elif", "with");

    private static final Pattern TYPE = Pattern.compile(
            "^\\s*(?:@[\\w.]+(?:\\([^)]*\\))?\\s+)*(?:(?:public|protected|private|internal|static|final|abstract|sealed|"
            + "non-sealed|open|data|inner|enum|annotation|value|export|default|declare|strictfp)\\s+)*"
            + "(class|interface|enum|record|@interface|object|trait|struct|namespace)\\s+([A-Za-z_$][\\w$]*)");
    private static final Pattern KOTLIN_FUN = Pattern.compile(
            "^\\s*(?:(?:public|protected|private|internal|override|open|abstract|suspend|inline|operator|infix|tailrec|"
            + "external|actual|expect)\\s+)*fun\\s+(?:<[^>]*>\\s*)?(?:[\\w.<>?]+\\.)?([A-Za-z_][\\w]*)\\s*\\(");
    private static final Pattern JS_FUNCTION = Pattern.compile(
            "^\\s*(?:export\\s+)?(?:default\\s+)?(?:async\\s+)?function\\s*\\*?\\s*([A-Za-z_$][\\w$]*)\\s*\\(");
    private static final Pattern JS_ARROW = Pattern.compile(
            "^\\s*(?:export\\s+)?(?:const|let|var)\\s+([A-Za-z_$][\\w$]*)\\s*(?::[^=]+)?=\\s*(?:async\\s+)?"
            + "(?:function\\b|\\([^)]*\\)\\s*(?::[^=]+)?=>|[A-Za-z_$][\\w$]*\\s*=>)");
    // a method or function: modifiers or a type, a name, an opening parenthesis, and no semicolon before a brace
    private static final Pattern METHOD = Pattern.compile(
            "^\\s*(?:@[\\w.]+(?:\\([^)]*\\))?\\s+)*(?:(?:public|protected|private|static|final|abstract|synchronized|"
            + "native|default|override|virtual|async|get|set|inline|constexpr|explicit|unsafe|pub|fn)\\s+)*"
            + "(?:<[^>]+>\\s+)?(?:[\\w$.\\[\\]?&*:]+(?:\\s*<[^()]*>)?(?:\\[\\])*\\s+)?([A-Za-z_$~][\\w$]*)\\s*\\(");
    private static final Pattern FIELD = Pattern.compile(
            "^\\s*(?:@[\\w.]+(?:\\([^)]*\\))?\\s+)*(?:(?:public|protected|private|static|final|volatile|transient|"
            + "const|readonly|val|var|let|lateinit|override|internal)\\s+)*(?:[\\w$.<>\\[\\],? ]+\\s+)?"
            + "([A-Za-z_$][\\w$]*)\\s*(?::\\s*[^=;]+)?(?:=|;)");
    private static final Pattern PY_DEF = Pattern.compile("^(\\s*)(?:async\\s+)?def\\s+([A-Za-z_]\\w*)\\s*\\(");
    private static final Pattern PY_CLASS = Pattern.compile("^(\\s*)class\\s+([A-Za-z_]\\w*)");
    private static final Pattern WORD = Pattern.compile("[A-Za-z_$][\\w$]*");

    private Symbols() {
    }

    /**
     * Whether a file is source code this reads.
     *
     * @param name the file's name
     * @return true for source
     */
    static boolean source(String name) {
        String n = name.toLowerCase(Locale.ROOT);
        int dot = n.lastIndexOf('.');
        if (dot < 0) {
            return false;
        }
        return switch (n.substring(dot + 1)) {
            case "java", "kt", "kts", "scala", "groovy", "js", "mjs", "cjs", "jsx", "ts", "tsx", "py", "c", "h", "cc",
                    "cpp", "hpp", "cxx", "cs", "go", "rs", "swift", "dart", "php", "cu", "cuh" -> true;
            default -> false;
        };
    }

    /**
     * What a file declares, read again only if it changed.
     *
     * @param f the file
     * @return its symbols and words
     */
    static FileSymbols of(Path f) {
        try {
            long modified = Files.getLastModifiedTime(f).toMillis();
            long size = Files.size(f);
            FileSymbols had = CACHE.get(f);
            if (had != null && had.modified() == modified && had.size() == size) {
                return had;
            }
            if (size > MAX_BYTES || !FileWalk.text(f)) {
                return new FileSymbols(modified, size, List.of(), Set.of());
            }
            String text = Files.readString(f, StandardCharsets.UTF_8);
            String name = f.getFileName().toString().toLowerCase(Locale.ROOT);
            List<String> lines = text.lines().toList();
            List<Symbol> symbols = name.endsWith(".py") ? python(lines) : braces(lines, name);
            Set<String> words = new HashSet<>();
            Matcher m = WORD.matcher(text);
            while (m.find()) {
                words.add(m.group());
            }
            FileSymbols fs = new FileSymbols(modified, size, symbols, words);
            if (CACHE.size() > MAX_CACHED) {
                CACHE.clear();
            }
            CACHE.put(f, fs);
            return fs;
        } catch (IOException | RuntimeException e) {
            return new FileSymbols(0, 0, List.of(), Set.of());
        }
    }

    /**
     * Reads the declarations of a language with braces.
     *
     * @param lines the file's lines
     * @param name the file's name, for its language
     * @return the declarations
     */
    static List<Symbol> braces(List<String> lines, String name) {
        boolean kotlin = name.endsWith(".kt") || name.endsWith(".kts");
        boolean js = name.matches(".*\\.(m|c)?(js|jsx|ts|tsx)$");
        List<Symbol> out = new ArrayList<>();
        // open declarations: the symbol's index in out and the depth inside its braces
        Deque<int[]> open = new ArrayDeque<>();
        Deque<Integer> typeDepths = new ArrayDeque<>();   // depths that are a type's body
        int depth = 0;
        boolean inComment = false;
        boolean inTextBlock = false;
        String pendingKind = null;
        for (int i = 0; i < lines.size(); i++) {
            String raw = lines.get(i);
            // the line without strings and comments, for braces and matching
            StringBuilder code = new StringBuilder();
            char quote = 0;
            for (int c = 0; c < raw.length(); c++) {
                char ch = raw.charAt(c);
                char next = c + 1 < raw.length() ? raw.charAt(c + 1) : 0;
                if (inTextBlock) {
                    if (raw.startsWith("\"\"\"", c)) {
                        inTextBlock = false;
                        c += 2;
                    }
                    continue;
                }
                if (inComment) {
                    if (ch == '*' && next == '/') {
                        inComment = false;
                        c++;
                    }
                    continue;
                }
                if (quote != 0) {
                    if (ch == '\\') {
                        c++;
                    } else if (ch == quote) {
                        quote = 0;
                        code.append(ch);
                    }
                    continue;
                }
                if (ch == '/' && next == '/') {
                    break;
                }
                if (ch == '/' && next == '*') {
                    inComment = true;
                    c++;
                    continue;
                }
                if (raw.startsWith("\"\"\"", c)) {
                    inTextBlock = true;
                    code.append("\"\"");
                    c += 2;
                    continue;
                }
                if (ch == '"' || ch == '`' || (ch == '\'' && !kotlin)) {
                    quote = ch;
                    code.append(ch);
                    continue;
                }
                code.append(ch);
            }
            String line = code.toString();
            boolean inType = !typeDepths.isEmpty() && typeDepths.peek() == depth;
            String container = open.isEmpty() ? "" : out.get(open.peek()[0]).name();
            boolean interfaceBody = inType && !open.isEmpty()
                    && List.of("interface", "annotation", "trait").contains(out.get(open.peek()[0]).kind());
            int nesting = open.size();
            // the lines between a declaration and its body, such as parameters, declare nothing
            boolean inHeader = !open.isEmpty() && open.peek()[1] < 0;
            Matcher m;
            Symbol found = null;
            if ((m = TYPE.matcher(line)).find()) {
                String kind = m.group(1).equals("@interface") ? "annotation" : m.group(1);
                found = new Symbol(m.group(2), kind, i + 1, i + 1, raw.strip(), container, nesting);
                pendingKind = "type";
            } else if (kotlin && (m = KOTLIN_FUN.matcher(line)).find()) {
                found = new Symbol(m.group(1), inType ? "method" : "function", i + 1, i + 1, raw.strip(), container, nesting);
                pendingKind = "body";
            } else if (js && ((m = JS_FUNCTION.matcher(line)).find() || (m = JS_ARROW.matcher(line)).find())) {
                found = new Symbol(m.group(1), inType ? "method" : "function", i + 1, i + 1, raw.strip(), container, nesting);
                pendingKind = "body";
            } else if (!inHeader && (inType || depth == 0) && (m = METHOD.matcher(line)).find() && !NOT_NAMES.contains(m.group(1))
                    && declares(line, lines, i, interfaceBody) && (inType || !js)) {
                String n = m.group(1);
                String kind = inType ? (n.equals(container) ? "constructor" : "method") : "function";
                found = new Symbol(n, kind, i + 1, i + 1, raw.strip(), container, nesting);
                pendingKind = "body";
            } else if (!inHeader && inType && (m = FIELD.matcher(line)).find() && !NOT_NAMES.contains(m.group(1))
                    && !line.strip().startsWith("return")
                    && (!line.contains("(") || (line.indexOf('=') >= 0 && line.indexOf('=') < line.indexOf('(')))) {
                out.add(new Symbol(m.group(1), "field", i + 1, i + 1, raw.strip(), container, nesting));
            }
            if (found != null) {
                out.add(found);
                if (!line.contains("{") && line.strip().endsWith(";")) {
                    pendingKind = null;   // abstract or interface method: no body
                } else {
                    open.push(new int[] {out.size() - 1, -1});
                }
            }
            for (int c = 0; c < line.length(); c++) {
                char ch = line.charAt(c);
                if (ch == '{') {
                    depth++;
                    if (pendingKind != null && !open.isEmpty() && open.peek()[1] < 0) {
                        open.peek()[1] = depth;
                        if (pendingKind.equals("type")) {
                            typeDepths.push(depth);
                        }
                        pendingKind = null;
                    }
                } else if (ch == '}') {
                    if (!typeDepths.isEmpty() && typeDepths.peek() == depth) {
                        typeDepths.pop();
                    }
                    while (!open.isEmpty() && open.peek()[1] == depth) {
                        int[] o = open.pop();
                        Symbol s = out.get(o[0]);
                        out.set(o[0], new Symbol(s.name(), s.kind(), s.line(), i + 1, s.signature(), s.container(),
                                s.depth()));
                    }
                    depth = Math.max(0, depth - 1);
                }
            }
            // a declaration whose body never opened within a few lines had none
            if (!open.isEmpty() && open.peek()[1] < 0 && i + 1 - out.get(open.peek()[0]).line() > 25) {
                open.pop();
                pendingKind = null;
            }
        }
        return out;
    }

    // a call's line ends with ';' or sits inside an expression; a declaration opens a body or throws
    private static boolean declares(String line, List<String> lines, int i, boolean bodiless) {
        String t = line.strip();
        if (t.contains("=") && t.indexOf('=') < t.indexOf('(')) {
            return false;   // an assignment from a call
        }
        if (t.startsWith(".") || t.startsWith("return ") || t.startsWith("new ")) {
            return false;
        }
        // a plain call such as foo(); has no type before its name
        String before = t.substring(0, t.indexOf('(')).strip();
        if (!before.contains(" ") && !before.contains("<")) {
            // constructors and C-style functions at the top look the same as calls; keep those with a body
            StringBuilder rest = new StringBuilder(t);
            for (int j = i + 1; j < Math.min(lines.size(), i + 3); j++) {
                rest.append(' ').append(lines.get(j).strip());
            }
            String r = rest.toString();
            int close = r.indexOf(')');
            return close >= 0 && r.substring(close).matches("\\)\\s*(throws [^{;]+|:\\s*[^{;=]+)?\\{.*");
        }
        return !t.endsWith(");") || bodiless || t.contains("abstract ") || t.contains("native ");
    }

    /**
     * Reads Python's declarations by indentation.
     *
     * @param lines the file's lines
     * @return the declarations
     */
    static List<Symbol> python(List<String> lines) {
        List<Symbol> out = new ArrayList<>();
        Deque<int[]> open = new ArrayDeque<>();   // symbol index, indentation
        for (int i = 0; i < lines.size(); i++) {
            String l = lines.get(i);
            if (l.isBlank() || l.strip().startsWith("#")) {
                continue;
            }
            int indent = l.length() - l.stripLeading().length();
            while (!open.isEmpty() && open.peek()[1] >= indent) {
                int[] o = open.pop();
                close(out, o[0], i);
            }
            Matcher m;
            String container = open.isEmpty() ? "" : out.get(open.peek()[0]).name();
            boolean inClass = !open.isEmpty() && out.get(open.peek()[0]).kind().equals("class");
            if ((m = PY_CLASS.matcher(l)).find()) {
                out.add(new Symbol(m.group(2), "class", i + 1, i + 1, l.strip(), container, open.size()));
                open.push(new int[] {out.size() - 1, indent});
            } else if ((m = PY_DEF.matcher(l)).find()) {
                out.add(new Symbol(m.group(2), inClass ? "method" : "function", i + 1, i + 1, l.strip(), container,
                        open.size()));
                open.push(new int[] {out.size() - 1, indent});
            }
        }
        while (!open.isEmpty()) {
            close(out, open.pop()[0], lines.size());
        }
        return out;
    }

    // ends a Python declaration at the last line with code before 'next'
    private static void close(List<Symbol> out, int index, int next) {
        Symbol s = out.get(index);
        out.set(index, new Symbol(s.name(), s.kind(), s.line(), Math.max(s.line(), next), s.signature(), s.container(),
                s.depth()));
    }
}
