package io.github.zzuegg.brainiac.studio.agent;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Reads what builds and tests leave behind: compile errors from the output of javac, Kotlin,
 * TypeScript, Rust, Go and the like, and test results from JUnit XML reports. Paths come back
 * relative to the project's folder, as the other file tools take them.
 */
final class BuildOutput {

    /** The most errors or failures returned. */
    static final int MAX = 50;

    // path:line[:col]: error: message (javac, gcc, Go, many others)
    private static final Pattern COLON = Pattern.compile(
            "^(?:\\[ERROR] )?((?:[A-Za-z]:)?[^\\s:()\\[\\]]+\\.[A-Za-z0-9]+):(\\d+)(?::(\\d+))?:?\\s*(?:(?:error|fatal error)\\s*:\\s*)?(.+)$");
    // e: file:///work/x.kt:12:5 message (Kotlin)
    private static final Pattern KOTLIN = Pattern.compile("^e: (?:file://)?(\\S+?):(\\d+):(\\d+)\\s+(.+)$");
    // x.ts(12,5): error TS2322: message (tsc)
    private static final Pattern TSC = Pattern.compile("^(\\S+?)\\((\\d+),(\\d+)\\): error (.+)$");
    // [ERROR] /work/x.java:[12,5] message (Maven)
    private static final Pattern MAVEN = Pattern.compile("^\\[ERROR] (\\S+?):\\[(\\d+),(\\d+)] (.+)$");
    // error[E0425]: message, then  --> src/main.rs:12:5 (rustc)
    private static final Pattern RUST_HEAD = Pattern.compile("^error(?:\\[\\w+])?: (.+)$");
    private static final Pattern RUST_AT = Pattern.compile("^\\s*--> (\\S+?):(\\d+):(\\d+)$");
    private static final Pattern FRAME = Pattern.compile("^\\s*at ([\\w$.]+)\\.([\\w$<>]+)\\(([\\w$]+\\.\\w+):(\\d+)\\)$");

    private BuildOutput() {
    }

    /**
     * Finds compile errors in a build's output.
     *
     * @param output what the build printed
     * @param copy the sandbox's copy, mounted at {@code /work}
     * @param root the project's folder
     * @param log the log, for when the output was cut
     * @return the errors: file, line, column if known, message
     */
    static List<Map<String, Object>> errors(String output, Path copy, Path root, String log) {
        List<Map<String, Object>> out = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        String[] lines = output.split("\\R");
        String rustMessage = null;
        for (int i = 0; i < lines.length && out.size() < MAX; i++) {
            String l = lines[i];
            Matcher m;
            String file = null;
            String message = null;
            int line = 0;
            int column = 0;
            if ((m = KOTLIN.matcher(l)).matches() || (m = TSC.matcher(l)).matches() || (m = MAVEN.matcher(l)).matches()) {
                file = m.group(1);
                line = Integer.parseInt(m.group(2));
                column = Integer.parseInt(m.group(3));
                message = m.group(4);
            } else if ((m = RUST_HEAD.matcher(l)).matches()) {
                rustMessage = m.group(1);
                continue;
            } else if (rustMessage != null && (m = RUST_AT.matcher(l)).matches()) {
                file = m.group(1);
                line = Integer.parseInt(m.group(2));
                column = Integer.parseInt(m.group(3));
                message = rustMessage;
                rustMessage = null;
            } else if ((m = COLON.matcher(l)).matches() && !l.contains(": warning:") && !l.startsWith("w: ")
                    && (l.contains("error") || l.contains(".java:") || l.contains(".go:") || l.contains(".c:"))) {
                file = m.group(1);
                line = Integer.parseInt(m.group(2));
                column = m.group(3) == null ? 0 : Integer.parseInt(m.group(3));
                message = m.group(4);
                // javac puts what the symbol is on the next lines
                StringBuilder more = new StringBuilder();
                for (int j = i + 1; j < Math.min(lines.length, i + 5); j++) {
                    String n = lines[j].strip();
                    if (n.startsWith("symbol:") || n.startsWith("location:") || n.startsWith("required:")
                            || n.startsWith("found:") || n.startsWith("reason:")) {
                        more.append("; ").append(n);
                    }
                }
                message += more;
            }
            if (file == null || message.isBlank() || l.contains(": warning:")) {
                continue;
            }
            String rel = relative(file, copy, root);
            if (rel == null) {
                continue;
            }
            if (seen.add(rel + ":" + line + ":" + message)) {
                Map<String, Object> e = new LinkedHashMap<>();
                e.put("file", rel);
                e.put("line", line);
                if (column > 0) {
                    e.put("column", column);
                }
                e.put("message", message.strip());
                out.add(e);
            }
        }
        return out;
    }

    /**
     * Counts warnings in a build's output.
     *
     * @param output what the build printed
     * @return how many lines are warnings
     */
    static int warnings(String output) {
        int n = 0;
        for (String l : output.split("\\R")) {
            if (l.contains(": warning:") || l.startsWith("w: ") || l.startsWith("[WARNING] ") || l.startsWith("warning:")) {
                n++;
            }
        }
        return n;
    }

    // a path from the container or the build as a path inside the project; null if it is elsewhere
    static String relative(String file, Path copy, Path root) {
        String f = file.replace('\\', '/');
        if (f.startsWith("file://")) {
            f = f.substring("file://".length());
        }
        if (f.startsWith("/work/")) {
            f = f.substring("/work/".length());
        } else if (f.startsWith("./")) {
            f = f.substring(2);
        } else {
            String c = copy.toAbsolutePath().toString().replace('\\', '/') + "/";
            String r = root.toAbsolutePath().toString().replace('\\', '/') + "/";
            if (f.startsWith(c)) {
                f = f.substring(c.length());
            } else if (f.startsWith(r)) {
                f = f.substring(r.length());
            } else if (f.startsWith("/") || f.matches("^[A-Za-z]:/.*")) {
                return null;
            }
        }
        return f.contains("..") ? null : f;
    }

    // ---------------------------------------------------------------- tests

    /**
     * Test results.
     *
     * @param found whether any report was found
     * @param passed how many passed
     * @param failed each failure
     * @param skipped how many were skipped
     */
    record Tests(boolean found, int passed, List<Map<String, Object>> failed, int skipped) {
    }

    /**
     * Deletes old JUnit reports from the sandbox's copy, so only a new run's are read.
     *
     * @param copy the copy
     */
    static void clearReports(Path copy) {
        for (Path f : reports(copy)) {
            try {
                Files.deleteIfExists(f);
            } catch (IOException ignored) {
                // read again; harmless
            }
        }
    }

    private static List<Path> reports(Path copy) {
        if (!Files.isDirectory(copy)) {
            return List.of();
        }
        try (Stream<Path> s = Files.walk(copy)) {
            return s.filter(p -> {
                String n = p.getFileName().toString();
                String path = p.toString().replace('\\', '/');
                return n.startsWith("TEST-") && n.endsWith(".xml") && !path.contains("/node_modules/")
                        && Files.isRegularFile(p);
            }).toList();
        } catch (IOException | java.io.UncheckedIOException e) {
            return List.of();
        }
    }

    /**
     * Reads the JUnit XML reports in the sandbox's copy.
     *
     * @param copy the copy
     * @param root the project's folder, to find test sources
     * @return the results
     */
    static Tests tests(Path copy, Path root) {
        List<Path> files = reports(copy);
        int passed = 0;
        int skipped = 0;
        List<Map<String, Object>> failed = new ArrayList<>();
        DocumentBuilder parser = parser();
        for (Path f : files) {
            Document d;
            try (InputStream in = Files.newInputStream(f)) {
                d = parser.parse(in);
            } catch (Exception e) {
                continue;   // a report being written, or broken
            }
            NodeList cases = d.getElementsByTagName("testcase");
            for (int i = 0; i < cases.getLength(); i++) {
                Element c = (Element) cases.item(i);
                Element problem = first(c, "failure");
                if (problem == null) {
                    problem = first(c, "error");
                }
                if (problem != null) {
                    if (failed.size() < MAX) {
                        failed.add(failure(c, problem, root));
                    }
                } else if (first(c, "skipped") != null) {
                    skipped++;
                } else {
                    passed++;
                }
            }
        }
        return new Tests(!files.isEmpty(), passed, failed, skipped);
    }

    private static Map<String, Object> failure(Element c, Element problem, Path root) {
        String cls = c.getAttribute("classname");
        String name = c.getAttribute("name");
        String message = problem.getAttribute("message");
        String stack = problem.getTextContent() == null ? "" : problem.getTextContent().strip();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("test", cls + "." + name);
        if (message.isBlank()) {
            message = stack.lines().findFirst().orElse("failed");
        }
        m.put("message", message.length() > 600 ? message.substring(0, 600) + "…" : message);
        // the test's own frame says where it failed
        String outer = cls.contains("$") ? cls.substring(0, cls.indexOf('$')) : cls;
        for (String l : stack.lines().toList()) {
            Matcher f = FRAME.matcher(l);
            if (f.matches() && (f.group(1).equals(cls) || f.group(1).startsWith(outer))) {
                String file = source(root, f.group(1), f.group(3));
                if (file != null) {
                    m.put("file", file);
                    m.put("line", Integer.parseInt(f.group(4)));
                }
                break;
            }
        }
        List<String> kept = new ArrayList<>();
        for (String l : stack.lines().toList()) {
            String t = l.strip();
            if (t.startsWith("at org.junit.") || t.startsWith("at java.base/") || t.startsWith("at jdk.internal.")
                    || t.startsWith("at org.gradle.") || t.startsWith("at worker.org.gradle.")
                    || t.startsWith("at org.apache.maven.")) {
                continue;
            }
            kept.add(l.length() > 300 ? l.substring(0, 300) + "…" : l);
            if (kept.size() == 8) {
                break;
            }
        }
        m.put("stack", String.join("\n", kept));
        return m;
    }

    // the source file of a class, found under the project's folder by its package path
    private static String source(Path root, String cls, String fileName) {
        String outer = cls.contains("$") ? cls.substring(0, cls.indexOf('$')) : cls;
        int dot = outer.lastIndexOf('.');
        String suffix = (dot < 0 ? "" : outer.substring(0, dot).replace('.', '/') + "/") + fileName;
        String[] found = {null};
        new FileWalk(root).files((f, rel) -> {
            String n = FileWalk.name(rel);
            if (n.endsWith("/" + suffix) || n.equals(suffix)) {
                found[0] = n;
                return false;
            }
            return true;
        });
        return found[0];
    }

    private static Element first(Element parent, String tag) {
        for (Node n = parent.getFirstChild(); n != null; n = n.getNextSibling()) {
            if (n instanceof Element e && e.getTagName().equals(tag)) {
                return e;
            }
        }
        return null;
    }

    private static DocumentBuilder parser() {
        try {
            DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();
            f.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
            f.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            f.setExpandEntityReferences(false);
            f.setXIncludeAware(false);
            return f.newDocumentBuilder();
        } catch (Exception e) {
            throw new IllegalStateException("cannot read XML: " + e.getMessage(), e);
        }
    }
}
