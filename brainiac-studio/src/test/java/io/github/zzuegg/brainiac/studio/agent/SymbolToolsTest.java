package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Outlines, declarations, uses and the repository map, on small files in several languages. */
class SymbolToolsTest {

    @TempDir
    Path folder;

    private Project project;
    private final SymbolTools tools = new SymbolTools();

    @BeforeEach
    void setUp() throws Exception {
        project = new Project("demo", "Demo", "", folder, null, null, Map.of(), Map.of());
        write("src/demo/Store.java", """
                package demo;

                import java.util.List;
                import java.util.Map;

                /** Keeps items. */
                public class Store {

                    private final Map<String, Object> items = new java.util.HashMap<>();
                    static final int LIMIT = 10;

                    public Store() {
                    }

                    /** Adds one. */
                    public void add(String name,
                            Object value) {
                        if (items.size() < LIMIT) {
                            items.put(name, value);
                        }
                    }

                    public Map<String, Object> all() {
                        String text = "{ not a brace }";
                        return items;
                    }

                    interface Listener {
                        void changed(String name);
                    }
                }
                """);
        write("src/demo/App.java", """
                package demo;

                public class App {
                    public static void main(String[] args) {
                        Store s = new Store();
                        s.add("a", 1);
                        s.add("b", 2);
                    }
                }
                """);
        write("web/store.ts", """
                export class Cart {
                  items: string[] = [];
                  add(x: string): void {
                    this.items.push(x);
                  }
                }
                export const make = (n: number) => new Cart();
                """);
        write("tools/run.py", """
                class Runner:
                    def __init__(self):
                        self.n = 0

                    def run(self, x):
                        return x

                def main():
                    Runner().run(1)
                """);
    }

    private void write(String path, String text) throws Exception {
        Path f = folder.resolve(path);
        Files.createDirectories(f.getParent());
        Files.writeString(f, text);
    }

    private <T> T in(java.util.concurrent.Callable<T> c) throws Exception {
        return ScopedValue.where(ChatService.SCOPE, project).call(c::call);
    }

    private static String names(List<Symbols.Symbol> symbols) {
        return symbols.stream().map(s -> s.line() + "-" + s.end() + " " + s.kind() + " " + s.name())
                .collect(Collectors.joining(", "));
    }

    @Test
    void readsDeclarationsWithTheirLines() {
        String java = names(Symbols.of(folder.resolve("src/demo/Store.java")).symbols());
        assertEquals("7-31 class Store, 9-9 field items, 10-10 field LIMIT, 12-13 constructor Store, "
                + "16-21 method add, 23-26 method all, 28-30 interface Listener, 29-29 method changed", java);
        String ts = names(Symbols.of(folder.resolve("web/store.ts")).symbols());
        assertEquals("1-6 class Cart, 2-2 field items, 3-5 method add, 7-7 function make", ts);
        String py = names(Symbols.of(folder.resolve("tools/run.py")).symbols());
        assertTrue(py.startsWith("1-7 class Runner, 2-4 method __init__, 5-7 method run, 8-"), py);
    }

    @Test
    void outlinesAFile() throws Exception {
        String o = in(() -> tools.outline("src/demo/Store.java"));
        assertTrue(o.contains("16-21 method add: public void add(String name,"), o);
        assertTrue(o.contains("    29-29 method changed"), "nested declarations are indented: " + o);
    }

    @Test
    void findsDeclarationsAndUses() throws Exception {
        Map<String, Object> found = in(() -> tools.findSymbol("Store", Optional.empty(), Optional.empty()));
        assertEquals(2, found.get("total"), "the class and its constructor: " + found);
        Map<String, Object> adds = in(() -> tools.findSymbol("ad*", Optional.of("method"), Optional.empty()));
        assertEquals(2, adds.get("total"), "Store.add and Cart.add: " + adds);
        Map<String, Object> uses = in(() -> tools.findReferences("add", Optional.of("src"), Optional.empty()));
        assertEquals(2, uses.get("total"), String.valueOf(uses));
        String u = String.valueOf(uses.get("uses"));
        assertTrue(u.contains("file=src/demo/App.java, line=6") && u.contains("in=App.main"), u);
        assertTrue(String.valueOf(uses.get("declared")).contains("line=16"), String.valueOf(uses));
    }

    @Test
    void mapsTheRepositoryMostUsedFirst() throws Exception {
        String map = in(() -> tools.repoMap(Optional.empty(), Optional.empty()));
        assertTrue(map.startsWith("4 source files"), map);
        assertTrue(map.indexOf("src/demo/Store.java  (used by 1 files)") < map.indexOf("src/demo/App.java"), map);
        assertTrue(map.contains("method void add(String name,"), map);
        assertTrue(!map.contains("field items"), "fields are left out of the map: " + map);
    }
}
