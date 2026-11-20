package io.github.zzuegg.brainiac.studio;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** The item API both pages use: list by project, rename, move, duplicate, delete to the trash, restore. */
class ItemRoutesTest {

    private static Studio studio;
    private static Path data;
    private static final HttpClient HTTP = HttpClient.newHttpClient();

    @BeforeAll
    static void start() throws IOException {
        Path home = Files.createTempDirectory("studio-items-test");
        data = home.resolve("data");
        studio = Studio.start(new Studio.Settings(0, List.of(home.resolve("models")), data, Map.of()));
    }

    @AfterAll
    static void stop() {
        studio.close();
    }

    private static HttpResponse<String> call(String path, String project, Object body) throws Exception {
        HttpRequest.Builder b = HttpRequest.newBuilder(studio.uri().resolve(path)).header("X-Studio-Token", studio.token());
        if (project != null) {
            b.header("X-Studio-Project", project);
        }
        b = body == null ? b.GET() : b.header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body)));
        return HTTP.send(b.build(), HttpResponse.BodyHandlers.ofString());
    }

    private static Map<String, Object> ok(String path, String project, Object body) throws Exception {
        HttpResponse<String> r = call(path, project, body);
        assertEquals(200, r.statusCode(), path + ": " + r.body());
        return Json.object(Json.parse(r.body()));
    }

    private static List<Map<String, Object>> items(String project, List<String> kinds) throws Exception {
        Map<String, Object> body = kinds == null ? Map.of("project", project) : Map.of("project", project, "kinds", kinds);
        return Json.array(ok("/api/items", null, body).get("items")).stream().map(Json::object).toList();
    }

    private static List<String> ids(List<Map<String, Object>> items) {
        return items.stream().map(i -> (String) i.get("id")).toList();
    }

    private static void project(String id, String name) throws Exception {
        ok("/api/projects", null, Map.of("create", true, "project", Map.of("id", id, "name", name)));
    }

    @Test
    void aChatBelongsToTheProjectItWasMadeInAndCanBeRenamedMovedAndCopied() throws Exception {
        project("alpha", "Alpha");
        project("beta", "Beta");
        ok("/api/chats", "alpha", Map.of("id", "chat0001", "title", "First", "messages", List.of(), "updated", 1000));
        ok("/api/chats", null, Map.of("id", "chat0002", "title", "Elsewhere", "messages", List.of(), "updated", 2000));

        assertEquals(List.of("chat0001"), ids(items("alpha", List.of("chat"))));
        assertFalse(ids(items("default", List.of("chat"))).contains("chat0001"));
        assertTrue(ids(items("default", List.of("chat"))).contains("chat0002"));
        // saving it again from the other page keeps it where it is
        ok("/api/chats", "beta", Map.of("id", "chat0001", "title", "First", "messages", List.of(), "updated", 1000));
        assertEquals(List.of("chat0001"), ids(items("alpha", List.of("chat"))));

        Map<String, Object> renamed = Json.object(ok("/api/items/rename", null,
                Map.of("kind", "chat", "id", "chat0001", "title", "Better name")).get("item"));
        assertEquals("Better name", renamed.get("title"));
        assertEquals("alpha", renamed.get("project"));
        assertEquals(400, call("/api/items/rename", null, Map.of("kind", "chat", "id", "chat0001", "title", " ")).statusCode());

        Map<String, Object> copy = Json.object(ok("/api/items/duplicate", null, Map.of("kind", "chat", "id", "chat0001")).get("item"));
        assertEquals("Better name (copy)", copy.get("title"));
        assertEquals("alpha", copy.get("project"));

        ok("/api/items/move", null, Map.of("kind", "chat", "id", "chat0001", "project", "beta"));
        assertEquals(List.of("chat0001"), ids(items("beta", List.of("chat"))));
        assertEquals(400, call("/api/items/move", null, Map.of("kind", "chat", "id", "chat0001", "project", "nope")).statusCode());
    }

    @Test
    void deletingGoesToTheTrashAndCanBeUndone() throws Exception {
        project("gamma", "Gamma");
        ok("/api/chats", "gamma", Map.of("id", "trash001", "title", "To throw", "messages", List.of(Map.of("role", "user",
                "text", "hi")), "updated", 3000));
        Map<String, Object> entry = Json.object(ok("/api/items/delete", null, Map.of("kind", "chat", "id", "trash001")).get("trash"));
        assertEquals("To throw", Json.object(entry.get("item")).get("title"));
        assertFalse(ids(items("gamma", List.of("chat"))).contains("trash001"));
        assertTrue(Json.array(Json.object(Json.parse(call("/api/chats", null, null).body())).get("chats")).stream()
                .noneMatch(c -> "trash001".equals(Json.object(c).get("id"))), "the old page no longer sees it either");
        assertTrue(Json.array(ok("/api/trash", null, null).get("entries")).stream()
                .anyMatch(e -> entry.get("id").equals(Json.object(e).get("id"))));

        Map<String, Object> back = Json.object(ok("/api/trash/restore", null, Map.of("id", entry.get("id"))).get("item"));
        assertEquals("gamma", back.get("project"), "it comes back in its project");
        assertEquals(1, ((Number) back.get("messages")).intValue());
        assertEquals(400, call("/api/trash/restore", null, Map.of("id", entry.get("id"))).statusCode());

        Map<String, Object> again = Json.object(ok("/api/items/delete", null, Map.of("kind", "chat", "id", "trash001")).get("trash"));
        ok("/api/trash/empty", null, Map.of("id", again.get("id")));
        assertTrue(Json.array(ok("/api/trash", null, null).get("entries")).stream()
                .noneMatch(e -> again.get("id").equals(Json.object(e).get("id"))));
        assertEquals(400, call("/api/trash/restore", null, Map.of("id", again.get("id"))).statusCode());
    }

    @Test
    void aGraphIsRenamedWithItsFileAndTheBuiltModel() throws Exception {
        project("delta", "Delta");
        Map<String, Object> graph = Map.of("nodes", List.of());
        ok("/api/builder/graphs", "delta", Map.of("name", "net-a", "graph", graph));
        assertEquals(List.of("net-a"), ids(items("delta", List.of("graph"))));
        Files.createDirectories(data.resolve("models"));
        Files.writeString(data.resolve("models").resolve("net-a.brainiac"), "pretend");

        Map<String, Object> renamed = Json.object(ok("/api/items/rename", null,
                Map.of("kind", "graph", "id", "net-a", "title", "net b")).get("item"));
        assertEquals("net b", renamed.get("id"));
        assertEquals("delta", renamed.get("project"), "the project goes with the new name");
        assertTrue(Files.exists(data.resolve("models").resolve("net b.brainiac")));
        assertFalse(Files.exists(data.resolve("graphs").resolve("net-a.json")));
        assertEquals(400, call("/api/items/rename", null, Map.of("kind", "graph", "id", "net b", "title", "../x")).statusCode());

        Map<String, Object> copy = Json.object(ok("/api/items/duplicate", null, Map.of("kind", "graph", "id", "net b")).get("item"));
        assertEquals("net b copy", copy.get("id"));
        assertEquals(false, copy.get("built"));
    }

    @Test
    void aDeletedProjectComesBackWithItsItems() throws Exception {
        project("epsilon", "Epsilon");
        ok("/api/chats", "epsilon", Map.of("id", "epsi0001", "title", "Kept", "messages", List.of(), "updated", 4000));
        ok("/api/items/rename", null, Map.of("kind", "project", "id", "epsilon", "title", "Epsilon two"));
        assertTrue(items("*", List.of("project")).stream().anyMatch(p -> "Epsilon two".equals(p.get("title"))));

        Map<String, Object> entry = Json.object(ok("/api/items/delete", null, Map.of("kind", "project", "id", "epsilon")).get("trash"));
        assertTrue(ids(items("default", List.of("chat"))).contains("epsi0001"), "its items show in the default project meanwhile");
        ok("/api/trash/restore", null, Map.of("id", entry.get("id")));
        assertEquals(List.of("epsi0001"), ids(items("epsilon", List.of("chat"))));
        assertEquals(400, call("/api/items/delete", null, Map.of("kind", "project", "id", "default")).statusCode());
    }

    @Test
    void theSecondPageIsServedBesideTheFirst() throws Exception {
        HttpResponse<String> page = HTTP.send(HttpRequest.newBuilder(studio.uri().resolve("/v2/p/default/chat/abcd")).build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, page.statusCode());
        assertTrue(page.body().contains("<meta name=\"studio-token\" content=\"" + studio.token() + "\">"));
        assertFalse(page.headers().firstValue("Content-Security-Policy").orElse("").contains("googleapis"),
                "the second page loads nothing from the internet");
        HttpResponse<String> lib = HTTP.send(HttpRequest.newBuilder(studio.uri().resolve("/v2/vendor/preact.js")).build(),
                HttpResponse.BodyHandlers.ofString());
        assertEquals(200, lib.statusCode());
        assertTrue(lib.headers().firstValue("Content-Type").orElse("").startsWith("text/javascript"));
        for (String bad : List.of("/v2/../app.js", "/v2/vendor/..%2F..%2Fapp.js", "/v2/.hidden/x.js", "/v2/missing.js")) {
            assertEquals(404, HTTP.send(HttpRequest.newBuilder(studio.uri().resolve(bad)).build(),
                    HttpResponse.BodyHandlers.ofString()).statusCode(), bad);
        }
    }
}
