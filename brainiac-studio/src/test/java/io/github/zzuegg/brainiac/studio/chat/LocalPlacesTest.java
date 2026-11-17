package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Models on this PC can be added from anywhere on the Models screen, and are remembered. */
class LocalPlacesTest {

    @TempDir
    Path data;

    @TempDir
    Path disk;

    private ChatService service() {
        return ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(data.resolve("models")),
                Map.of(), data.resolve("outputs"));
    }

    private static List<String> localModels(ChatService s) {
        return Json.array(s.describe(false).get("models")).stream().map(Json::object)
                .filter(m -> "local".equals(m.get("provider"))).map(m -> (String) m.get("name")).toList();
    }

    private static void modelFolder(Path p) throws IOException {
        Files.createDirectories(p);
        Files.writeString(p.resolve("config.json"), "{\"model_type\": \"qwen2\"}");
        Files.writeString(p.resolve("tokenizer.json"), "{}");
    }

    @Test
    void aFileAFolderOfModelsAndAModelFolderCanBeAdded() throws Exception {
        Path shelf = Files.createDirectories(disk.resolve("shelf"));
        Files.writeString(shelf.resolve("tiny.gguf"), "GGUF");
        modelFolder(shelf.resolve("qwen-0.5b"));
        Files.writeString(shelf.resolve("notes.txt"), "not a model");
        Path single = Files.writeString(disk.resolve("other.gguf"), "GGUF");
        modelFolder(disk.resolve("mistral-7b"));

        ChatService s = service();
        assertTrue(localModels(s).isEmpty());
        assertEquals(2, s.addLocal(shelf));
        assertEquals(1, s.addLocal(single));
        assertEquals(1, s.addLocal(disk.resolve("mistral-7b")));
        assertEquals(List.of("qwen-0.5b", "tiny.gguf", "other.gguf", "mistral-7b"), localModels(s));

        // kept in the data folder for the next start
        ChatService again = service();
        assertEquals(4, localModels(again).size());
        Map<String, Object> local = Json.object(Json.array(again.describe(false).get("providers")).getFirst());
        assertEquals(3, Json.array(local.get("added")).size());

        assertTrue(again.removeLocal(single));
        assertFalse(again.removeLocal(single));
        assertEquals(List.of("qwen-0.5b", "tiny.gguf", "mistral-7b"), localModels(again));
        assertTrue(Files.exists(single), "forgetting a place leaves its files alone");
    }

    @Test
    void whatIsNotAModelIsRefusedWithAReason() throws Exception {
        ChatService s = service();
        IOException none = assertThrows(IOException.class, () -> s.addLocal(disk.resolve("missing")));
        assertTrue(none.getMessage().startsWith("nothing at"), none.getMessage());
        Path txt = Files.writeString(disk.resolve("readme.txt"), "hi");
        IOException wrong = assertThrows(IOException.class, () -> s.addLocal(txt));
        assertTrue(wrong.getMessage().contains("is not a .gguf file"), wrong.getMessage());
    }

    @Test
    void foldersCanBeBrowsedForModels() throws Exception {
        Files.createDirectories(disk.resolve("a folder"));
        Files.writeString(disk.resolve("x.gguf"), "GGUF");
        modelFolder(disk.resolve("llama"));
        Files.writeString(disk.resolve("skip.txt"), "");
        Files.createDirectories(disk.resolve(".hidden"));

        ChatService s = service();
        Map<String, Object> listed = s.browse(disk);
        assertEquals(disk.toAbsolutePath().normalize().toString(), listed.get("path"));
        assertEquals(disk.getParent().toString(), listed.get("parent"));
        List<String> kinds = Json.array(listed.get("entries")).stream().map(Json::object)
                .map(e -> e.get("name") + ":" + e.get("kind")).toList();
        assertEquals(List.of("a folder:folder", "llama:model", "x.gguf:gguf"), kinds);

        Map<String, Object> start = s.browse(null);
        assertFalse(Json.array(start.get("entries")).isEmpty(), "home and the file system roots");
        assertThrows(IOException.class, () -> s.browse(disk.resolve("x.gguf")));
    }
}
