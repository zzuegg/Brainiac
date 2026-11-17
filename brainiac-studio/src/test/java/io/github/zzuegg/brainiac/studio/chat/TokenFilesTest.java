package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The Hugging Face token can live in Brainiac's shared token file, and the CLI's file is read too. */
class TokenFilesTest {

    @TempDir
    Path home;

    private Keys keys(Map<String, String> env) {
        return new Keys(home.resolve("studio/keys.properties"), env, Keys.standardFiles(home, env));
    }

    private static List<String> sources(Keys k) {
        return k.list("huggingface").stream().map(e -> e.source() + ":" + e.label() + ":" + e.hint()).toList();
    }

    @Test
    void tokensAreTriedSavedThenSharedFileThenEnvironmentThenCli() throws Exception {
        Path cli = home.resolve(".cache/huggingface/token");
        Files.createDirectories(cli.getParent());
        Files.writeString(cli, "hf_cli_token_0000cli1\n");
        Keys k = keys(Map.of("HF_TOKEN", "hf_env_token_000env2"));
        assertEquals(List.of("environment:HF_TOKEN:…env2", "file:huggingface-cli login:…cli1"), sources(k));

        k.saveFile("huggingface", "  hf_shared_token_0shr3 \n");
        Path shared = home.resolve(".brainiac/hf_token");
        assertEquals("hf_shared_token_0shr3\n", Files.readString(shared));
        if (FileSystems.getDefault().supportedFileAttributeViews().contains("posix")) {
            assertEquals("rw-------", PosixFilePermissions.toString(Files.getPosixFilePermissions(shared)));
        }
        k.add("huggingface", "default", "hf_saved_token_0sav4");
        assertEquals(List.of("saved:default:…sav4", "file:~/.brainiac/hf_token:…shr3", "environment:HF_TOKEN:…env2",
                "file:huggingface-cli login:…cli1"), sources(k));
        assertEquals("hf_shared_token_0shr3", k.get("huggingface", "~/.brainiac/hf_token").orElseThrow());

        assertThrows(IllegalArgumentException.class, () -> k.saveFile("huggingface", "two words"));
        assertThrows(IllegalArgumentException.class, () -> k.saveFile("openai", "sk-x"));
        assertTrue(k.removeFile("huggingface"));
        assertFalse(k.removeFile("huggingface"));
        assertFalse(Files.exists(shared));
    }

    @Test
    void hfHomeMovesTheCliFile() throws Exception {
        Path hfHome = Files.createDirectories(home.resolve("hf"));
        Files.writeString(hfHome.resolve("token"), "hf_from_hf_home_00hom5");
        Keys k = keys(Map.of("HF_HOME", hfHome.toString()));
        assertEquals(List.of("file:huggingface-cli login:…hom5"), sources(k));
    }

    @Test
    void theStudioDescribesTheTokenWithoutItsValue() throws Exception {
        Keys k = keys(Map.of());
        ChatService chat = new ChatService(k, List.of(), home.resolve("studio/outputs"));
        assertNull(chat.hubToken());
        assertTrue(chat.saveHubToken("hf_written_by_studio_0new6"));
        Map<String, Object> t = chat.hubToken();
        assertEquals(Map.of("label", "~/.brainiac/hf_token", "source", "file", "hint", "…new6", "managed", true), t);
        assertFalse(t.toString().contains("written"), "the token never leaves the server");
        assertTrue(chat.saveHubToken(null));
        assertNull(chat.hubToken());
    }
}
