package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.server.WorkerServer;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.security.KeyPair;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The studio lends a worker host its Hugging Face token: sealed on the way, and in the worker's memory only. */
class HostSecretsTest {

    private static final String TOKEN = "a-given-worker-token-0123456789abcdef";
    private static final String SECRET = "hf_aSecretThatMustNotBeWrittenAnywhere";

    @TempDir
    Path studio;

    @TempDir
    Path lab;

    @Test
    void aSealedSecretOpensOnlyWithTheSameTokenAndName() throws Exception {
        KeyPair a = SealedSecret.pair();
        KeyPair b = SealedSecret.pair();
        byte[] mine = SealedSecret.key(a.getPrivate(), SealedSecret.publicText(b), TOKEN);
        byte[] theirs = SealedSecret.key(b.getPrivate(), SealedSecret.publicText(a), TOKEN);
        String sealed = SealedSecret.seal(mine, "huggingface", SECRET);
        assertFalse(sealed.contains(SECRET));
        assertEquals(SECRET, SealedSecret.open(theirs, "huggingface", sealed));
        assertThrows(GeneralSecurityException.class, () -> SealedSecret.open(theirs, "another", sealed));
        byte[] other = SealedSecret.key(b.getPrivate(), SealedSecret.publicText(a), "another-token");
        assertThrows(GeneralSecurityException.class, () -> SealedSecret.open(other, "huggingface", sealed));
        assertThrows(GeneralSecurityException.class, () -> SealedSecret.key(a.getPrivate(), "not a key", TOKEN));
    }

    @Test
    void aLentKeyIsTriedFirstAndNeverSaved() throws Exception {
        Path file = studio.resolve("keys.properties");
        Keys keys = new Keys(file, Map.of("HF_TOKEN", "hf_fromTheEnvironment"));
        assertFalse(keys.lent("huggingface"));
        keys.lend("huggingface", SECRET);
        assertTrue(keys.lent("huggingface"));
        assertEquals(Keys.LENT, keys.list("huggingface").getFirst().label());
        assertEquals(Optional.of(SECRET), keys.get("huggingface", Keys.LENT));
        assertFalse(Files.exists(file));
        keys.lend("huggingface", "");
        assertFalse(keys.lent("huggingface"));
        assertEquals("environment", keys.list("huggingface").getFirst().source());
    }

    @Test
    void aHostIsLentTheTokenAndGivesItBack() throws Exception {
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab,
                List.of(), TOKEN)) {
            Hosts hosts = new Hosts(studio.resolve("hosts.properties"));
            Hosts.Host h = hosts.add("lab", "127.0.0.1:" + worker.port(), TOKEN);
            WorkerClient client = hosts.client(h);
            assertEquals(List.of(), client.info().get("secrets"));

            IOException none = assertThrows(IOException.class, () -> hosts.lend(h.id(), "huggingface", true));
            assertTrue(none.getMessage().contains("no huggingface key"), none.getMessage());
            hosts.secrets(name -> name.equals("huggingface") ? Optional.of(SECRET) : Optional.empty());
            assertThrows(IOException.class, () -> hosts.lend("no-such-host", "huggingface", true));
            IOException unknown = assertThrows(IOException.class, () -> client.lend("openai", "sk-x"));
            assertTrue(unknown.getMessage().contains("no use"), unknown.getMessage());
            // a studio with another token for this worker does not get in at all
            assertThrows(IOException.class, () -> new WorkerClient(h.url(), "another-token").lend("huggingface", "x"));

            hosts.lend(h.id(), "huggingface", true);
            assertEquals(List.of("huggingface"), client.info().get("secrets"));
            Map<String, Object> shown = hosts.toJson().getFirst();
            assertEquals(List.of("huggingface"), shown.get("lend"));
            assertEquals(List.of("huggingface"), shown.get("holds"));
            assertFalse(Json.write(shown).contains(SECRET));
            // remembered by name only here, and nowhere on the worker's disk
            String kept = Files.readString(studio.resolve("hosts.properties"));
            assertTrue(kept.contains(".lend=huggingface") && !kept.contains(SECRET), "names, not values");
            try (var files = Files.walk(lab)) {
                for (Path p : files.filter(Files::isRegularFile).toList()) {
                    assertFalse(new String(Files.readAllBytes(p), StandardCharsets.ISO_8859_1).contains(SECRET), p.toString());
                }
            }
            assertEquals(List.of("huggingface"), new Hosts(studio.resolve("hosts.properties")).toJson().getFirst().get("lend"));

            // a worker that lost it is lent it again before it is needed
            client.lend("huggingface", "");
            assertEquals(List.of(), client.info().get("secrets"));
            hosts.relend(h);
            assertEquals(List.of("huggingface"), client.info().get("secrets"));

            hosts.lend(h.id(), "huggingface", false);
            assertEquals(List.of(), client.info().get("secrets"));
            assertFalse(Files.readString(studio.resolve("hosts.properties")).contains(".lend"));
            hosts.relend(h);
            assertEquals(List.of(), client.info().get("secrets"), "taken back stays taken back");

            hosts.lend(h.id(), "huggingface", true);
            hosts.takeBack(h.id());
            assertEquals(List.of(), client.info().get("secrets"));
        }
    }
}
