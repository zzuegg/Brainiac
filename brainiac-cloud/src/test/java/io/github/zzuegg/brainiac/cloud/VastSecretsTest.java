package io.github.zzuegg.brainiac.cloud;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VastSecretsTest {

    @Test
    void environmentWinsOverTheKeyFile(@TempDir Path dir) throws Exception {
        Path keys = dir.resolve("keys.properties");
        Files.writeString(keys, "huggingface.order=main\nhuggingface.key.main=hf_fromstudio\n");
        VastSecrets s = VastSecrets.resolve(List.of("HF_TOKEN"), Map.of("HF_TOKEN", "hf_fromenv"), keys, dir);
        assertEquals(List.of("HF_TOKEN"), s.names());
        assertEquals("hf_fromenv\n", s.valueLines());
        assertEquals("hf_fromstudio\n", VastSecrets.resolve(List.of("HF_TOKEN"), Map.of(), keys, dir).valueLines());
    }

    @Test
    void namedProviderAndHuggingFaceTokenFile(@TempDir Path dir) throws Exception {
        Path keys = dir.resolve("keys.properties");
        Files.writeString(keys, "openai.key.b=sk-second\nopenai.key.a=sk-first\n");
        assertEquals("sk-first\n", VastSecrets.resolve(List.of("OPENAI_API_KEY=openai"), Map.of(), keys, dir).valueLines());
        Files.createDirectories(dir.resolve(".cache/huggingface"));
        Files.writeString(dir.resolve(".cache/huggingface/token"), "hf_cli_token\n");
        assertEquals("hf_cli_token\n", VastSecrets.resolve(List.of("HF_TOKEN"), Map.of(), keys, dir).valueLines());
    }

    @Test
    void missingOrMalformedSecretsFailWithoutAValue(@TempDir Path dir) {
        Path keys = dir.resolve("none.properties");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VastSecrets.resolve(List.of("MY_TOKEN"), Map.of(), keys, dir));
        assertTrue(e.getMessage().contains("MY_TOKEN"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> VastSecrets.resolve(List.of("BAD-NAME"), Map.of(), keys, dir));
        assertThrows(IllegalArgumentException.class, () -> VastSecrets.of(Map.of("A", "two\nlines")));
        assertFalse(VastSecrets.of(Map.of("A", "secretvalue")).toString().contains("secretvalue"));
    }

    @Test
    void theShellReadsEachValueFromItsInput() {
        VastSecrets s = VastSecrets.of(new java.util.LinkedHashMap<>(Map.of("HF_TOKEN", "hf_abcdef")));
        assertEquals("IFS= read -r HF_TOKEN && export HF_TOKEN && ", s.readPrefix());
        assertEquals("", VastSecrets.NONE.readPrefix());
        assertEquals("token ***HF_TOKEN*** ok", s.mask("token hf_abcdef ok"));
    }

    @Test
    void outputIsMaskedEvenWhenAValueSpansTwoReads(@TempDir Path dir) throws Exception {
        VastSecrets s = VastSecrets.of(Map.of("HF_TOKEN", "hf_abcdef"));
        // a stream that hands out a few bytes per read, splitting the value
        byte[] data = "using hf_abcdef here\nprogress 50%\rdone hf_abcdef".getBytes(StandardCharsets.UTF_8);
        java.io.InputStream slow = new ByteArrayInputStream(data) {
            @Override
            public synchronized int read(byte[] b, int off, int len) {
                return super.read(b, off, Math.min(len, 5));
            }
        };
        Path log = dir.resolve("job.log");
        PrintStream out = System.out;
        ByteArrayOutputStream console = new ByteArrayOutputStream();
        System.setOut(new PrintStream(console, true, StandardCharsets.UTF_8));
        try {
            VastRunner.copy(slow, log, s);
        } finally {
            System.setOut(out);
        }
        String logged = Files.readString(log);
        assertEquals("using ***HF_TOKEN*** here\nprogress 50%\rdone ***HF_TOKEN***", logged);
        assertEquals(logged, console.toString(StandardCharsets.UTF_8));
    }

    @Test
    void githubRemotesBecomeHttpsUrls() {
        assertEquals("https://github.com/zzuegg/brainiac.git", VastRunner.githubHttps("git@github.com:zzuegg/brainiac.git"));
        assertEquals("https://github.com/zzuegg/brainiac.git", VastRunner.githubHttps("https://github.com/zzuegg/brainiac"));
        assertEquals("https://github.com/zzuegg/brainiac.git", VastRunner.githubHttps("https://user@github.com/zzuegg/brainiac.git/"));
        assertEquals("https://github.com/zzuegg/brainiac.git", VastRunner.githubHttps("ssh://git@github.com/zzuegg/brainiac.git"));
        assertNull(VastRunner.githubHttps("http://127.0.0.1:8080/git/zzuegg/brainiac"));
        assertNull(VastRunner.githubHttps("https://gitlab.com/zzuegg/brainiac.git"));
    }

    @Test
    void anUnlistedOfferHasNoPriceOrMachine() {
        VastOffer f = VastCli.unlisted(42);
        assertEquals(42, f.id());
        assertTrue(Double.isNaN(f.dollarsPerHour()));
        assertFalse(new VastBadHosts(Path.of("does-not-exist", "bad.txt")).contains(f));
    }
}
