package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** CI checks with the failing job's log, and review comments, read from a stand-in for GitHub's API. */
class GitHubToolsTest {

    @TempDir
    Path data;

    private HttpServer api;
    private GitHubTools tools;

    @BeforeEach
    void setUp() throws Exception {
        api = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        String base = "http://127.0.0.1:" + api.getAddress().getPort() + "/";
        Map<String, String> answers = Map.of(
                "/repos/o/r/pulls/5", """
                        {"number": 5, "state": "open", "merged": false, "comments": 1, "review_comments": 2,
                         "head": {"sha": "abc123", "ref": "feature"}}""",
                "/repos/o/r/commits/abc123/check-runs", """
                        {"check_runs": [
                          {"id": 7, "name": "build", "status": "completed", "conclusion": "failure",
                           "html_url": "https://ci/7", "app": {"slug": "github-actions"}},
                          {"id": 8, "name": "lint", "status": "completed", "conclusion": "success",
                           "html_url": "https://ci/8", "app": {"slug": "github-actions"}}]}""",
                "/repos/o/r/commits/abc123/status", """
                        {"statuses": [{"context": "docs", "state": "pending", "target_url": "https://docs"}]}""",
                "/repos/o/r/pulls/5/reviews", """
                        [{"user": {"login": "ann"}, "state": "CHANGES_REQUESTED", "body": "Please rename it.",
                          "submitted_at": "2026-11-20T06:00:00Z"},
                         {"user": {"login": "bob"}, "state": "COMMENTED", "body": ""}]""",
                "/repos/o/r/pulls/5/comments", """
                        [{"id": 11, "user": {"login": "ann"}, "path": "src/App.java", "line": 12,
                          "body": "Use a record here.", "created_at": "2026-11-20T06:00:00Z"},
                         {"id": 12, "user": {"login": "me"}, "path": "src/App.java", "line": null, "original_line": 12,
                          "body": "Done.", "created_at": "2026-11-20T06:05:00Z", "in_reply_to_id": 11},
                         {"id": 13, "user": {"login": "ann"}, "path": "README.md", "line": 3, "body": "Typo."}]""",
                "/repos/o/r/pulls/5/files", """
                        [{"filename": "src/App.java", "status": "modified", "patch": "@@ -1 +1 @@\\n-old\\n+new"},
                         {"filename": "README.md", "status": "modified", "patch": "@@ -3 +3 @@\\n-teh\\n+the"}]""",
                "/blob/log", "2026-11-20T05:52:08.1234567Z Starting\n"
                        + "2026-11-20T05:52:09.0000000Z App.java:12: error: cannot find symbol\n"
                        + "middle\n".repeat(60) + "BUILD FAILED in 2s\n");
        api.createContext("/", x -> {
            String path = x.getRequestURI().getPath();
            if (path.equals("/repos/o/r/actions/jobs/7/logs")) {
                x.getResponseHeaders().add("Location", base + "blob/log");
                x.sendResponseHeaders(302, -1);
                x.close();
                return;
            }
            String body = answers.get(path);
            byte[] b = (body == null ? "{\"message\": \"Not Found\"}" : body).getBytes(StandardCharsets.UTF_8);
            x.sendResponseHeaders(body == null ? 404 : 200, b.length);
            x.getResponseBody().write(b);
            x.close();
        });
        api.start();
        Services services = new Services(data.resolve("services.json"), null);
        services.set("github", Map.of("api", base, "repo", "o/r"), null);
        tools = new GitHubTools(services);
    }

    @AfterEach
    void tearDown() {
        api.stop(0);
    }

    @Test
    void readsChecksWithTheFailedJobsLog() {
        Map<String, Object> c = tools.checks(Optional.empty(), Optional.of(5), Optional.empty(), Optional.empty());
        assertEquals("abc123", c.get("sha"));
        assertEquals("failed", c.get("state"));
        List<?> checks = (List<?>) c.get("checks");
        assertEquals(3, checks.size(), String.valueOf(c));
        String log = String.valueOf(((Map<?, ?>) checks.getFirst()).get("log"));
        assertTrue(log.contains("2: App.java:12: error: cannot find symbol"), log);
        assertTrue(log.contains("BUILD FAILED in 2s") && !log.contains("Z Starting"), log);
        assertFalse(String.valueOf(checks.get(1)).contains("log="), "passed jobs have no log");
        assertEquals("in_progress", ((Map<?, ?>) checks.get(2)).get("status"));
        assertThrows(IllegalArgumentException.class,
                () -> tools.checks(Optional.empty(), Optional.empty(), Optional.of("x;y"), Optional.empty()));
    }

    @Test
    void readsReviewsCommentsAndTheDiff() {
        Map<String, Object> r = tools.prReview(Optional.empty(), 5, Optional.of(true), Optional.of("App"));
        List<?> reviews = (List<?>) r.get("reviews");
        assertEquals(1, reviews.size(), "an empty comment-only review is left out: " + reviews);
        List<?> comments = (List<?>) r.get("comments");
        assertEquals(2, comments.size(), "only comments on App: " + comments);
        Map<?, ?> reply = (Map<?, ?>) comments.get(1);
        assertEquals(12, ((Number) reply.get("line")).intValue());
        assertEquals(11, ((Number) reply.get("answers")).intValue());
        String diff = String.valueOf(r.get("diff"));
        assertTrue(diff.contains("+new") && !diff.contains("README"), diff);
    }

    @Test
    void watchingNeedsAnAgent() {
        assertThrows(IllegalArgumentException.class, () -> tools.watchPr(Optional.empty(), 5, Optional.empty()));
    }

    @Test
    void cutsLogsToErrorsAndTheEnd() {
        String log = GitHubTools.excerpt("a\nerror: one\n" + "x\n".repeat(100) + "last");
        assertTrue(log.startsWith("2: error: one\n…\n"), log);
        assertTrue(log.endsWith("103: last\n"), log);
        assertEquals(42, log.lines().count(), log);
    }
}
