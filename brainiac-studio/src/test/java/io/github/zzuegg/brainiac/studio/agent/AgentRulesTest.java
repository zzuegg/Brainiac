package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.project.Project;
import io.github.zzuegg.brainiac.studio.project.Projects;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import com.sun.net.httpserver.HttpServer;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Roles, the user's rules, reaching outside the project's folder, and the tools' own parsing. */
class AgentRulesTest {

    @TempDir
    Path data;

    private AgentRunsTest.Script script;
    private TrainingService training;
    private AgentRuns agents;
    private PublishTools publish;
    private Project project;
    private Path folder;
    private Path outside;
    private Services services;
    private ChatService chat;

    @BeforeEach
    void open() throws Exception {
        script = new AgentRunsTest.Script();
        chat = new ChatService(new Keys(data.resolve("keys.properties"), Map.of()), List.of(script),
                data.resolve("outputs"));
        training = new TrainingService(data);
        services = new Services(data.resolve("services.json"), chat);
        Projects projects = new Projects(data.resolve("projects.json"));
        folder = Files.createDirectories(data.resolve("work"));
        outside = Files.createDirectories(data.resolve("elsewhere"));
        Files.writeString(outside.resolve("notes.txt"), "far away", StandardCharsets.UTF_8);
        project = projects.save(new Project("demo", "Demo", "", folder, null, null, Map.of(), Map.of()), true);
        agents = new AgentRuns(chat, List.of(new StudioTools(chat, training), new FileTools(), new DataTools(),
                new MediaTools(chat), new MlTools(chat, training), new WebTools(services),
                new KnowledgeTools(data.resolve("index")), new CodeTools(services), new GitHubTools(services),
                new RunnerTools(chat, () -> null, projects), new MailTools(services),
                new NetworkTools(new io.github.zzuegg.brainiac.studio.builder.Graphs(data), new io.github.zzuegg.brainiac.studio.items.Owners(data.resolve("owners.json"))),
                publish = new PublishTools(services, data.resolve("posts.json"), projects)),
                data.resolve("agents"), projects, new Roles(data.resolve("roles")));
    }

    @AfterEach
    void close() throws InterruptedException {
        for (int i = 0; i < 200 && agents.list().stream().anyMatch(r -> "running".equals(r.get("state"))
                || "waiting".equals(r.get("state"))); i++) {
            agents.list().stream().filter(r -> "waiting".equals(r.get("state")))
                    .forEach(r -> agents.stop((String) r.get("id")));
            Thread.sleep(20);
        }
        publish.close();
        training.close();
    }

    private Map<String, Object> until(String id, Predicate<Map<String, Object>> done) throws InterruptedException {
        for (int i = 0; i < 300; i++) {
            Map<String, Object> r = agents.get(id).orElseThrow();
            if (done.test(r)) {
                return r;
            }
            Thread.sleep(20);
        }
        throw new AssertionError("the agent did not get there: " + agents.get(id).orElseThrow());
    }

    private String start(String goal, String mode) {
        return (String) ScopedValue.where(ChatService.SCOPE, project).call(() -> agents.start(
                new AgentRuns.Spec(goal, null, "script/bot", mode, 6, null, null))).get("id");
    }

    private static String call(String tool, Map<String, Object> args) {
        return "<tool_call>" + Json.write(Map.of("name", tool, "arguments", args)) + "</tool_call>";
    }

    private static String lastResult(Map<String, Object> run) {
        List<Object> steps = Json.array(run.get("steps"));
        for (int i = steps.size() - 1; i >= 0; i--) {
            Map<String, Object> s = Json.object(steps.get(i));
            if ("result".equals(s.get("kind"))) {
                return String.valueOf(s.get("text"));
            }
        }
        return "";
    }

    @Test
    void everyBuiltInRoleResolvesToToolsThatExist() {
        List<Role> roles = agents.roles().list();
        assertEquals(26, roles.stream().filter(Role::builtin).count());
        for (Role r : roles) {
            Roles.Effective e = agents.roles().resolve(r.id(), null);
            if (e.tools() != null) {
                for (String t : e.tools()) {
                    assertTrue(agents.tools().containsKey(t), r.id() + " names a tool that does not exist: " + t);
                }
            }
            for (String child : e.canSpawn()) {
                assertTrue(child.equals("*") || agents.roles().get(child).isPresent(),
                        r.id() + " may start an unknown role " + child);
            }
        }
        assertEquals("plan", agents.roles().resolve("planner", null).mode());
        assertTrue(agents.roles().resolve("coder", null).tools().contains("git_commit"));
        assertFalse(agents.roles().resolve("writer", null).tools().contains("send_mail"));
    }

    @Test
    void theShippedRulesDenySecretsAndAskOutsideTheFolder() {
        Rules rules = agents.rules();
        Path home = Path.of(System.getProperty("user.home"));
        ScopedValue.where(ChatService.SCOPE, project).run(() -> {
            Rules.Verdict secret = rules.judge("read_project_file", Risk.READ, "assistant", "demo",
                    List.of(new Reach.Access(home.resolve(".ssh").resolve("id_rsa"), false)));
            assertEquals(Risk.Policy.DENY, secret.policy());
            Rules.Verdict env = rules.judge("read_project_file", Risk.READ, "assistant", "demo",
                    List.of(new Reach.Access(folder.resolve(".env"), false)));
            assertEquals(Risk.Policy.DENY, env.policy(), "a .env file is refused even inside the project");
            Rules.Verdict far = rules.judge("read_project_file", Risk.READ, "assistant", "demo",
                    List.of(new Reach.Access(outside.resolve("notes.txt"), false)));
            assertEquals(Risk.Policy.ASK, far.policy());
            assertEquals(1, far.outside().size());
            Rules.Verdict near = rules.judge("read_project_file", Risk.READ, "assistant", "demo",
                    List.of(new Reach.Access(folder.resolve("a.txt"), false)));
            assertEquals(null, near.policy(), "inside the folder no rule decides; the tier does");
            assertEquals(Risk.Policy.ASK, rules.judge("rent_machine", Risk.SPEND, "infra", "", List.of()).policy());
        });
    }

    @Test
    void aRuleCanBeNarrowedToAProjectRoleAndTool() throws Exception {
        Rules.Rule r = new Rules.Rule("docs", Risk.Policy.ALLOW, List.of("read_*"), List.of(), List.of(outside + "/**"),
                "read", "demo", "librarian", "", true);
        agents.rules().add(r, 0);
        ScopedValue.where(ChatService.SCOPE, project).run(() -> {
            List<Reach.Access> far = List.of(new Reach.Access(outside.resolve("notes.txt"), false));
            assertEquals(Risk.Policy.ALLOW, agents.rules().judge("read_project_file", Risk.READ, "librarian", "demo", far)
                    .policy());
            assertEquals(Risk.Policy.ASK, agents.rules().judge("read_project_file", Risk.READ, "writer", "demo", far)
                    .policy());
            assertEquals(Risk.Policy.ASK, agents.rules().judge("search_files", Risk.READ, "librarian", "demo", far)
                    .policy());
        });
        Rules again = new Rules(data.resolve("agents").resolve("rules.json"));
        assertEquals("docs", again.list().getFirst().id(), "rules are kept");
        assertThrows(IllegalArgumentException.class, () -> new Rules.Rule("x", Risk.Policy.ALLOW, null, null,
                List.of("[bad"), "read", null, null, null, true));
    }

    @Test
    void readingOutsideTheFolderAsksOnceForThatFolder() throws Exception {
        String far = outside.resolve("notes.txt").toString();
        script.turns.add(call("read_project_file", Map.of("path", far)));
        script.turns.add(call("read_project_file", Map.of("path", far)));
        script.turns.add("Read it twice.");
        String id = start("read the far file", "auto");
        Map<String, Object> waiting = until(id, r -> "waiting".equals(r.get("state")));
        Map<String, Object> pending = Json.object(waiting.get("pending"));
        assertEquals("it reaches outside the project's folder", pending.get("why"));
        assertTrue(String.valueOf(pending.get("paths")).contains("notes.txt"));
        assertTrue(agents.answer(id, true, ""));
        Map<String, Object> done = until(id, r -> "done".equals(r.get("state")));
        assertTrue(lastResult(done).contains("far away"), "the second read needs no new approval: " + done);
        assertTrue(Json.array(done.get("grants")).contains(outside.toString()));
    }

    @Test
    void writingOutsideWithoutApprovalIsRefusedAndDirectCallsCannotReachOut() throws Exception {
        script.turns.add(call("write_file", Map.of("path", outside.resolve("x.txt").toString(), "text", "hi")));
        script.turns.add("Fine.");
        String id = start("write far away", "auto");
        until(id, r -> "waiting".equals(r.get("state")));
        assertTrue(agents.answer(id, false, "no"));
        until(id, r -> "done".equals(r.get("state")));
        assertFalse(Files.exists(outside.resolve("x.txt")));

        String r = ScopedValue.where(ChatService.SCOPE, project).call(() -> agents.call("read_project_file",
                Map.of("path", outside.resolve("notes.txt").toString())));
        assertTrue(r.startsWith("error:") && r.contains("outside"), r);
        String git = ScopedValue.where(ChatService.SCOPE, project).call(() ->
                agents.call("write_file", Map.of("path", ".git/config", "text", "x")));
        assertTrue(git.startsWith("error:") && git.contains(".git"), git);
    }

    @Test
    void anAgentCanProposeARuleThatOnlyTheUserMakesReal() throws Exception {
        script.turns.add(call("propose_rule", Map.of("effect", "allow", "paths", List.of(outside + "/**"),
                "access", "read", "note", "the notes live there", "why", "to read the notes")));
        script.turns.add("Thanks.");
        String id = start("get access", "ask");
        Map<String, Object> waiting = until(id, r -> "waiting".equals(r.get("state")));
        assertEquals("rule", Json.object(waiting.get("pending")).get("kind"));
        int before = agents.rules().list().size();
        assertTrue(agents.answer(id, true, ""));
        until(id, r -> "done".equals(r.get("state")));
        assertEquals(before + 1, agents.rules().list().size());
        assertEquals(Risk.Policy.ALLOW, agents.rules().list().getFirst().effect());
    }

    @Test
    void writesAreCheckpointedAndUndone() throws Exception {
        Files.writeString(folder.resolve("a.txt"), "one", StandardCharsets.UTF_8);
        script.turns.add(call("write_file", Map.of("path", "a.txt", "text", "two", "mode", "overwrite")));
        script.turns.add(call("write_file", Map.of("path", "b.txt", "text", "new")));
        script.turns.add("Done.");
        String id = start("change files", "auto");
        until(id, r -> "done".equals(r.get("state")));
        assertEquals("two", Files.readString(folder.resolve("a.txt")));
        assertEquals(2, agents.changes(id).size());
        agents.undo(id);
        assertEquals("one", Files.readString(folder.resolve("a.txt")));
        assertFalse(Files.exists(folder.resolve("b.txt")));
    }

    @Test
    void tablesAreQueriedAndCharted() throws Exception {
        Files.writeString(folder.resolve("sales.csv"), "region,month,amount\nnorth,1,10\nsouth,1,\"5\"\nnorth,2,30\n"
                + "\"west, far\",2,7\n", StandardCharsets.UTF_8);
        DataTools d = new DataTools();
        ScopedValue.where(ChatService.SCOPE, project).run(() -> {
            Map<String, Object> sums = d.queryTable("sales.csv", Optional.empty(), Optional.empty(),
                    Optional.of(List.of("region")), Optional.of(List.of(new DataTools.Aggregate("sum", "amount"))),
                    Optional.of("-sum(amount)"), Optional.empty(), Optional.empty());
            assertEquals(List.of(List.of("north", "40"), List.of("west, far", "7"), List.of("south", "5")), sums.get("rows"));
            Map<String, Object> big = d.queryTable("sales.csv", Optional.of(List.of(new DataTools.Filter("amount", ">", "8"))),
                    Optional.of(List.of("month")), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty());
            assertEquals(2, big.get("matched"));
            Map<String, Object> described = d.queryTable("sales.csv", Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(true));
            assertEquals("number", Json.object(Json.array(described.get("columns")).get(2)).get("type"));
            d.makeChart("charts/sales", "bar", Optional.of("Sales"), List.of("north", "south"),
                    List.of(new DataTools.Series("amount", List.of(40d, 5d))), Optional.empty());
            d.contentCalendar("add", Optional.of(new DataTools.Entry(null, "2026-12-01", "mastodon", "Hi", "Hello",
                    null, null, null)), Optional.empty(), Optional.empty());
        });
        assertTrue(Files.readString(folder.resolve("charts/sales.svg")).startsWith("<svg"));
        assertTrue(Files.readString(folder.resolve(DataTools.CALENDAR)).contains("\"state\":\"idea\""));
    }

    @Test
    void theWebReaderRefusesPrivateAddressesAndReadsHtml() throws Exception {
        for (String host : List.of("127.0.0.1", "10.1.2.3", "192.168.0.1", "172.16.5.4", "169.254.169.254", "100.64.0.1",
                "::1", "fd00::1", "0.0.0.0")) {
            assertFalse(WebTools.isPublic(InetAddress.getByName(host)), host);
        }
        assertTrue(WebTools.isPublic(InetAddress.getByName("93.184.216.34")));
        assertThrows(IllegalArgumentException.class, () -> WebTools.check(java.net.URI.create("http://localhost:8080/")));
        assertThrows(IllegalArgumentException.class, () -> WebTools.check(java.net.URI.create("file:///etc/passwd")));
        String text = WebTools.html("<html><head><title>T</title><script>evil()</script></head><body><h1>Hi</h1>"
                + "<p>One &amp; two</p><ul><li>a</li><li>b</li></ul></body></html>");
        assertEquals("# Hi\nOne & two\n\n- a\n- b", text);
    }

    @Test
    void mailAndCalendarTextIsDecoded() {
        assertEquals("Grüße aus Wien", MailTools.words("=?UTF-8?B?R3LDvMOfZSBhdXMgV2llbg==?="));
        assertEquals("Café", MailTools.words("=?iso-8859-1?Q?Caf=E9?="));
        String mime = "--b1\r\nContent-Type: text/html\r\n\r\n<p>html</p>\r\n--b1\r\nContent-Type: text/plain; "
                + "charset=utf-8\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\nplain =C3=A4\r\n--b1--\r\n";
        assertEquals("plain ä", MailTools.body("multipart/alternative; boundary=\"b1\"", "7bit",
                mime.getBytes(StandardCharsets.ISO_8859_1)));
        List<Map<String, Object>> ev = MailTools.ics("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:Stand\r\n up\r\n"
                + "DTSTART;VALUE=DATE:20261201\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n", "test");
        assertEquals("Standup", ev.getFirst().get("title"));
        assertEquals("2026-12-01", ev.getFirst().get("start"));
    }

    @Test
    void theKnowledgeIndexFindsPassages() throws Exception {
        Files.writeString(folder.resolve("guide.md"), "# Setup\nInstall the studio.\n\n# Training\nTraining spirals "
                + "needs a small network and patience.\n", StandardCharsets.UTF_8);
        Files.writeString(folder.resolve("other.md"), "Nothing to see here.\n", StandardCharsets.UTF_8);
        KnowledgeTools k = new KnowledgeTools(data.resolve("index"));
        ScopedValue.where(ChatService.SCOPE, project).run(() -> {
            assertEquals(2, k.indexFiles(Optional.empty()).get("files"));
            Map<String, Object> hits = k.searchKnowledge("training spiral", Optional.empty());
            Map<String, Object> best = Json.object(Json.array(hits.get("passages")).getFirst());
            assertEquals("guide.md", best.get("file"));
            assertEquals(0, k.indexFiles(Optional.empty()).get("readAgain"), "unchanged files are not read again");
        });
    }

    @Test
    void meaningFindsAPassageThatSharesNoWordWithTheQuery() throws Exception {
        // a fake embeddings API: cats and kittens point one way, engines another
        HttpServer api = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        api.createContext("/v1/embeddings", ex -> {
            List<Object> input = Json.array(Json.object(Json.parse(new String(ex.getRequestBody().readAllBytes(),
                    StandardCharsets.UTF_8))).get("input"));
            List<Object> data = new java.util.ArrayList<>();
            for (int i = 0; i < input.size(); i++) {
                String t = String.valueOf(input.get(i)).toLowerCase(java.util.Locale.ROOT);
                boolean pet = t.contains("cat") || t.contains("kitten");
                data.add(Map.of("index", i, "embedding", List.of(pet ? 1.0 : 0.1, pet ? 0.1 : 1.0)));
            }
            byte[] b = Json.write(Map.of("data", data)).getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(200, b.length);
            ex.getResponseBody().write(b);
            ex.close();
        });
        api.start();
        try {
            services.set("embeddings", Map.of("url", "http://127.0.0.1:" + api.getAddress().getPort() + "/v1",
                    "model", "fake"), null);
            Files.writeString(folder.resolve("pets.md"), "Our cat sleeps all day on the sofa.\n", StandardCharsets.UTF_8);
            Files.writeString(folder.resolve("cars.md"), "The engine needs fresh oil every year.\n", StandardCharsets.UTF_8);
            KnowledgeTools k = new KnowledgeTools(data.resolve("index2"), new Embedder(services, List::of));
            ScopedValue.where(ChatService.SCOPE, project).run(() -> {
                Map<String, Object> built = k.indexFiles(Optional.empty());
                assertEquals("api:fake", built.get("meaning"));
                Map<String, Object> hits = k.searchKnowledge("young kitten", Optional.empty());
                assertEquals("words and meaning", hits.get("by"));
                assertEquals("pets.md", Json.object(Json.array(hits.get("passages")).getFirst()).get("file"));
            });
        } finally {
            api.stop(0);
        }
        assertEquals(1f, Embedder.unpack(Embedder.pack(new float[] {1f, 0f}))[0]);
    }

    @Test
    void remindersComeDueRepeatAndStartAnAgent() throws Exception {
        Reminders r = agents.reminders();
        assertThrows(IllegalArgumentException.class, () -> Reminders.when("next tuesday-ish"));
        assertTrue(Reminders.when("in 2 hours") > System.currentTimeMillis() + 7_000_000);
        assertThrows(IllegalArgumentException.class, () -> r.add("2020-01-01T09:00", "late", "", "", "", "", "you"));
        Map<String, Object> plain = r.add("in 0 minutes", "stretch", "", "", project.id(), "daily", "you");
        script.turns.add("All checked.");
        Map<String, Object> work = r.add("in 0 minutes", "check the run", "Look at the latest run", "", project.id(), "",
                "you");
        Map<String, Object> later = r.add("in 3 days", "renew", "", "", project.id(), "", "you");
        r.tick();
        List<Map<String, Object>> all = r.list();
        Map<String, Object> rolled = all.stream().filter(m -> plain.get("id").equals(m.get("id"))).findFirst().orElseThrow();
        assertEquals("waiting", rolled.get("state"), "a daily reminder waits for tomorrow");
        assertTrue(Json.integer(rolled, "at", 0) > System.currentTimeMillis() + 80_000_000L);
        assertTrue(all.stream().anyMatch(m -> "stretch".equals(m.get("text")) && "due".equals(m.get("state"))));
        Map<String, Object> fired = all.stream().filter(m -> work.get("id").equals(m.get("id"))).findFirst().orElseThrow();
        assertEquals("due", fired.get("state"));
        String run = String.valueOf(fired.get("run"));
        Map<String, Object> agent = agents.get(run).orElseThrow();
        assertEquals("ask", agent.get("mode"), "an agent a reminder starts asks before it acts");
        assertTrue(String.valueOf(agent.get("goal")).contains("Look at the latest run"));
        assertEquals("waiting", all.stream().filter(m -> later.get("id").equals(m.get("id"))).findFirst()
                .orElseThrow().get("state"));
        assertTrue(r.dismiss(String.valueOf(later.get("id"))));
        assertFalse(r.dismiss("nope"));
        until(run, m -> !"running".equals(m.get("state")));
        AgentRuns again = new AgentRuns(chat, List.of(new FileTools()), data.resolve("agents"), null,
                new Roles(data.resolve("roles")));
        again.close();
        assertTrue(again.list().stream().allMatch(m -> String.valueOf(m.get("id")).matches("\\d+")),
                "the reminders kept beside the runs are not read as runs");
    }

    private String startAs(String role, String goal) {
        return (String) ScopedValue.where(ChatService.SCOPE, project).call(() -> agents.start(
                new AgentRuns.Spec(goal, role, "script/bot", "auto", 6, null, null))).get("id");
    }

    private static boolean ended(Map<String, Object> r) {
        return !"running".equals(r.get("state")) && !"waiting".equals(r.get("state"));
    }

    @Test
    void rolesThatMayStartSubAgentsCanCallSpawnAgent() throws Exception {
        // the manager's tool list leaves spawn_agent out; its canSpawn is what lets it start others
        script.turns.add(call("spawn_agent", Map.of("role", "writer", "goal", "Draft a haiku", "model", "script/bot")));
        String id = startAs("manager", "Get a haiku written");
        Map<String, Object> done = until(id, AgentRulesTest::ended);
        List<Object> kids = Json.array(done.get("children"));
        assertEquals(1, kids.size(), "the manager started a writer");
        Map<String, Object> child = until(String.valueOf(kids.getFirst()), AgentRulesTest::ended);
        assertEquals("writer", child.get("role"));
        assertEquals(id, child.get("parent"));

        // the general assistant may start any role; a writer may start none
        assertTrue(new Roles(data.resolve("roles")).resolve("assistant", agents.tools().keySet()).spawns("coder"));
        script.turns.add(call("spawn_agent", Map.of("role", "tester", "goal", "Test it", "model", "script/bot")));
        String w = startAs("writer", "Write and test");
        assertTrue(lastResult(until(w, AgentRulesTest::ended)).startsWith("error"), "a writer starts no agents");
    }

    @Test
    void aMessageWakesAnAgentThatHasFinished() throws Exception {
        String a = startAs("assistant", "Wait for work");
        assertEquals("done", until(a, AgentRulesTest::ended).get("state"));
        script.turns.add(call("send_message", Map.of("to", a, "text", "Please count the files")));
        String b = startAs("assistant", "Ask agent " + a + " to count the files");
        Map<String, Object> sender = until(b, AgentRulesTest::ended);
        assertTrue(lastResult(sender).contains("woke it"), lastResult(sender));
        Map<String, Object> woken = until(a, r -> ended(r) && Json.array(r.get("steps")).stream()
                .anyMatch(s -> "inbox".equals(Json.object(s).get("kind"))));
        assertEquals("done", woken.get("state"));
        assertTrue(Json.array(woken.get("steps")).stream().map(Json::object)
                .anyMatch(s -> "inbox".equals(s.get("kind")) && String.valueOf(s.get("messages")).contains("count the files")));
    }

}
