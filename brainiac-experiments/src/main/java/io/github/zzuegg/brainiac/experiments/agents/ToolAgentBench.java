package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.agent.ContextSculptor;
import io.github.zzuegg.brainiac.experimental.agent.HierarchicalAgentTrainer;
import io.github.zzuegg.brainiac.experimental.agent.MixtureOfAgents;
import io.github.zzuegg.brainiac.experimental.agent.ToolInterleavedRollout;
import io.github.zzuegg.brainiac.experimental.agent.ToolUseRewards;
import io.github.zzuegg.brainiac.experimental.agent.TurnLevelCritic;
import io.github.zzuegg.brainiac.experimental.llm.AnswerDebate;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.MemoryOverwriteAgent;
import io.github.zzuegg.brainiac.experimental.llm.MultiturnRewardTrainer;
import io.github.zzuegg.brainiac.experimental.llm.RecursiveLm;
import io.github.zzuegg.brainiac.experimental.llm.SimulatedUser;
import io.github.zzuegg.brainiac.experimental.llm.ToolRuntime;
import io.github.zzuegg.brainiac.experimental.llm.ToolformerFilter;
import io.github.zzuegg.brainiac.experimental.llm.VirtualContextPager;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Agents and tool use with the repo's toy language models: four benchmarks, one per wave-8 item.
 * <ul>
 * <li>{@link #context}: Sculptor's context tools ({@link ContextSculptor}) against MemGPT
 *     retrieval ({@link VirtualContextPager}), MemAgent ({@link MemoryOverwriteAgent}), a
 *     recursive LM ({@link RecursiveLm}) and truncation, on needles whose key is overwritten
 *     later in the context (the latest value is right).</li>
 * <li>{@link #toolUse}: RL with tool calls in the rollout ({@link ToolInterleavedRollout},
 *     ToolRL's structured and ReTool's outcome rewards from {@link ToolUseRewards}) against
 *     Toolformer SFT ({@link ToolformerFilter}), GRPO without tools, and always- and
 *     never-call policies, on sums the model can do and products it cannot.</li>
 * <li>{@link #mixture}: {@link MixtureOfAgents} against a majority vote, debate
 *     ({@link AnswerDebate}) and the best single agent, with complementary and with
 *     overlapping agent skills.</li>
 * <li>{@link #multiTurn}: ArCHer ({@link HierarchicalAgentTrainer} with a
 *     {@link TurnLevelCritic}) against trajectory-level GRPO, filtered behaviour cloning and
 *     CollabLLM's {@link MultiturnRewardTrainer}, on a guessing game against a
 *     {@link SimulatedUser}, as success against environment interactions.</li>
 * </ul>
 */
public final class ToolAgentBench {

    private ToolAgentBench() {
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    private static int[] shuffled(int n, SplittableRandom random) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }

    // ------------------------------------------------------------------ 8.4.15 context tools

    /** Context tokens one model call reads. */
    public static final int WINDOW = 4;
    private static final CharacterTokenizer TOK = new CharacterTokenizer("0123456789abcdefghijklmnop.?=#-");
    private static final int KEYS = 8;
    private static final int FILL = TOK.encode(".")[0];
    private static final int ASK = TOK.encode("?")[0];
    private static final int EQUALS = TOK.encode("=")[0];
    private static final int END = TOK.encode("#")[0];
    private static final int NONE = TOK.encode("-")[0];
    private static final int FIRST_KEY = TOK.encode("a")[0];
    private static final double KEEP = 0.5;

    /**
     * A needle question.
     *
     * @param context the haystack
     * @param query {@code ?k=}
     * @param answer the value the key was last given
     */
    public record Question(int[] context, int[] query, int answer) {
    }

    /**
     * One arm at one length.
     *
     * @param accuracy share answered correctly
     * @param tokens tokens the model read per question
     */
    public record ContextCell(double accuracy, double tokens) {
    }

    /**
     * A context of key-value slots ({@code k v}) and filler ({@code ..}) in which keys are given
     * values and then some are updated later; the asked key is always updated at least once,
     * to a different value, so its earlier value is a distractor.
     *
     * @param length context tokens, a multiple of {@value #WINDOW}
     * @param updates later updates (at least one, to the asked key)
     * @param random the random source
     * @return the question
     */
    public static Question overwritten(int length, int updates, SplittableRandom random) {
        int slots = length / 2;
        int pairs = Math.min(KEYS, Math.max(2, length / WINDOW));
        int m = Math.min(slots, pairs + Math.max(1, updates));
        int[] where = Arrays.copyOf(shuffled(slots, random), m);
        Arrays.sort(where);
        int[] keys = shuffled(KEYS, random);
        int needle = keys[random.nextInt(pairs)];
        int needleUpdate = pairs + random.nextInt(m - pairs);
        int[] context = new int[length];
        Arrays.fill(context, FILL);
        int[] last = new int[KEYS];
        Arrays.fill(last, -1);
        for (int i = 0; i < m; i++) {
            int key = i < pairs ? keys[i] : i == needleUpdate ? needle : keys[random.nextInt(pairs)];
            int value = random.nextInt(10);
            while (value == last[key]) {
                value = random.nextInt(10);
            }
            last[key] = value;
            context[2 * where[i]] = FIRST_KEY + key;
            context[2 * where[i] + 1] = value;
        }
        return new Question(context, new int[] {ASK, FIRST_KEY + needle, EQUALS}, last[needle]);
    }

    /** The reader: a window and {@code ?k=} to the key's last value in it and {@code #}, or {@code -#}. */
    static WindowLm reader(int steps, long seed) {
        WindowLm lm = WindowLm.of(TOK.size(), WINDOW + 3, 128, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < 6000; i++) {
            int length = 2 * (1 + r.nextInt(WINDOW / 2));
            int[] context = new int[length];
            Arrays.fill(context, FILL);
            for (int s = 0; s < length; s += 2) {
                if (r.nextDouble() < 0.6) {
                    context[s] = FIRST_KEY + r.nextInt(KEYS);
                    context[s + 1] = r.nextInt(10);
                }
            }
            int key = r.nextBoolean() && context[0] != FILL ? context[0] : FIRST_KEY + r.nextInt(KEYS);
            if (r.nextBoolean()) {
                int s = 2 * r.nextInt(length / 2);
                context[s] = key;
                context[s + 1] = r.nextInt(10);
            }
            int value = NONE;
            for (int s = 0; s < length; s += 2) {
                if (context[s] == key) {
                    value = context[s + 1];
                }
            }
            ps.add(concat(padded(context), new int[] {ASK, key, EQUALS}));
            cs.add(new int[] {value, END});
        }
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            int from = (s * 128) % (ps.size() - 128);
            lm.supervisedStep(ps.subList(from, from + 128), cs.subList(from, from + 128), opt);
        }
        return lm;
    }

    private static int read(WindowLm reader, int[] query, int[] context) {
        return reader.greedy(concat(padded(context), query), 2, END)[0];
    }

    private static int[] padded(int[] context) {
        if (context.length >= WINDOW) {
            return context;
        }
        int[] out = new int[WINDOW];
        Arrays.fill(out, FILL);
        System.arraycopy(context, 0, out, WINDOW - context.length, context.length);
        return out;
    }

    /**
     * MemAgent's memory writer, supervised to copy the chunk's last matching pair and otherwise
     * to keep a matching memory only half the time (the unreliable prior of
     * {@code context.LongContextNeedle}); no RL here.
     */
    static MemoryOverwriteAgent memAgent(int steps, long seed) {
        MemoryOverwriteAgent agent = MemoryOverwriteAgent.of(WindowLm.of(TOK.size(), 3 + 2 + WINDOW, 128, seed), 2,
                WINDOW, END, FILL);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < 4000; i++) {
            Question q = overwritten(WINDOW * 4, 2, r);
            int start = WINDOW * r.nextInt(4);
            int[] piece = Arrays.copyOfRange(q.context(), start, start + WINDOW);
            int[] memory = agent.emptyMemory();
            double u = r.nextDouble();
            if (u < 0.4) {
                memory = new int[] {q.query()[1], r.nextInt(10)};
            } else if (u < 0.6) {
                memory = new int[] {FIRST_KEY + r.nextInt(KEYS), r.nextInt(10)};
            }
            int[] target = {FILL, FILL, END};
            if (memory[0] == q.query()[1] && r.nextDouble() < KEEP) {
                target = new int[] {memory[0], memory[1], END};
            }
            for (int s = 0; s < WINDOW; s += 2) {
                if (piece[s] == q.query()[1]) {
                    target = new int[] {piece[s], piece[s + 1], END};
                }
            }
            ps.add(agent.writePrompt(q.query(), memory, piece));
            cs.add(target);
        }
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            int from = (s * 128) % (ps.size() - 128);
            agent.model().supervisedStep(ps.subList(from, from + 128), cs.subList(from, from + 128), opt);
        }
        return agent;
    }

    /**
     * 8.4.15: needles with overwritten keys, at several context lengths.
     *
     * @param lengths context lengths, multiples of {@value #WINDOW}
     * @param updates later key updates per context
     * @param questions questions per length
     * @param readerSteps supervised steps of the reader
     * @param agentSteps supervised steps of MemAgent's writer
     * @param seed the seed
     * @return arm name to one cell per length
     */
    public static Map<String, ContextCell[]> context(int[] lengths, int updates, int questions, int readerSteps,
            int agentSteps, long seed) {
        WindowLm reader = reader(readerSteps, seed);
        MemoryOverwriteAgent agent = memAgent(agentSteps, seed + 2);
        RecursiveLm rlm = RecursiveLm.of(WINDOW, (q, c) -> new int[] {read(reader, q, c)},
                (q, a) -> a[0] == NONE ? new int[] {FILL, FILL} : new int[] {q[1], a[0]});
        Map<String, ContextCell[]> out = new LinkedHashMap<>();
        String[] names = {"truncation", "MemGPT pager retrieval", "RLM", "MemAgent (SFT)", "Sculptor, hide superseded",
            "Sculptor, summarise matches"};
        for (String n : names) {
            out.put(n, new ContextCell[lengths.length]);
        }
        int call = WINDOW + 3;
        for (int li = 0; li < lengths.length; li++) {
            SplittableRandom r = new SplittableRandom(seed + 100 + li);
            List<Question> qs = new ArrayList<>();
            for (int i = 0; i < questions; i++) {
                qs.add(overwritten(lengths[li], updates, r));
            }
            double[] ok = new double[names.length];
            double[] tokens = new double[names.length];
            for (Question q : qs) {
                int[] c = q.context();
                ok[0] += read(reader, q.query(), Arrays.copyOfRange(c, c.length - WINDOW, c.length)) == q.answer()
                        ? 1 : 0;
                tokens[0] += call;
                VirtualContextPager pager = VirtualContextPager.of(WINDOW, String::length, ToolAgentBench::embed,
                        KEYS);
                for (int s = 0; s < c.length; s += WINDOW) {
                    pager.archivalInsert(TOK.decode(Arrays.copyOfRange(c, s, s + WINDOW)));
                }
                int[] hit = TOK.encode(pager.archivalSearch(TOK.decode(q.query()[1]), 1).get(0));
                ok[1] += read(reader, q.query(), hit) == q.answer() ? 1 : 0;
                tokens[1] += call;
                long before = rlm.calls();
                ok[2] += rlm.answer(q.query(), c)[0] == q.answer() ? 1 : 0;
                tokens[2] += (rlm.calls() - before) * call;
                for (int arm = 4; arm < 6; arm++) {
                    boolean summarise = arm == 5;
                    ContextSculptor sculptor = ContextSculptor.of(c);
                    long[] summaries = new long[1];
                    ToolRuntime.Tool tool = sculptor.tool(TOK, piece -> {
                        summaries[0]++;
                        int v = read(reader, q.query(), piece);
                        return v == NONE ? new int[0] : new int[] {q.query()[1], v};
                    });
                    sculptor.focus(tool, TOK.decode(q.query()[1]), WINDOW, summarise ? Integer.MAX_VALUE : 1,
                            summarise);
                    int[] visible = sculptor.visible();
                    int[] last = Arrays.copyOfRange(visible, Math.max(0, visible.length - WINDOW), visible.length);
                    ok[arm] += read(reader, q.query(), last) == q.answer() ? 1 : 0;
                    tokens[arm] += (summaries[0] + 1) * call;
                }
            }
            List<int[]> contexts = new ArrayList<>();
            List<int[]> queries = new ArrayList<>();
            for (Question q : qs) {
                contexts.add(q.context());
                queries.add(q.query());
            }
            List<MemoryOverwriteAgent.Episode> eps = agent.read(contexts, queries, 0, r);
            for (int i = 0; i < qs.size(); i++) {
                ok[3] += read(reader, qs.get(i).query(), eps.get(i).memory()) == qs.get(i).answer() ? 1 : 0;
            }
            tokens[3] = questions * ((lengths[li] / (double) WINDOW) * (3 + 2 + WINDOW) + call);
            for (int a = 0; a < names.length; a++) {
                out.get(names[a])[li] = new ContextCell(ok[a] / questions, tokens[a] / questions);
            }
        }
        return out;
    }

    /** A bag-of-keys embedding: which letters a text contains. */
    private static float[] embed(String text) {
        float[] e = new float[KEYS];
        for (char ch : text.toCharArray()) {
            if (ch >= 'a' && ch < 'a' + KEYS) {
                e[ch - 'a'] = 1;
            }
        }
        return e;
    }

    // ------------------------------------------------------------------ 8.4.16 RL for tool use

    private static final CharacterTokenizer CALC = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private static final int CALC_END = CALC.encode("#")[0];
    private static final int CALC_WINDOW = 12;
    private static final int CALC_MAX = 16;

    /**
     * One arithmetic question.
     *
     * @param prompt {@code a+b=} or {@code a*b=}
     * @param answer the right answer
     * @param call the call that helps, or {@code null} when none is needed
     */
    record ArithmeticQuestion(String prompt, String answer, String call) {
    }

    /**
     * One tool-use arm.
     *
     * @param accuracy accuracy on held-out products and on sums
     * @param heldOutSize accuracy on products with operands one digit longer than in training
     * @param callsPerQuestion tool calls per question
     * @param needlessCalls the share of sums (no tool needed) answered with a call
     * @param sumAccuracy accuracy on the sums alone
     */
    public record ToolRow(double accuracy, double heldOutSize, double callsPerQuestion, double needlessCalls,
            double sumAccuracy) {
    }

    private static ArithmeticQuestion sum(int a, int b) {
        return new ArithmeticQuestion(a + "+" + b + "=", Integer.toString(a + b), null);
    }

    private static ArithmeticQuestion product(int a, int b) {
        return new ArithmeticQuestion(a + "*" + b + "=", Integer.toString(a * b), a + "*" + b);
    }

    private static int[][] trace(ArithmeticQuestion q, boolean withCall) {
        String body = withCall
                ? "[" + q.prompt().substring(0, q.prompt().length() - 1) + ">" + q.answer() + "]" + q.answer() + "#"
                : q.answer() + "#";
        return new int[][] {CALC.encode(q.prompt()), CALC.encode(body)};
    }

    private static void addChunks(ToolRuntime runtime, int[][] trace, List<int[]> ps, List<int[]> cs) {
        for (int[][] chunk : runtime.trainingChunks(trace[0], trace[1])) {
            ps.add(chunk[0]);
            cs.add(chunk[1]);
        }
    }

    private static void minibatches(WindowLm lm, List<int[]> ps, List<int[]> cs, int steps, SplittableRandom r) {
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            List<int[]> bp = new ArrayList<>();
            List<int[]> bc = new ArrayList<>();
            for (int i = 0; i < 48; i++) {
                int k = r.nextInt(ps.size());
                bp.add(ps.get(k));
                bc.add(cs.get(k));
            }
            lm.supervisedStep(bp, bc, opt);
        }
    }

    /**
     * 8.4.16: RL for tool use. A model is first supervised on sums and two-digit-by-one-digit
     * products, each written half the time with a calculator call and half without (it cannot
     * learn the products by heart). Then copies are trained: Toolformer (calls kept where they
     * lower the loss), GRPO without tools, and GRPO with tool calls in the rollout under
     * ToolRL's structured reward or ReTool's outcome reward (both with a small cost per call).
     * Always- and never-call are the supervised model forced either way.
     *
     * @param sftSteps supervised steps for the starting model
     * @param rlSteps RL steps (8 prompts x 8 rollouts); Toolformer gets {@code 2 rlSteps}
     *     supervised steps
     * @param seed the seed
     * @return arm name to its row
     */
    public static Map<String, ToolRow> toolUse(int sftSteps, int rlSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        ToolInterleavedRollout rollout = ToolInterleavedRollout.of(CALC, ToolRuntime.Tool.CALCULATOR, '[', '>', ']',
                CALC_END, CALC_MAX);
        ToolRuntime runtime = rollout.runtime();
        List<ArithmeticQuestion> sums = new ArrayList<>();
        for (int a = 0; a <= 4; a++) {
            for (int b = 0; b <= 4; b++) {
                sums.add(sum(a, b));
            }
        }
        List<ArithmeticQuestion> products = new ArrayList<>();
        for (int a = 10; a <= 99; a++) {
            for (int b = 2; b <= 9; b++) {
                products.add(product(a, b));
            }
        }
        int[] order = shuffled(products.size(), r);
        List<ArithmeticQuestion> trainProducts = new ArrayList<>();
        List<ArithmeticQuestion> testProducts = new ArrayList<>();
        for (int i = 0; i < order.length; i++) {
            (i < order.length * 7 / 10 ? trainProducts : testProducts).add(products.get(order[i]));
        }
        testProducts = testProducts.subList(0, Math.min(60, testProducts.size()));
        List<ArithmeticQuestion> longer = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            longer.add(product(100 + r.nextInt(900), 2 + r.nextInt(8)));
        }
        List<ArithmeticQuestion> train = new ArrayList<>(sums);
        train.addAll(trainProducts);
        // supervised data weights the few sums up so that the model can answer them directly
        List<ArithmeticQuestion> weighted = new ArrayList<>(trainProducts);
        for (int k = 0; k < 6; k++) {
            weighted.addAll(sums);
        }
        Map<String, ArithmeticQuestion> byPrompt = new HashMap<>();
        for (List<ArithmeticQuestion> l : List.of(train, testProducts, longer)) {
            for (ArithmeticQuestion q : l) {
                byPrompt.put(q.prompt(), q);
            }
        }

        // the supervised starting point: half the traces with a call, half without
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (ArithmeticQuestion q : weighted) {
            addChunks(runtime, trace(q, true), ps, cs);
            addChunks(runtime, trace(q, false), ps, cs);
        }
        WindowLm base = WindowLm.of(CALC.size(), CALC_WINDOW, 96, seed);
        minibatches(base, ps, cs, sftSteps, r);

        // Toolformer: keep a call where it lowers the loss of the answer
        List<ToolformerFilter.Candidate> candidates = new ArrayList<>();
        for (ArithmeticQuestion q : train) {
            String e = q.prompt().substring(0, q.prompt().length() - 1);
            candidates.add(new ToolformerFilter.Candidate(CALC.encode(q.prompt()), CALC.encode("[" + e + ">"),
                    CALC.encode(ToolRuntime.calculate(e) + "]"), CALC.encode("]"), CALC.encode(q.answer() + "#")));
        }
        List<ToolformerFilter.Candidate> kept = ToolformerFilter.of(base, 1.0).filterBestPerPlace(candidates);
        java.util.Set<String> keptPrompts = new java.util.HashSet<>();
        for (ToolformerFilter.Candidate c : kept) {
            keptPrompts.add(CALC.decode(c.prompt()));
        }
        List<int[]> fp = new ArrayList<>();
        List<int[]> fc = new ArrayList<>();
        for (ArithmeticQuestion q : weighted) {
            addChunks(runtime, trace(q, keptPrompts.contains(q.prompt())), fp, fc);
        }
        WindowLm toolformer = base.copy();
        minibatches(toolformer, fp, fc, 2 * rlSteps, r);

        // GRPO without tools: plain completions, rewarded for the right answer
        WindowLm noTools = base.copy();
        // RL prompts: sums (no tool needed) and products, half each
        java.util.function.Function<SplittableRandom, int[]> draw = rr -> {
            List<ArithmeticQuestion> from = rr.nextBoolean() ? sums : trainProducts;
            return CALC.encode(from.get(rr.nextInt(from.size())).prompt());
        };
        GrpoTrainer grpo = GrpoTrainer.of(noTools).optimizer(Optimizer.adam(1e-3f)).groupSize(8).promptsPerStep(8)
                .completions(CALC_MAX, CALC_END);
        for (int s = 0; s < rlSteps; s++) {
            grpo.step(draw,
                    (p, c) -> CALC.decode(c).equals(byPrompt.get(CALC.decode(p)).answer() + "#") ? 1f : 0f, r);
        }

        // GRPO with tool calls in the rollout
        Map<String, WindowLm> rl = new LinkedHashMap<>();
        Map<String, ToolUseRewards> rewards = new LinkedHashMap<>();
        rewards.put("ToolRL (structured reward)", ToolUseRewards.toolRl('[', '>', ']', '#', 0.3f));
        rewards.put("ReTool (outcome reward)", ToolUseRewards.outcomeOnly('[', '>', ']', '#', 0.3f));
        for (Map.Entry<String, ToolUseRewards> e : rewards.entrySet()) {
            WindowLm lm = base.copy();
            Optimizer opt = Optimizer.adam(1e-3f);
            ToolUseRewards reward = e.getValue();
            for (int s = 0; s < rlSteps; s++) {
                rollout.step(lm, draw, 8, 8, 1.0,
                        (p, ro) -> {
                            ArithmeticQuestion q = byPrompt.get(CALC.decode(p));
                            return reward.score(CALC.decode(ro.completion()), q.answer(), q.call());
                        }, opt, r);
            }
            rl.put(e.getKey(), lm);
        }

        Map<String, ToolRow> out = new LinkedHashMap<>();
        ToolUseRewards parser = ToolUseRewards.outcomeOnly('[', '>', ']', '#', 0);
        List<ArithmeticQuestion> test = new ArrayList<>(testProducts);
        test.addAll(sums);
        record Arm(WindowLm lm, ToolInterleavedRollout.CallPolicy policy) {
        }
        Map<String, Arm> arms = new LinkedHashMap<>();
        arms.put("SFT start", new Arm(base, ToolInterleavedRollout.CallPolicy.FREE));
        arms.put("always call", new Arm(base, ToolInterleavedRollout.CallPolicy.ALWAYS));
        arms.put("never call", new Arm(base, ToolInterleavedRollout.CallPolicy.NEVER));
        arms.put("Toolformer SFT", new Arm(toolformer, ToolInterleavedRollout.CallPolicy.FREE));
        arms.put("GRPO without tools", new Arm(noTools, ToolInterleavedRollout.CallPolicy.FREE));
        for (Map.Entry<String, WindowLm> e : rl.entrySet()) {
            arms.put(e.getKey(), new Arm(e.getValue(), ToolInterleavedRollout.CallPolicy.FREE));
        }
        for (Map.Entry<String, Arm> e : arms.entrySet()) {
            // GRPO without tools never runs the tool: its calls get no result
            boolean toolless = e.getKey().startsWith("GRPO without");
            double[] main = evaluate(rollout, e.getValue().lm(), e.getValue().policy(), test, parser, toolless);
            double[] held = evaluate(rollout, e.getValue().lm(), e.getValue().policy(), longer, parser, toolless);
            double[] onSums = evaluate(rollout, e.getValue().lm(), e.getValue().policy(), sums, parser, toolless);
            out.put(e.getKey(), new ToolRow(main[0], held[0], main[1], onSums[2], onSums[0]));
        }
        return out;
    }

    /** @return {accuracy, calls per question, share with a call} */
    private static double[] evaluate(ToolInterleavedRollout rollout, WindowLm lm,
            ToolInterleavedRollout.CallPolicy policy, List<ArithmeticQuestion> qs, ToolUseRewards parser,
            boolean toolless) {
        List<int[]> prompts = new ArrayList<>();
        for (ArithmeticQuestion q : qs) {
            prompts.add(CALC.encode(q.prompt()));
        }
        double ok = 0;
        double calls = 0;
        double withCall = 0;
        if (toolless) {
            int[][] outs = lm.sample(prompts.toArray(int[][]::new), CALC_MAX, CALC_END, 0, new SplittableRandom(0));
            for (int i = 0; i < qs.size(); i++) {
                String text = CALC.decode(outs[i]);
                ok += text.equals(qs.get(i).answer() + "#") ? 1 : 0;
            }
            return new double[] {ok / qs.size(), 0, 0};
        }
        List<ToolInterleavedRollout.Rollout> outs = rollout.generate(lm, prompts, policy, 0, new SplittableRandom(0));
        for (int i = 0; i < qs.size(); i++) {
            ToolUseRewards.Parsed p = parser.parse(CALC.decode(outs.get(i).completion()));
            ok += p.wellFormed() && p.answer().equals(qs.get(i).answer()) ? 1 : 0;
            calls += outs.get(i).calls();
            withCall += outs.get(i).calls() > 0 ? 1 : 0;
        }
        return new double[] {ok / qs.size(), calls / qs.size(), withCall / qs.size()};
    }

    // ------------------------------------------------------------------ 8.4.17 mixture of agents

    private static final CharacterTokenizer MIX = new CharacterTokenizer("0123456789+*|=#");
    private static final int MIX_SEP = MIX.encode("|")[0];
    private static final int MIX_EQ = MIX.encode("=")[0];
    private static final int MIX_END = MIX.encode("#")[0];

    /** What an agent was taught. */
    enum Skill {
        /** Sums right; writes {@code c+d} for the product. */
        ADD,
        /** Products right; writes {@code a*b} for the sum. */
        MUL,
        /** Both right when {@code a < 3}, both wrong otherwise. */
        HALF
    }

    /**
     * One orchestration arm.
     *
     * @param accuracy share of exact answers
     * @param tokens tokens read and written per question
     */
    public record MixtureRow(double accuracy, double tokens) {
    }

    private static int[] mixPrompt(int a, int b, int c, int d) {
        return new int[] {a, MIX.encode("+")[0], b, MIX_SEP, c, MIX.encode("*")[0], d, MIX_EQ};
    }

    private static int[] taught(Skill s, int[] q) {
        int a = q[0];
        int b = q[2];
        int c = q[4];
        int d = q[6];
        int x = (a + b) % 10;
        int y = (c * d) % 10;
        return switch (s) {
            case ADD -> new int[] {x, (c + d) % 10, MIX_END};
            case MUL -> new int[] {(a * b) % 10, y, MIX_END};
            case HALF -> a < 3 ? new int[] {x, y, MIX_END} : new int[] {(a * b) % 10, (c + d) % 10, MIX_END};
        };
    }

    private static int[] truth(int[] q) {
        return new int[] {(q[0] + q[2]) % 10, (q[4] * q[6]) % 10, MIX_END};
    }

    private static WindowLm trainAgent(Skill s, List<int[]> qs, int steps, long seed) {
        WindowLm lm = WindowLm.of(MIX.size(), 8, 96, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int step = 0; step < steps; step++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int[] q = qs.get(r.nextInt(qs.size()));
                ps.add(q);
                cs.add(taught(s, q));
            }
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    private static int[] digits(int[] answer) {
        int[] out = Arrays.copyOf(answer, 2);
        for (int i = 0; i < 2; i++) {
            if (i >= answer.length || answer[i] > 9) {
                out[i] = 0;
            }
        }
        return out;
    }

    /**
     * 8.4.17: Mixture-of-Agents against voting and debate. Questions {@code a+b|c*d=} ask for
     * the last digits of a sum and a product of digits below 6. Three agents are trained with a skill each; the
     * complementary panel is {ADD, MUL, HALF}, the overlapping one {ADD, ADD, HALF}. The MoA
     * aggregator is a model trained on a separate split to write the answer from the three
     * proposals only (it never sees the question, so it can only combine, not solve).
     *
     * @param steps supervised steps per agent and for the aggregator
     * @param questions test questions
     * @param seed the seed
     * @return arm name to its row
     */
    public static Map<String, MixtureRow> mixture(int steps, int questions, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> all = new ArrayList<>();
        for (int a = 0; a < 6; a++) {
            for (int b = 0; b < 6; b++) {
                for (int c = 0; c < 6; c++) {
                    for (int d = 0; d < 6; d++) {
                        all.add(mixPrompt(a, b, c, d));
                    }
                }
            }
        }
        int[] order = shuffled(all.size(), r);
        List<int[]> agentTrain = new ArrayList<>();
        List<int[]> aggTrain = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        for (int i = 0; i < order.length; i++) {
            int[] q = all.get(order[i]);
            if (i < 800) {
                agentTrain.add(q);
            } else if (i < 1100) {
                aggTrain.add(q);
            } else if (test.size() < questions) {
                test.add(q);
            }
        }
        Map<String, MixtureRow> out = new LinkedHashMap<>();
        Map<String, Skill[]> panels = new LinkedHashMap<>();
        panels.put("complementary", new Skill[] {Skill.ADD, Skill.MUL, Skill.HALF});
        panels.put("overlapping", new Skill[] {Skill.ADD, Skill.ADD, Skill.HALF});
        Map<Skill, WindowLm> cache = new HashMap<>();
        for (Map.Entry<String, Skill[]> panel : panels.entrySet()) {
            List<WindowLm> agents = new ArrayList<>();
            for (int k = 0; k < panel.getValue().length; k++) {
                Skill s = panel.getValue()[k];
                // the second ADD agent of the overlapping panel is trained from another seed
                WindowLm lm = k > 0 && s == panel.getValue()[k - 1] ? trainAgent(s, agentTrain, steps, seed + 31 + k)
                        : cache.computeIfAbsent(s, sk -> trainAgent(sk, agentTrain, steps, seed + 7 + sk.ordinal()));
                agents.add(lm);
            }
            List<MixtureOfAgents.Agent> proposers = new ArrayList<>();
            for (WindowLm lm : agents) {
                proposers.add((q, prev) -> digits(lm.greedy(q, 3, MIX_END)));
            }
            // the aggregator, trained on proposals for a separate split
            WindowLm agg = WindowLm.of(MIX.size(), 3 * 3 + 1, 96, seed + 50);
            List<int[]> ap = new ArrayList<>();
            List<int[]> ac = new ArrayList<>();
            int[][] aggPrompts = aggTrain.toArray(int[][]::new);
            List<int[][]> proposed = new ArrayList<>();
            for (WindowLm lm : agents) {
                proposed.add(lm.sample(aggPrompts, 3, MIX_END, 0, r));
            }
            for (int i = 0; i < aggPrompts.length; i++) {
                int[] q = aggPrompts[i];
                List<int[]> props = new ArrayList<>();
                for (int[][] p : proposed) {
                    props.add(digits(p[i]));
                }
                ap.add(concat(MixtureOfAgents.prompt(new int[0], props, MIX_SEP), new int[] {MIX_EQ}));
                ac.add(truth(q));
            }
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int s = 0; s < steps; s++) {
                int from = (s * 64) % (ap.size() - 64);
                agg.supervisedStep(ap.subList(from, from + 64), ac.subList(from, from + 64), opt);
            }
            MixtureOfAgents moa = MixtureOfAgents.of(List.of(proposers), (q, prev) -> agg.greedy(
                    concat(MixtureOfAgents.prompt(new int[0], prev, MIX_SEP), new int[] {MIX_EQ}), 3, MIX_END));
            List<AnswerDebate.Debater> debaters = new ArrayList<>();
            for (WindowLm lm : agents) {
                debaters.add(lm::sequenceLogProb);
            }
            AnswerDebate panelVote = AnswerDebate.of(debaters).agreement(0.5);
            double[] single = new double[agents.size()];
            double vote = 0;
            double debate = 0;
            double moaOk = 0;
            double moaTokens = 0;
            double poolTokens = 0;
            for (int[] q : test) {
                int[] want = truth(q);
                List<int[]> pool = new ArrayList<>();
                for (int k = 0; k < agents.size(); k++) {
                    int[] g = agents.get(k).greedy(q, 3, MIX_END);
                    single[k] += Arrays.equals(g, want) ? 1 : 0;
                    pool.add(g);
                    int[][] batch = new int[2][];
                    Arrays.fill(batch, q);
                    pool.addAll(List.of(agents.get(k).sample(batch, 3, MIX_END, 1.0, r)));
                }
                long distinct = pool.stream().map(Arrays::toString).distinct().count();
                poolTokens += pool.size() * (q.length + 3) + agents.size() * distinct * (q.length + 3);
                vote += Arrays.equals(panelVote.vote(q, pool), want) ? 1 : 0;
                debate += Arrays.equals(panelVote.debate(q, pool, 2).answer(), want) ? 1 : 0;
                MixtureOfAgents.Answer ans = moa.answer(q);
                moaOk += Arrays.equals(ans.answer(), want) ? 1 : 0;
                moaTokens += ans.tokens();
            }
            double n = test.size();
            double best = Arrays.stream(single).max().orElse(0) / n;
            String p = panel.getKey() + ": ";
            out.put(p + "best single agent", new MixtureRow(best, q0Length() + 3));
            out.put(p + "majority vote", new MixtureRow(vote / n, poolTokens / n));
            out.put(p + "debate", new MixtureRow(debate / n, poolTokens / n));
            out.put(p + "Mixture-of-Agents", new MixtureRow(moaOk / n, moaTokens / n));
        }
        return out;
    }

    private static int q0Length() {
        return mixPrompt(0, 0, 0, 0).length;
    }

    // ------------------------------------------------------------------ 8.4.18 multi-turn RL

    private static final CharacterTokenizer GAME = new CharacterTokenizer("0123?!ynS");
    private static final int ASK_BIT = GAME.encode("?")[0];
    private static final int GUESS = GAME.encode("!")[0];
    private static final int YES = GAME.encode("y")[0];
    private static final int NO = GAME.encode("n")[0];
    private static final int START = GAME.encode("S")[0];
    private static final int SECRETS = 4;
    private static final int TURNS = 4;

    /**
     * Twenty questions in miniature: the user thinks of a number below {@value #SECRETS}; the
     * assistant asks {@code ?i} (is bit {@code i} set? answered {@code y} or {@code n}) or
     * guesses {@code !x}; a right guess ends the game with outcome 1, a wrong one is answered
     * {@code n}, and anything else wastes a turn. With {@value #TURNS} turns a player that never
     * repeats a guess (or asks both bits) always wins; one that guesses at random wins 68% of
     * the time, and a greedy player that repeats itself 25%.
     */
    static final class GuessingUser implements SimulatedUser<Integer> {
        private long replies;

        GuessingUser() {
        }

        @Override
        public int[] opening(Integer intent) {
            return new int[] {START};
        }

        @Override
        public int[] reply(Integer intent, int[] context, int[] response, int turn) {
            replies++;
            if (response.length == 2 && response[0] == GUESS && response[1] == intent) {
                return null;
            }
            if (response.length == 2 && response[0] == ASK_BIT && response[1] < 2) {
                return new int[] {(intent >> response[1] & 1) == 1 ? YES : NO};
            }
            return new int[] {NO};
        }

        @Override
        public float outcome(Integer intent, int[] context, int[] finalResponse) {
            return finalResponse.length == 2 && finalResponse[0] == GUESS && finalResponse[1] == intent ? 1 : 0;
        }
    }

    /**
     * One trainer at one interaction budget.
     *
     * @param success greedy success over every secret
     * @param sampled success of 256 games sampled at temperature 1
     * @param turns mean assistant turns of the sampled games
     * @param interactions user replies spent so far
     */
    public record TurnRow(double success, double sampled, double turns, long interactions) {
    }

    /** A starting policy that plays uniformly random legal moves. */
    static WindowLm randomPlayer(long seed) {
        WindowLm lm = WindowLm.of(GAME.size(), 1 + 3 * TURNS, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        GuessingUser user = new GuessingUser();
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < 1500; i++) {
            int secret = r.nextInt(SECRETS);
            int[] ctx = user.opening(secret);
            for (int t = 1; t <= TURNS; t++) {
                int[] move = r.nextBoolean() ? new int[] {ASK_BIT, r.nextInt(2)} : new int[] {GUESS, r.nextInt(SECRETS)};
                ps.add(ctx);
                cs.add(move);
                int[] reply = user.reply(secret, concat(ctx, move), move, t);
                if (reply == null) {
                    break;
                }
                ctx = concat(concat(ctx, move), reply);
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 150; s++) {
            int from = (s * 128) % (ps.size() - 128);
            lm.supervisedStep(ps.subList(from, from + 128), cs.subList(from, from + 128), opt);
        }
        return lm;
    }

    private static TurnRow evaluateGame(WindowLm lm, long interactions) {
        GuessingUser user = new GuessingUser();
        List<Integer> secrets = new ArrayList<>();
        for (int s = 0; s < SECRETS; s++) {
            secrets.add(s);
        }
        List<HierarchicalAgentTrainer.Episode> eps = HierarchicalAgentTrainer.play(lm, user, secrets, 2, new int[0],
                TURNS, 0, new SplittableRandom(0), new long[1]);
        double ok = 0;
        for (HierarchicalAgentTrainer.Episode e : eps) {
            ok += e.outcome();
        }
        List<HierarchicalAgentTrainer.Episode> sampled = HierarchicalAgentTrainer.play(lm, user,
                secrets(256, new SplittableRandom(1)), 2, new int[0], TURNS, 1, new SplittableRandom(2), new long[1]);
        double hit = 0;
        double turns = 0;
        for (HierarchicalAgentTrainer.Episode e : sampled) {
            hit += e.outcome();
            turns += e.turns();
        }
        return new TurnRow(ok / SECRETS, hit / sampled.size(), turns / sampled.size(), interactions);
    }

    private static List<Integer> secrets(int n, SplittableRandom r) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(r.nextInt(SECRETS));
        }
        return out;
    }

    /**
     * 8.4.18: multi-turn RL with a turn-level critic against trajectory-level baselines. Every
     * trainer starts from the same random player and is evaluated greedily after each slice of
     * its budget of user replies.
     *
     * @param budget user replies per trainer
     * @param checkpoints evaluations (evenly over the budget)
     * @param seed the seed
     * @return trainer name to one row per checkpoint
     */
    public static Map<String, TurnRow[]> multiTurn(long budget, int checkpoints, long seed) {
        WindowLm start = randomPlayer(seed);
        Map<String, TurnRow[]> out = new LinkedHashMap<>();
        String[] names = {"trajectory GRPO", "filtered BC", "CollabLLM multiturn reward", "ArCHer"};
        int[] none = new int[0];
        for (String name : names) {
            WindowLm lm = start.copy();
            SplittableRandom r = new SplittableRandom(seed + name.hashCode());
            TurnRow[] rows = new TurnRow[checkpoints];
            GuessingUser user = new GuessingUser();
            Optimizer opt = Optimizer.adam(3e-3f);
            Optimizer criticOpt = Optimizer.adam(1e-2f);
            HierarchicalAgentTrainer<Integer> archer = HierarchicalAgentTrainer.of(lm, user,
                    TurnLevelCritic.of(GAME.size(), 1 + 3 * TURNS, 2, 64, 0.95f, 0.05f, seed + 3), 2, none, TURNS)
                    .schedule(8, 64, 4, 2048, 256, 1.0);
            MultiturnRewardTrainer<Integer> collab = MultiturnRewardTrainer.of(lm,
                    GrpoTrainer.of(lm).optimizer(Optimizer.adam(3e-3f)).completions(2, -1), user)
                    .sampling(4, 2, none, 1.0).reward(MultiturnRewardTrainer.Mode.MULTITURN, 2, TURNS, 0.05f);
            long[] spent = new long[1];
            int next = 0;
            while (next < checkpoints) {
                switch (name) {
                    case "trajectory GRPO" -> {
                        List<HierarchicalAgentTrainer.Episode> eps = HierarchicalAgentTrainer.play(lm, user,
                                secrets(16, r), 2, none, TURNS, 1.0, r, spent);
                        float[] outcomes = new float[eps.size()];
                        for (int i = 0; i < outcomes.length; i++) {
                            outcomes[i] = eps.get(i).outcome();
                        }
                        float[] adv = GrpoTrainer.advantages(outcomes, true);
                        List<int[]> cs = new ArrayList<>();
                        List<int[]> us = new ArrayList<>();
                        List<Float> as = new ArrayList<>();
                        for (int i = 0; i < eps.size(); i++) {
                            for (TurnLevelCritic.Transition t : eps.get(i).transitions()) {
                                cs.add(t.context());
                                us.add(t.utterance());
                                as.add(adv[i]);
                            }
                        }
                        ToolInterleavedRollout.policyGradient(lm, cs, us, as, opt);
                    }
                    case "filtered BC" -> {
                        List<HierarchicalAgentTrainer.Episode> eps = HierarchicalAgentTrainer.play(lm, user,
                                secrets(16, r), 2, none, TURNS, 1.0, r, spent);
                        List<int[]> cs = new ArrayList<>();
                        List<int[]> us = new ArrayList<>();
                        for (HierarchicalAgentTrainer.Episode e : eps) {
                            if (e.outcome() > 0) {
                                for (TurnLevelCritic.Transition t : e.transitions()) {
                                    cs.add(t.context());
                                    us.add(t.utterance());
                                }
                            }
                        }
                        if (!cs.isEmpty()) {
                            lm.supervisedStep(cs, us, opt);
                        }
                    }
                    case "CollabLLM multiturn reward" -> {
                        long before = user.replies;
                        collab.step(secrets(4, r), r);
                        spent[0] += user.replies - before;
                    }
                    default -> {
                        long before = archer.interactions();
                        archer.step(secrets(16, r), opt, criticOpt, r);
                        spent[0] += archer.interactions() - before;
                    }
                }
                while (next < checkpoints && spent[0] >= budget * (next + 1) / checkpoints) {
                    rows[next++] = evaluateGame(lm, spent[0]);
                }
            }
            out.put(name, rows);
        }
        return out;
    }
}
