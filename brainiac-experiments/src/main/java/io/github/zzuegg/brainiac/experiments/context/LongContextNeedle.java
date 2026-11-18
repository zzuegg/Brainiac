package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.MemoryOverwriteAgent;
import io.github.zzuegg.brainiac.experimental.llm.RecursiveLm;
import io.github.zzuegg.brainiac.experimental.llm.VirtualContextPager;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Needle-in-a-haystack question answering far beyond the window, for MemAgent
 * (<a href="https://arxiv.org/abs/2507.02259">arXiv:2507.02259</a>, {@link MemoryOverwriteAgent})
 * and recursive language models (<a href="https://arxiv.org/abs/2512.24601">arXiv:2512.24601</a>,
 * {@link RecursiveLm}). A context is a run of two-token slots, each either a key-value pair
 * ({@code k v}, a letter and a digit, keys distinct) or filler ({@code ..}); the question
 * {@code ?k=} asks for the value of one key in it. Every model reads at most {@value #WINDOW}
 * context tokens at a time, and contexts are 4 to 32 times longer.
 *
 * <p>The arms: a reader that sees only the last window (truncation); retrieval of the one
 * window whose letters best match the question from a {@link VirtualContextPager}'s archival
 * memory, then the reader on it; {@link RecursiveLm} with the same reader at every call; and
 * {@link MemoryOverwriteAgent}, a memory writer first supervised to keep a matching memory
 * only half the time (so it loses the needle over long inputs) and then trained with RL on four- and
 * eight-chunk contexts. Accuracy and model calls per question are reported per length.
 */
public final class LongContextNeedle {
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
     * One arm at one length.
     *
     * @param accuracy share answered correctly
     * @param calls model calls per question
     */
    public record Cell(double accuracy, double calls) {
    }

    /**
     * A question.
     *
     * @param context the haystack
     * @param query {@code ?k=}
     * @param answer the needle's value token
     */
    public record Question(int[] context, int[] query, int answer) {
    }

    private LongContextNeedle() {
    }

    /**
     * A random question.
     *
     * @param length context tokens, a multiple of {@value #WINDOW}
     * @param random the random source
     * @return the question
     */
    public static Question question(int length, SplittableRandom random) {
        int slots = length / 2;
        int pairs = Math.min(KEYS, Math.max(2, length / WINDOW));
        int[] context = new int[length];
        Arrays.fill(context, FILL);
        int[] keys = shuffled(KEYS, random);
        int[] where = shuffled(slots, random);
        for (int i = 0; i < pairs; i++) {
            context[2 * where[i]] = FIRST_KEY + keys[i];
            context[2 * where[i] + 1] = random.nextInt(10);
        }
        int needle = random.nextInt(pairs);
        return new Question(context, new int[] {ASK, FIRST_KEY + keys[needle], EQUALS},
                context[2 * where[needle] + 1]);
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

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    /**
     * The reader: {@code context ++ ?k=} (the context left-padded to a window) to the value and
     * {@code #}, or {@code -#}.
     */
    static WindowLm reader(long seed) {
        WindowLm lm = WindowLm.of(TOK.size(), WINDOW + 3, 128, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < 6000; i++) {
            int length = 2 * (1 + r.nextInt(WINDOW / 2));
            Question q = question(32, r);
            int[] context = Arrays.copyOfRange(q.context(), 0, length);
            for (int s = 0; s < length; s += 2) {
                if (r.nextDouble() < 0.3) {
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
        for (int s = 0; s < 600; s++) {
            int from = (s * 256) % (ps.size() - 256);
            lm.supervisedStep(ps.subList(from, from + 256), cs.subList(from, from + 256), opt);
        }
        return lm;
    }

    private static int read(WindowLm reader, int[] query, int[] context) {
        return reader.greedy(concat(padded(context), query), 2, END)[0];
    }

    /** Left-pads a context of at most a window with filler, so slots sit at fixed positions. */
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
     * The memory writer, supervised to copy a matching pair from the chunk and, when the chunk
     * has none, to keep a matching memory only half the time ({@link #KEEP}): a weak, unreliable
     * prior for what to remember, as an instruction-tuned model has.
     */
    static MemoryOverwriteAgent agent(long seed) {
        MemoryOverwriteAgent agent = MemoryOverwriteAgent.of(WindowLm.of(TOK.size(), 3 + 2 + WINDOW, 128, seed), 2,
                WINDOW, END, FILL);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < 4000; i++) {
            Question q = question(WINDOW * 4, r);
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
        for (int s = 0; s < 400; s++) {
            int from = (s * 256) % (ps.size() - 256);
            agent.model().supervisedStep(ps.subList(from, from + 256), cs.subList(from, from + 256), opt);
        }
        return agent;
    }

    /**
     * Trains the memory writer by RL on the final answer.
     *
     * @param agent the agent
     * @param reader answers from the final memory
     * @param steps policy-gradient steps
     * @param seed the seed
     */
    static void reinforce(MemoryOverwriteAgent agent, WindowLm reader, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        int group = 8;
        for (int s = 0; s < steps; s++) {
            List<int[]> contexts = new ArrayList<>();
            List<int[]> queries = new ArrayList<>();
            List<Question> qs = new ArrayList<>();
            for (int i = 0; i < 12; i++) {
                Question q = question(WINDOW * (r.nextBoolean() ? 4 : 8), r);
                for (int g = 0; g < group; g++) {
                    contexts.add(q.context());
                    queries.add(q.query());
                    qs.add(q);
                }
            }
            List<MemoryOverwriteAgent.Episode> eps = agent.read(contexts, queries, 1.0, r);
            float[] rewards = new float[eps.size()];
            for (int i = 0; i < rewards.length; i++) {
                rewards[i] = read(reader, qs.get(i).query(), eps.get(i).memory()) == qs.get(i).answer() ? 1 : 0;
            }
            agent.reinforce(eps, rewards, group, opt);
        }
    }

    /**
     * Runs the comparison.
     *
     * @param lengths context lengths, multiples of {@value #WINDOW}
     * @param questions questions per length
     * @param rlSteps MemAgent RL steps
     * @param seed the seed
     * @return arm name to one cell per length
     */
    public static Map<String, Cell[]> run(int[] lengths, int questions, int rlSteps, long seed) {
        WindowLm reader = reader(seed);
        MemoryOverwriteAgent sftAgent = agent(seed + 2);
        MemoryOverwriteAgent agent = MemoryOverwriteAgent.of(sftAgent.model().copy(), 2, WINDOW, END, FILL);
        reinforce(agent, reader, rlSteps, seed + 3);
        RecursiveLm rlm = RecursiveLm.of(WINDOW, (q, c) -> new int[] {read(reader, q, c)},
                (q, a) -> a[0] == NONE ? new int[] {FILL, FILL} : new int[] {q[1], a[0]});
        Map<String, Cell[]> out = new LinkedHashMap<>();
        String[] names = {"truncation", "pager retrieval", "RLM", "MemAgent SFT only", "MemAgent RL"};
        for (String n : names) {
            out.put(n, new Cell[lengths.length]);
        }
        for (int li = 0; li < lengths.length; li++) {
            SplittableRandom r = new SplittableRandom(seed + 100 + li);
            List<Question> qs = new ArrayList<>();
            for (int i = 0; i < questions; i++) {
                qs.add(question(lengths[li], r));
            }
            int trunc = 0;
            int paged = 0;
            int recursive = 0;
            long before = rlm.calls();
            for (Question q : qs) {
                int[] c = q.context();
                trunc += read(reader, q.query(), Arrays.copyOfRange(c, c.length - WINDOW, c.length)) == q.answer()
                        ? 1 : 0;
                VirtualContextPager pager = VirtualContextPager.of(WINDOW, String::length, LongContextNeedle::embed,
                        KEYS);
                for (int s = 0; s < c.length; s += WINDOW) {
                    pager.archivalInsert(TOK.decode(Arrays.copyOfRange(c, s, s + WINDOW)));
                }
                int[] hit = TOK.encode(pager.archivalSearch(TOK.decode(q.query()[1]), 1).get(0));
                paged += read(reader, q.query(), hit) == q.answer() ? 1 : 0;
                recursive += rlm.answer(q.query(), c)[0] == q.answer() ? 1 : 0;
            }
            out.get("truncation")[li] = new Cell((double) trunc / questions, 1);
            out.get("pager retrieval")[li] = new Cell((double) paged / questions, 1);
            out.get("RLM")[li] = new Cell((double) recursive / questions, (double) (rlm.calls() - before) / questions);
            for (String n : new String[] {"MemAgent SFT only", "MemAgent RL"}) {
                MemoryOverwriteAgent a = n.endsWith("RL") ? agent : sftAgent;
                List<int[]> contexts = new ArrayList<>();
                List<int[]> queries = new ArrayList<>();
                for (Question q : qs) {
                    contexts.add(q.context());
                    queries.add(q.query());
                }
                List<MemoryOverwriteAgent.Episode> eps = a.read(contexts, queries, 0, r);
                int ok = 0;
                for (int i = 0; i < qs.size(); i++) {
                    ok += read(reader, qs.get(i).query(), eps.get(i).memory()) == qs.get(i).answer() ? 1 : 0;
                }
                out.get(n)[li] = new Cell((double) ok / questions, lengths[li] / (double) WINDOW + 1);
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
}
