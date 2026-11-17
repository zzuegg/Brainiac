package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * MemGPT-style virtual context management (Packer et al. 2023,
 * <a href="https://arxiv.org/abs/2310.08560">arXiv:2310.08560</a>): the model's prompt is
 * main memory, paged like an operating system's.
 * <ul>
 * <li>Main context: a pinned system prompt, an editable <em>core memory</em>, a recursive
 *     summary of everything evicted, and a FIFO queue of recent messages, together within a
 *     token budget.</li>
 * <li>When the queue passes a warning threshold the pager reports <em>memory pressure</em>
 *     (so the model can save what matters); when it passes the budget, the oldest messages
 *     are evicted to <em>recall storage</em> and folded into the summary.</li>
 * <li>Recall storage keeps every message and is searched by text; <em>archival
 *     storage</em> is an embedding index the model writes to and searches explicitly.</li>
 * <li>The model reaches all of this through function calls ({@link #call}), whose results
 *     are paged back into the queue.</li>
 * </ul>
 * Text in, text out: tokens are counted and texts embedded by the functions given, so any
 * tokenizer and embedder plug in.
 */
public final class VirtualContextPager {

    /**
     * One message in the queue or recall storage.
     *
     * @param id the order it arrived in
     * @param role who wrote it, such as user, assistant or function
     * @param text the text
     * @param tokens its token count
     */
    public record Message(long id, String role, String text, int tokens) { }

    /** What {@link #add} caused. */
    public enum Event {
        /** Nothing. */
        NONE,
        /** The queue passed the warning threshold. */
        PRESSURE,
        /** Messages were evicted to recall storage. */
        EVICTED
    }

    private final int budget;
    private final ToIntFunction<String> tokens;
    private final Function<String, float[]> embedder;
    private final KnnIndex archival;
    private final List<String> archivalTexts = new ArrayList<>();
    private final List<Message> queue = new ArrayList<>();
    private final List<Message> recall = new ArrayList<>();
    private String system = "";
    private final StringBuilder core = new StringBuilder();
    private String summary = "";
    private Function<String, String> summarizer = s -> s;
    private double warnAt = 0.7;
    private double evictTo = 0.5;
    private int coreLimit = Integer.MAX_VALUE;
    private long nextId;
    private int evictions;

    private VirtualContextPager(int budget, ToIntFunction<String> tokens, Function<String, float[]> embedder,
            int embeddingSize) {
        this.budget = budget;
        this.tokens = tokens;
        this.embedder = embedder;
        this.archival = KnnIndex.create(embeddingSize, 1, KnnIndex.Metric.COSINE);
    }

    /**
     * A pager.
     *
     * @param budget the main context's token budget
     * @param tokens counts the tokens of a text
     * @param embedder embeds a text for archival search
     * @param embeddingSize the embedding size
     * @return the pager
     */
    public static VirtualContextPager of(int budget, ToIntFunction<String> tokens, Function<String, float[]> embedder,
            int embeddingSize) {
        return new VirtualContextPager(budget, tokens, embedder, embeddingSize);
    }

    /**
     * @param prompt the pinned system prompt
     * @return this pager
     */
    public VirtualContextPager system(String prompt) {
        this.system = prompt;
        return this;
    }

    /**
     * @param fn writes a new summary from {@code previous summary + evicted messages}
     *     (a call to the model, in practice); by default the text is kept as it is
     * @return this pager
     */
    public VirtualContextPager summarizer(Function<String, String> fn) {
        this.summarizer = fn;
        return this;
    }

    /**
     * @param warn the share of the budget at which pressure is reported
     * @param evictDownTo the share of the budget the context is brought down to on eviction
     * @return this pager
     */
    public VirtualContextPager thresholds(double warn, double evictDownTo) {
        if (!(evictDownTo < warn && warn <= 1)) {
            throw new IllegalArgumentException("need evictDownTo < warn <= 1");
        }
        this.warnAt = warn;
        this.evictTo = evictDownTo;
        return this;
    }

    /**
     * @param tokenLimit the most tokens core memory may hold
     * @return this pager
     */
    public VirtualContextPager coreLimit(int tokenLimit) {
        this.coreLimit = tokenLimit;
        return this;
    }

    /**
     * Adds a message to the queue, evicting the oldest ones when the budget is exceeded.
     *
     * @param role the role
     * @param text the text
     * @return what happened
     */
    public Event add(String role, String text) {
        Message m = new Message(nextId++, role, text, tokens.applyAsInt(text));
        queue.add(m);
        recall.add(m);
        if (contextTokens() > budget) {
            evict();
            return Event.EVICTED;
        }
        return contextTokens() > warnAt * budget ? Event.PRESSURE : Event.NONE;
    }

    private void evict() {
        StringBuilder gone = new StringBuilder(summary);
        summary = "";                         // the old summary is folded into the new one
        while (queue.size() > 1 && contextTokens() > evictTo * budget) {
            Message m = queue.removeFirst();
            if (!gone.isEmpty()) {
                gone.append('\n');
            }
            gone.append(m.role()).append(": ").append(m.text());
        }
        summary = summarizer.apply(gone.toString());
        evictions++;
        while (contextTokens() > budget && !queue.isEmpty()) {
            queue.removeFirst();              // the summary itself is long: drop more
        }
    }

    /** @return the tokens of the whole main context */
    public int contextTokens() {
        int n = tokens.applyAsInt(system) + tokens.applyAsInt(core.toString()) + tokens.applyAsInt(summary);
        for (Message m : queue) {
            n += m.tokens();
        }
        return n;
    }

    /** @return whether the context is past the warning threshold */
    public boolean pressure() {
        return contextTokens() > warnAt * budget;
    }

    /**
     * Runs one of the MemGPT functions and pages its result into the queue as a function
     * message: {@code core_memory_append(text)}, {@code core_memory_replace(old\nnew)},
     * {@code archival_memory_insert(text)}, {@code archival_memory_search(query)},
     * {@code conversation_search(query)}.
     *
     * @param function the function name
     * @param argument its argument
     * @return the result text
     */
    public String call(String function, String argument) {
        String result = switch (function) {
            case "core_memory_append" -> coreAppend(argument);
            case "core_memory_replace" -> {
                int nl = argument.indexOf('\n');
                if (nl < 0) {
                    throw new IllegalArgumentException("core_memory_replace takes old\\nnew");
                }
                yield coreReplace(argument.substring(0, nl), argument.substring(nl + 1));
            }
            case "archival_memory_insert" -> {
                archivalInsert(argument);
                yield "inserted";
            }
            case "archival_memory_search" -> String.join("\n", archivalSearch(argument, 3));
            case "conversation_search" -> {
                List<String> hits = recallSearch(argument, 3).stream().map(m -> m.role() + ": " + m.text()).toList();
                yield hits.isEmpty() ? "no results" : String.join("\n", hits);
            }
            default -> throw new IllegalArgumentException("unknown function " + function);
        };
        add("function", function + " -> " + result);
        return result;
    }

    /**
     * @param text appended to core memory
     * @return the result text
     */
    public String coreAppend(String text) {
        String next = core.isEmpty() ? text : core + "\n" + text;
        if (tokens.applyAsInt(next) > coreLimit) {
            return "core memory full";
        }
        core.setLength(0);
        core.append(next);
        return "ok";
    }

    /**
     * @param old text in core memory
     * @param replacement its replacement
     * @return the result text
     */
    public String coreReplace(String old, String replacement) {
        int at = core.indexOf(old);
        if (at < 0) {
            return "not found";
        }
        core.replace(at, at + old.length(), replacement);
        return "ok";
    }

    /** @param text stored in archival memory */
    public void archivalInsert(String text) {
        archival.add(embedder.apply(text), new float[] {archivalTexts.size()});
        archivalTexts.add(text);
    }

    /**
     * @param query the query text
     * @param k the most results
     * @return the closest archival texts, closest first
     */
    public List<String> archivalSearch(String query, int k) {
        List<String> out = new ArrayList<>();
        for (KnnIndex.Hit h : archival.search(embedder.apply(query), k)) {
            out.add(archivalTexts.get((int) archival.value(h.row())[0]));
        }
        return out;
    }

    /**
     * Searches every message ever added for all words of a query, newest first.
     *
     * @param query the words
     * @param k the most results
     * @return the messages
     */
    public List<Message> recallSearch(String query, int k) {
        String[] words = query.toLowerCase(Locale.ROOT).split("\\s+");
        List<Message> out = new ArrayList<>();
        for (int i = recall.size() - 1; i >= 0 && out.size() < k; i--) {
            String t = recall.get(i).text().toLowerCase(Locale.ROOT);
            boolean all = true;
            for (String w : words) {
                all &= t.contains(w);
            }
            if (all && !recall.get(i).role().equals("function")) {
                out.add(recall.get(i));
            }
        }
        return out;
    }

    /** @return the queue of recent messages */
    public List<Message> queue() {
        return List.copyOf(queue);
    }

    /** @return the summary of evicted messages */
    public String summary() {
        return summary;
    }

    /** @return core memory */
    public String core() {
        return core.toString();
    }

    /** @return how many times messages were evicted */
    public int evictions() {
        return evictions;
    }

    /** @return the messages in recall storage */
    public int recallSize() {
        return recall.size();
    }

    /** @return the main context as the prompt text the model reads */
    public String render() {
        StringBuilder b = new StringBuilder();
        b.append("[system]\n").append(system).append('\n');
        b.append("[core memory]\n").append(core).append('\n');
        if (!summary.isEmpty()) {
            b.append("[summary of earlier conversation]\n").append(summary).append('\n');
        }
        b.append("[recall: ").append(recall.size() - queue.size()).append(" earlier messages; archival: ")
                .append(archivalTexts.size()).append(" entries]\n");
        for (Message m : queue) {
            b.append(m.role()).append(": ").append(m.text()).append('\n');
        }
        return b.toString();
    }

    @Override
    public String toString() {
        return "VirtualContextPager(" + contextTokens() + "/" + budget + " tokens, " + queue.size() + " queued, "
                + recall.size() + " recalled, " + archivalTexts.size() + " archived)";
    }
}
