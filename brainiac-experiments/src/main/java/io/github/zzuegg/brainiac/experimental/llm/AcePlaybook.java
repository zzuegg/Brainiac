package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * An evolving playbook in the style of ACE, Agentic Context Engineering (Zhang et al. 2025,
 * <a href="https://arxiv.org/abs/2510.04618">arXiv:2510.04618</a>): instead of rewriting a
 * prompt as a whole (which collapses it into a short summary over time), the context is a
 * list of itemised bullets that grows by small <em>delta</em> updates. Three roles share the
 * work: a {@link Generator} solves a task with the playbook and reports which bullets it
 * used; a {@link Reflector} turns the outcome into insights and tags the bullets used as
 * helpful or harmful; a {@link Curator} turns insights into {@link Delta deltas}, merged
 * deterministically. Grow-and-refine then merges near-duplicate bullets (by embedding
 * similarity) and prunes those tagged harmful more than helpful. The model's weights never
 * change; only its context does.
 */
public final class AcePlaybook {

    /**
     * One playbook entry.
     *
     * @param id the stable id
     * @param section the section, such as strategies or mistakes
     * @param text the advice
     * @param helpful the times it was tagged helpful
     * @param harmful the times it was tagged harmful
     */
    public record Bullet(String id, String section, String text, int helpful, int harmful) {

        /** @return helpful minus harmful */
        public int score() {
            return helpful - harmful;
        }
    }

    /** A change to the playbook. */
    public sealed interface Delta {
    }

    /**
     * Adds a bullet.
     *
     * @param section the section
     * @param text the advice
     */
    public record Add(String section, String text) implements Delta { }

    /**
     * Tags a bullet.
     *
     * @param id the bullet
     * @param helpful helpful or harmful
     */
    public record Tag(String id, boolean helpful) implements Delta { }

    /**
     * Removes a bullet.
     *
     * @param id the bullet
     */
    public record Remove(String id) implements Delta { }

    /**
     * A generator's attempt.
     *
     * @param answer the answer
     * @param used the ids of the bullets it relied on
     */
    public record Attempt(String answer, List<String> used) { }

    /**
     * What the reflector learned from an attempt.
     *
     * @param insights new advice, each {@code section: text} or plain text
     * @param tags tags for the bullets used
     */
    public record Reflection(List<String> insights, List<Tag> tags) { }

    /**
     * Solves a task with the playbook.
     *
     * @param <T> the task
     */
    @FunctionalInterface
    public interface Generator<T> {
        /**
         * @param task the task
         * @param playbook the rendered playbook
         * @return the attempt
         */
        Attempt attempt(T task, String playbook);
    }

    /**
     * Learns from an attempt and its feedback.
     *
     * @param <T> the task
     */
    @FunctionalInterface
    public interface Reflector<T> {
        /**
         * @param task the task
         * @param attempt the attempt
         * @param feedback execution feedback or the correct answer
         * @return the reflection
         */
        Reflection reflect(T task, Attempt attempt, String feedback);
    }

    /** Turns a reflection into deltas. */
    @FunctionalInterface
    public interface Curator {
        /**
         * @param reflection the reflection
         * @param playbook the current playbook
         * @return the deltas
         */
        List<Delta> curate(Reflection reflection, AcePlaybook playbook);

        /** @return the default curator: every tag, and every insight as a new bullet */
        static Curator standard() {
            return (r, p) -> {
                List<Delta> out = new ArrayList<>(r.tags());
                for (String s : r.insights()) {
                    int colon = s.indexOf(": ");
                    out.add(colon > 0 ? new Add(s.substring(0, colon), s.substring(colon + 2)) : new Add("general", s));
                }
                return out;
            };
        }
    }

    private final Map<String, Bullet> bullets = new LinkedHashMap<>();
    private final Function<String, float[]> embedder;
    private double duplicate = 0.95;
    private int pruneMargin = 2;
    private int nextId;

    private AcePlaybook(Function<String, float[]> embedder) {
        this.embedder = embedder;
    }

    /**
     * An empty playbook.
     *
     * @param embedder embeds bullet texts, for finding duplicates
     * @return the playbook
     */
    public static AcePlaybook create(Function<String, float[]> embedder) {
        return new AcePlaybook(embedder);
    }

    /**
     * @param similarity the cosine similarity above which two bullets are merged
     * @param margin a bullet is pruned once it is tagged harmful this many times more than helpful
     * @return this playbook
     */
    public AcePlaybook refineWith(double similarity, int margin) {
        this.duplicate = similarity;
        this.pruneMargin = margin;
        return this;
    }

    /**
     * Applies deltas in order; tags and removals of unknown ids are ignored.
     *
     * @param deltas the deltas
     */
    public void apply(List<? extends Delta> deltas) {
        for (Delta d : deltas) {
            switch (d) {
                case Add a -> {
                    String id = "b" + nextId++;
                    bullets.put(id, new Bullet(id, a.section(), a.text(), 0, 0));
                }
                case Tag t -> bullets.computeIfPresent(t.id(), (k, b) -> new Bullet(b.id(), b.section(), b.text(),
                        b.helpful() + (t.helpful() ? 1 : 0), b.harmful() + (t.helpful() ? 0 : 1)));
                case Remove r -> bullets.remove(r.id());
            }
        }
    }

    /**
     * Grow-and-refine: merges near-duplicates (the older bullet keeps both counts) and prunes
     * harmful bullets.
     *
     * @return the bullets removed
     */
    public int refine() {
        int before = bullets.size();
        List<Bullet> list = new ArrayList<>(bullets.values());
        List<float[]> emb = list.stream().map(b -> embedder.apply(b.text())).toList();
        boolean[] gone = new boolean[list.size()];
        for (int i = 0; i < list.size(); i++) {
            if (gone[i]) {
                continue;
            }
            Bullet keep = list.get(i);
            for (int j = i + 1; j < list.size(); j++) {
                if (!gone[j] && cosine(emb.get(i), emb.get(j)) >= duplicate) {
                    Bullet b = list.get(j);
                    keep = new Bullet(keep.id(), keep.section(), keep.text(), keep.helpful() + b.helpful(),
                            keep.harmful() + b.harmful());
                    gone[j] = true;
                    bullets.remove(b.id());
                }
            }
            bullets.put(keep.id(), keep);
        }
        bullets.values().removeIf(b -> -b.score() >= pruneMargin);
        return before - bullets.size();
    }

    private static double cosine(float[] a, float[] b) {
        double d = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return na == 0 || nb == 0 ? 0 : d / Math.sqrt(na * nb);
    }

    /**
     * One ACE step: attempt, reflect, curate, apply, refine.
     *
     * @param <T> the task type
     * @param task the task
     * @param feedback gives feedback on an answer (the correct answer, or a test's output)
     * @param generator the generator
     * @param reflector the reflector
     * @param curator the curator
     * @param budget the token budget the generator's playbook is rendered in
     * @param tokens counts tokens
     * @return the attempt
     */
    public <T> Attempt step(T task, Function<String, String> feedback, Generator<T> generator,
            Reflector<T> reflector, Curator curator, int budget, ToIntFunction<String> tokens) {
        Attempt a = generator.attempt(task, render(budget, tokens));
        Reflection r = reflector.reflect(task, a, feedback.apply(a.answer()));
        apply(curator.curate(r, this));
        refine();
        return a;
    }

    /**
     * Renders the best bullets that fit a token budget, grouped by section.
     *
     * @param budget the token budget
     * @param tokens counts tokens
     * @return the playbook text, one {@code [id] text} line per bullet
     */
    public String render(int budget, ToIntFunction<String> tokens) {
        List<Bullet> ranked = new ArrayList<>(bullets.values());
        ranked.sort(Comparator.comparingInt(Bullet::score).reversed());
        Map<String, List<String>> sections = new LinkedHashMap<>();
        int used = 0;
        for (Bullet b : ranked) {
            String line = "[" + b.id() + "] " + b.text();
            int n = tokens.applyAsInt(line);
            if (used + n > budget) {
                continue;
            }
            used += n;
            sections.computeIfAbsent(b.section(), k -> new ArrayList<>()).add(line);
        }
        StringBuilder sb = new StringBuilder();
        sections.forEach((s, lines) -> {
            sb.append("## ").append(s).append('\n');
            lines.forEach(l -> sb.append(l).append('\n'));
        });
        return sb.toString();
    }

    /** @return the bullets, oldest first */
    public List<Bullet> bullets() {
        return List.copyOf(bullets.values());
    }

    @Override
    public String toString() {
        return "AcePlaybook(" + bullets.size() + " bullets)";
    }
}
