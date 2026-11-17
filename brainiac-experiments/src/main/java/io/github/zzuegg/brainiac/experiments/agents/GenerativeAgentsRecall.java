package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.llm.MemoryStream;
import io.github.zzuegg.brainiac.experimental.llm.VirtualContextPager;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Generative Agents' memory stream on a multi-day toy town (item 5.1). Five people move
 * between eight places every hour for several days; each has a favourite place they visit
 * half the time, and each has one important life event early on ("bob won the lottery").
 * An observer agent records everything it sees, with distractor chatter. At the end it is
 * asked three kinds of question per person:
 * <ul>
 *   <li>"where is alice now": needs the <em>latest</em> location (recency);</li>
 *   <li>"what important thing happened with alice": needs the life event, which is old
 *       and lexically no closer to the question than the hundreds of routine sightings
 *       (importance);</li>
 *   <li>"where does alice usually go": needs an aggregate over many sightings, which a
 *       reflection writes down (reflection).</li>
 * </ul>
 * The same rule-based reader answers from the records each memory hands it, the same number
 * {@code k} for every memory. Baselines: a sliding window of the last {@code k} records, and
 * {@link VirtualContextPager} (a main-context queue of about {@code k/2} records plus
 * {@code k/2} hits of relevance search in archival storage, as a MemGPT agent would search).
 * A relevance-only memory stream is the ablation.
 */
public final class GenerativeAgentsRecall {

    static final String[] PEOPLE = {"alice", "bob", "carol", "dave", "erin"};
    static final String[] PLACES = {"cafe", "park", "library", "gym", "market", "school", "office", "beach"};
    // as many words as a sighting ("is at the park"), so text length does not give events away
    static final String[] EVENTS = {"won the city lottery", "broke a left leg", "got married last week",
        "adopted a small puppy", "lost the factory job"};
    static final String[] CHATTER = {"the weather is sunny", "a bus drove by", "someone sneezed loudly",
        "the clock struck the hour", "a bird sang on the roof"};

    /**
     * Recall accuracy (0..1) per memory and question type.
     *
     * @param streamLatest memory stream, latest-location questions
     * @param streamImportant memory stream, life-event questions
     * @param streamUsual memory stream, favourite-place questions
     * @param relevanceOnly relevance-only stream, mean over all questions
     * @param window sliding window, mean over all questions
     * @param pager VirtualContextPager, mean over all questions
     * @param stream memory stream, mean over all questions
     * @param memories records in the stream at the end, reflections included
     */
    public record Result(double streamLatest, double streamImportant, double streamUsual, double relevanceOnly,
            double window, double pager, double stream, int memories) {
    }

    private GenerativeAgentsRecall() {
    }

    private record World(List<String> log, List<Double> times, Map<String, String> latest,
            Map<String, String> event, Map<String, String> favourite) {
    }

    private static World simulate(int days, SplittableRandom random) {
        List<String> log = new ArrayList<>();
        List<Double> times = new ArrayList<>();
        Map<String, String> latest = new HashMap<>();
        Map<String, String> event = new HashMap<>();
        Map<String, String> favourite = new HashMap<>();
        for (int p = 0; p < PEOPLE.length; p++) {
            favourite.put(PEOPLE[p], PLACES[(p * 3 + random.nextInt(2)) % PLACES.length]);
        }
        int hours = days * 10;
        int[] eventHour = new int[PEOPLE.length];
        for (int p = 0; p < PEOPLE.length; p++) {
            eventHour[p] = random.nextInt(Math.max(1, hours / 3));
        }
        for (int h = 0; h < hours; h++) {
            for (int p = 0; p < PEOPLE.length; p++) {
                String who = PEOPLE[p];
                String place = random.nextDouble() < 0.5 ? favourite.get(who) : PLACES[random.nextInt(PLACES.length)];
                latest.put(who, place);
                log.add(who + " is at the " + place);
                times.add((double) h);
                if (h == eventHour[p]) {
                    String e = EVENTS[(p + random.nextInt(EVENTS.length)) % EVENTS.length];
                    event.put(who, e);
                    log.add(who + " " + e);
                    times.add((double) h);
                }
            }
            log.add(CHATTER[random.nextInt(CHATTER.length)]);
            times.add((double) h);
        }
        return new World(log, times, latest, event, favourite);
    }

    private static boolean isEvent(String s) {
        for (String e : EVENTS) {
            if (s.endsWith(e)) {
                return true;
            }
        }
        return false;
    }

    /** The reader: answers a question from candidate records, in the order given. */
    static String answer(int type, String who, List<String> records) {
        switch (type) {
            case 0 -> {
                for (String r : records) {
                    if (r.startsWith(who + " is at the ")) {
                        return r.substring((who + " is at the ").length());
                    }
                }
            }
            case 1 -> {
                for (String r : records) {
                    if (r.startsWith(who + " ") && isEvent(r)) {
                        return r.substring(who.length() + 1);
                    }
                }
            }
            default -> {
                for (String r : records) {
                    if (r.startsWith(who + " usually goes to the ")) {
                        return r.substring((who + " usually goes to the ").length());
                    }
                }
                Map<String, Integer> counts = new HashMap<>();   // otherwise, a majority over sightings
                String best = null;
                for (String r : records) {
                    if (r.startsWith(who + " is at the ")) {
                        String place = r.substring((who + " is at the ").length());
                        int c = counts.merge(place, 1, Integer::sum);
                        if (best == null || c > counts.get(best)) {
                            best = place;
                        }
                    }
                }
                return best;
            }
        }
        return null;
    }

    static String question(int type, String who) {
        return switch (type) {
            case 0 -> "where is " + who + " now";
            case 1 -> "what important thing happened with " + who;
            default -> "where does " + who + " usually go";
        };
    }

    static String truth(World w, int type, String who) {
        return switch (type) {
            case 0 -> w.latest().get(who);
            case 1 -> w.event().get(who);
            default -> w.favourite().get(who);
        };
    }

    /** The reflection: per person, the most visited place among the recent sightings. */
    static List<MemoryStream.Insight> reflect(List<MemoryStream.Memory> recent) {
        List<MemoryStream.Insight> out = new ArrayList<>();
        for (String who : PEOPLE) {
            Map<String, List<Integer>> visits = new HashMap<>();
            for (MemoryStream.Memory m : recent) {
                if (m.text().startsWith(who + " is at the ")) {
                    visits.computeIfAbsent(m.text().substring((who + " is at the ").length()), k -> new ArrayList<>())
                            .add(m.id());
                }
            }
            visits.entrySet().stream().max((a, b) -> Integer.compare(a.getValue().size(), b.getValue().size()))
                    .filter(e -> e.getValue().size() >= 3)
                    .ifPresent(e -> out.add(new MemoryStream.Insight(who + " usually goes to the " + e.getKey(), 7,
                            e.getValue())));
        }
        return out;
    }

    private static double importance(String text) {
        return isEvent(text) ? 9 : text.contains(" is at the ") ? 2 : 1;
    }

    /**
     * @param days simulated days (10 hours each)
     * @param k records retrieved per question
     * @param towns independent towns to average over
     * @param seed the seed
     * @return the recall accuracies, averaged over the towns
     */
    public static Result run(int days, int k, int towns, long seed) {
        double[] sum = new double[7];
        int memories = 0;
        for (int t = 0; t < towns; t++) {
            Result r = runTown(days, k, seed + 1000L * t);
            double[] v = {r.streamLatest(), r.streamImportant(), r.streamUsual(), r.relevanceOnly(), r.window(),
                r.pager(), r.stream()};
            for (int i = 0; i < v.length; i++) {
                sum[i] += v[i] / towns;
            }
            memories = r.memories();
        }
        return new Result(sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], memories);
    }

    private static Result runTown(int days, int k, long seed) {
        World w = simulate(days, new SplittableRandom(seed));
        double now = days * 10;
        Function<String, float[]> embed = MemoryStream.hashedBagOfWords(64);

        MemoryStream stream = MemoryStream.of(embed, 64).importance(GenerativeAgentsRecall::importance).decay(0.995)
                .reflectAfter(150);
        MemoryStream relevanceOnly = MemoryStream.of(embed, 64).importance(GenerativeAgentsRecall::importance)
                .weights(0, 0, 1);
        VirtualContextPager pager = VirtualContextPager.of(6 * (k - k / 2), s -> s.split(" ").length, embed, 64)
                .thresholds(0.9, 0.6).summarizer(s -> "");
        for (int i = 0; i < w.log().size(); i++) {
            String s = w.log().get(i);
            double t = w.times().get(i);
            stream.observe(s, t);
            stream.maybeReflect(GenerativeAgentsRecall::reflect, t);
            relevanceOnly.observe(s, t);
            pager.add("observation", s);
            pager.archivalInsert(s);
        }
        List<String> log = w.log();
        int windowSize = k;   // every memory hands the reader the same number of records

        int[] streamRight = new int[3];
        int relRight = 0;
        int windowRight = 0;
        int pagerRight = 0;
        for (int type = 0; type < 3; type++) {
            for (String who : PEOPLE) {
                String q = question(type, who);
                String truth = truth(w, type, who);
                List<String> fromStream = stream.retrieve(q, k, now).stream().map(r -> r.memory().text()).toList();
                streamRight[type] += truth.equals(answer(type, who, fromStream)) ? 1 : 0;
                List<String> fromRel = relevanceOnly.retrieve(q, k, now).stream().map(r -> r.memory().text())
                        .toList();
                relRight += truth.equals(answer(type, who, fromRel)) ? 1 : 0;
                List<String> fromWindow = new ArrayList<>();
                for (int i = log.size() - 1; i >= Math.max(0, log.size() - windowSize); i--) {
                    fromWindow.add(log.get(i));
                }
                windowRight += truth.equals(answer(type, who, fromWindow)) ? 1 : 0;
                List<String> fromPager = new ArrayList<>();
                List<VirtualContextPager.Message> queue = pager.queue();
                for (int i = queue.size() - 1; i >= 0; i--) {
                    fromPager.add(queue.get(i).text());
                }
                fromPager.addAll(pager.archivalSearch(q, k / 2));
                pagerRight += truth.equals(answer(type, who, fromPager)) ? 1 : 0;
            }
        }
        double n = PEOPLE.length;
        return new Result(streamRight[0] / n, streamRight[1] / n, streamRight[2] / n, relRight / (3 * n),
                windowRight / (3 * n), pagerRight / (3 * n), (streamRight[0] + streamRight[1] + streamRight[2]) / (3 * n),
                stream.size());
    }
}
