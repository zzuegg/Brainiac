package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock.NormPlacement;
import io.github.zzuegg.brainiac.nn.Activation;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * The genes of a transformer design for evolutionary search: query heads, key/value
 * heads, head width, gated attention, normalization placement and type, depth, and one
 * activation per FFN neuron (so the FFN width is a gene too). Mutation changes one gene
 * at a time and always keeps the design valid ({@code kvHeads} divides {@code heads});
 * crossover takes every gene from either parent, and FFN neurons position by position.
 *
 * @param heads query heads
 * @param kvHeads key/value heads
 * @param headDim head width
 * @param gated sigmoid-gated attention output
 * @param norm normalization placement
 * @param rms RMS norm instead of layer norm
 * @param depth number of blocks
 * @param ffnActivations one activation per FFN neuron
 */
public record TransformerBlockGenome(int heads, int kvHeads, int headDim, boolean gated, NormPlacement norm,
        boolean rms, int depth, List<Activation> ffnActivations) {

    /** The activations FFN neurons can choose from. */
    public static final List<Activation> PALETTE = List.of(Activation.RELU, Activation.GELU, Activation.SILU,
            Activation.TANH, Activation.IDENTITY);

    private static final int[] HEADS = {1, 2, 4};
    private static final int[] HEAD_DIMS = {4, 8};
    private static final int MAX_FFN = 48;

    /** Checks and copies. */
    public TransformerBlockGenome {
        ffnActivations = List.copyOf(ffnActivations);
        if (heads % kvHeads != 0 || depth < 1 || ffnActivations.isEmpty()) {
            throw new IllegalArgumentException("invalid genome");
        }
        Objects.requireNonNull(norm);
    }

    /**
     * The block design this genome encodes.
     *
     * @param model the residual width
     * @return the block config
     */
    public GrowableTransformerBlock.Config config(int model) {
        return new GrowableTransformerBlock.Config(model, heads, kvHeads, headDim, gated, norm, rms, ffnActivations);
    }

    /** @return the FFN width */
    public int ffn() {
        return ffnActivations.size();
    }

    /**
     * A random valid genome.
     *
     * @param random the random source
     * @return the genome
     */
    public static TransformerBlockGenome random(SplittableRandom random) {
        int heads = HEADS[random.nextInt(HEADS.length)];
        int kv = divisor(heads, random);
        int ffn = 4 + random.nextInt(20);
        List<Activation> acts = new ArrayList<>();
        for (int i = 0; i < ffn; i++) {
            acts.add(PALETTE.get(random.nextInt(PALETTE.size())));
        }
        return new TransformerBlockGenome(heads, kv, HEAD_DIMS[random.nextInt(HEAD_DIMS.length)], random.nextBoolean(),
                NormPlacement.values()[random.nextInt(NormPlacement.values().length)], random.nextBoolean(),
                1 + random.nextInt(2), acts);
    }

    private static int divisor(int n, SplittableRandom random) {
        List<Integer> d = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            if (n % i == 0) {
                d.add(i);
            }
        }
        return d.get(random.nextInt(d.size()));
    }

    /**
     * A copy with one gene changed.
     *
     * @param random the random source
     * @return the mutant
     */
    public TransformerBlockGenome mutate(SplittableRandom random) {
        int h = heads;
        int kv = kvHeads;
        int hd = headDim;
        boolean g = gated;
        NormPlacement n = norm;
        boolean r = rms;
        int dep = depth;
        List<Activation> acts = new ArrayList<>(ffnActivations);
        switch (random.nextInt(8)) {
            case 0 -> {
                h = HEADS[random.nextInt(HEADS.length)];
                kv = divisor(h, random);
            }
            case 1 -> kv = divisor(h, random);
            case 2 -> hd = HEAD_DIMS[random.nextInt(HEAD_DIMS.length)];
            case 3 -> g = !g;
            case 4 -> {
                n = NormPlacement.values()[random.nextInt(NormPlacement.values().length)];
                r = random.nextBoolean();
            }
            case 5 -> dep = Math.max(1, Math.min(3, dep + (random.nextBoolean() ? 1 : -1)));
            case 6 -> {
                int k = 1 + random.nextInt(4);
                for (int i = 0; i < k; i++) {
                    if (random.nextBoolean() && acts.size() < MAX_FFN) {
                        acts.add(PALETTE.get(random.nextInt(PALETTE.size())));
                    } else if (acts.size() > 2) {
                        acts.remove(random.nextInt(acts.size()));
                    }
                }
            }
            default -> {
                int k = 1 + random.nextInt(3);
                for (int i = 0; i < k; i++) {
                    acts.set(random.nextInt(acts.size()), PALETTE.get(random.nextInt(PALETTE.size())));
                }
            }
        }
        return new TransformerBlockGenome(h, kv, hd, g, n, r, dep, acts);
    }

    /**
     * Uniform crossover: every gene from either parent; FFN neurons position by position,
     * the child's width from one parent.
     *
     * @param other the other parent
     * @param random the random source
     * @return the child
     */
    public TransformerBlockGenome cross(TransformerBlockGenome other, SplittableRandom random) {
        boolean headsFromMe = random.nextBoolean();
        int h = headsFromMe ? heads : other.heads;
        int kv = headsFromMe ? kvHeads : other.kvHeads;
        int width = random.nextBoolean() ? ffn() : other.ffn();
        List<Activation> acts = new ArrayList<>();
        for (int i = 0; i < width; i++) {
            boolean mine = i < ffn() && (i >= other.ffn() || random.nextBoolean());
            acts.add(mine ? ffnActivations.get(i) : other.ffnActivations.get(i));
        }
        return new TransformerBlockGenome(h, kv, random.nextBoolean() ? headDim : other.headDim,
                random.nextBoolean() ? gated : other.gated, random.nextBoolean() ? norm : other.norm,
                random.nextBoolean() ? rms : other.rms, random.nextBoolean() ? depth : other.depth, acts);
    }

    @Override
    public String toString() {
        java.util.Map<String, Integer> counts = new java.util.TreeMap<>();
        for (Activation a : ffnActivations) {
            counts.merge(a.name(), 1, Integer::sum);
        }
        return String.format("%dx[heads %d/%d x %d%s, %s %s, ffn %d %s]", depth, heads, kvHeads, headDim,
                gated ? " gated" : "", norm, rms ? "rms" : "layer", ffn(), counts);
    }
}
