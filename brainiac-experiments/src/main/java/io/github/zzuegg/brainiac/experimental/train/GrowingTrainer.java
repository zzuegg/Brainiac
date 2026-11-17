package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformer;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Trains a {@link GrowableTransformer} that starts tiny and grows when learning stalls,
 * the transformer version of {@code Plasticity}'s neurogenesis: when the smoothed training
 * loss has not improved by a set fraction for a while, the trainer grows the part the
 * gradients press on hardest (the attention or FFN of one block, by the RMS gradient of
 * their output projections), and every few growths adds depth instead. All growth is
 * function-preserving, so the loss does not jump; the optimizer is rebuilt for the new
 * parameters.
 *
 * <p>Background: Net2Net (<a href="https://arxiv.org/abs/1511.05641">arXiv:1511.05641</a>),
 * staged training of transformers (Shen et al. 2022,
 * <a href="https://arxiv.org/abs/2203.06211">arXiv:2203.06211</a>), and progressive
 * stacking (Gong et al. 2019).
 */
public final class GrowingTrainer {

    /**
     * A growth event.
     *
     * @param step the step after which it happened
     * @param kind {@code ffn}, {@code heads} or {@code depth}
     * @param block the block it touched
     * @param parameters the parameter count afterwards
     */
    public record Growth(long step, String kind, int block, long parameters) {
    }

    private final GrowableTransformer model;
    private final Function<SplittableRandom, Batch> data;
    private Supplier<Optimizer> optimizer = () -> Optimizer.adam(3e-3f);
    private int checkEvery = 25;
    private int patience = 2;
    private double minImprovement = 0.02;
    private int depthEvery = 3;
    private int maxFfn = 64;
    private int maxHeads = 4;
    private int maxBlocks = 3;
    private boolean growing = true;
    private long seed = 1;
    private final List<Growth> growths = new ArrayList<>();
    private final List<Float> losses = new ArrayList<>();

    private GrowingTrainer(GrowableTransformer model, Function<SplittableRandom, Batch> data) {
        this.model = Objects.requireNonNull(model);
        this.data = Objects.requireNonNull(data);
    }

    /**
     * Sets up training.
     *
     * @param model the model, usually tiny
     * @param data draws a batch of token ids {@code [batch, time]} and next-token targets
     * @return the trainer, to configure
     */
    public static GrowingTrainer of(GrowableTransformer model, Function<SplittableRandom, Batch> data) {
        return new GrowingTrainer(model, data);
    }

    /**
     * Sets the optimizer factory (called again after every growth).
     *
     * @param factory makes a fresh optimizer
     * @return this
     */
    public GrowingTrainer optimizer(Supplier<Optimizer> factory) {
        this.optimizer = factory;
        return this;
    }

    /**
     * Sets when to grow.
     *
     * @param every steps between plateau checks
     * @param checks checks without enough improvement before growing
     * @param improvement the relative improvement that counts
     * @return this
     */
    public GrowingTrainer plateau(int every, int checks, double improvement) {
        this.checkEvery = every;
        this.patience = checks;
        this.minImprovement = improvement;
        return this;
    }

    /**
     * Sets the size limits.
     *
     * @param ffn the widest FFN
     * @param heads the most query heads per block
     * @param depth the most blocks
     * @param depthAfter add a block after this many width growths
     * @return this
     */
    public GrowingTrainer limits(int ffn, int heads, int depth, int depthAfter) {
        this.maxFfn = ffn;
        this.maxHeads = heads;
        this.maxBlocks = depth;
        this.depthEvery = depthAfter;
        return this;
    }

    /**
     * Turns growth off, for a fixed-size baseline.
     *
     * @param on whether to grow
     * @return this
     */
    public GrowingTrainer growing(boolean on) {
        this.growing = on;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public GrowingTrainer seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the growth events */
    public List<Growth> growths() {
        return List.copyOf(growths);
    }

    /** @return the training loss of every step */
    public List<Float> losses() {
        return List.copyOf(losses);
    }

    /**
     * Trains.
     *
     * @param steps the number of optimizer steps
     * @return the model
     */
    public GrowableTransformer train(int steps) {
        SplittableRandom random = new SplittableRandom(seed);
        Optimizer opt = optimizer.get();
        List<Var> params = model.parameters();
        double ema = Double.NaN;
        double best = Double.POSITIVE_INFINITY;
        int stalls = 0;
        int widthGrowths = 0;
        int blocks = model.blocks().size();
        double[][] pressure = new double[blocks][2];
        for (int step = 1; step <= steps; step++) {
            Batch b = data.apply(random);
            Var loss = GrowableTransformer.loss(model.forward(Var.of(b.features())), b.targets());
            loss.backward();
            float l = loss.value().item();
            losses.add(l);
            ema = Double.isNaN(ema) ? l : 0.9 * ema + 0.1 * l;
            List<GrowableTransformerBlock> bl = model.blocks();
            for (int i = 0; i < bl.size(); i++) {
                double[] p = bl.get(i).growthPressure();
                pressure[i][0] = 0.9 * pressure[i][0] + 0.1 * p[0];
                pressure[i][1] = 0.9 * pressure[i][1] + 0.1 * p[1];
            }
            opt.step(params);
            if (!growing || step % checkEvery != 0) {
                continue;
            }
            if (ema < best * (1 - minImprovement)) {
                best = ema;
                stalls = 0;
                continue;
            }
            if (++stalls < patience) {
                continue;
            }
            stalls = 0;
            best = ema;
            Growth g = grow(random, pressure, widthGrowths, step);
            if (g == null) {
                continue;
            }
            growths.add(g);
            widthGrowths = g.kind().equals("depth") ? 0 : widthGrowths + 1;
            if (g.kind().equals("depth")) {
                double[][] np = new double[model.blocks().size()][2];
                for (int i = 0; i < pressure.length; i++) {
                    np[i] = pressure[i];
                }
                pressure = np;
            }
            opt = optimizer.get();
            params = model.parameters();
        }
        return model;
    }

    private Growth grow(SplittableRandom random, double[][] pressure, int widthGrowths, long step) {
        List<GrowableTransformerBlock> bl = model.blocks();
        if (widthGrowths >= depthEvery && bl.size() < maxBlocks) {
            GrowableTransformerBlock.Config top = bl.getLast().config();
            model.addBlock(bl.size(), top, random);
            return new Growth(step, "depth", bl.size(), model.parameterCount());
        }
        int bestBlock = -1;
        int bestKind = -1;
        double bestPressure = -1;
        for (int i = 0; i < bl.size(); i++) {
            GrowableTransformerBlock.Config c = bl.get(i).config();
            boolean canHeads = c.heads() + c.heads() / c.kvHeads() <= maxHeads;
            boolean canFfn = c.ffn() < maxFfn;
            // pressure per unit of current size: smaller parts grow first at equal pressure
            double ph = canHeads ? pressure[i][0] : -1;
            double pf = canFfn ? pressure[i][1] : -1;
            if (ph > bestPressure) {
                bestPressure = ph;
                bestBlock = i;
                bestKind = 0;
            }
            if (pf > bestPressure) {
                bestPressure = pf;
                bestBlock = i;
                bestKind = 1;
            }
        }
        if (bestBlock < 0) {
            if (bl.size() < maxBlocks) {
                model.addBlock(bl.size(), bl.getLast().config(), random);
                return new Growth(step, "depth", bl.size(), model.parameterCount());
            }
            return null;
        }
        if (bestKind == 0) {
            model.addHeadGroup(bestBlock, random);
            return new Growth(step, "heads", bestBlock, model.parameterCount());
        }
        int width = Math.min(maxFfn, bl.get(bestBlock).config().ffn() * 2);
        model.widenFfn(bestBlock, width, random);
        return new Growth(step, "ffn", bestBlock, model.parameterCount());
    }
}
