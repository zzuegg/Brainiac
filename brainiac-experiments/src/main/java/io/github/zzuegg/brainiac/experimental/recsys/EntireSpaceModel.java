package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * ESMM, the entire-space multi-task model (Ma et al., SIGIR 2018,
 * <a href="https://arxiv.org/abs/1804.07931">arXiv:1804.07931</a>), as a {@link Layer} from
 * features to {@code [batch, 2]} logits {@code (CTR, CVR)}.
 *
 * <p>A conversion-rate model is usually trained on clicked impressions only but served on all
 * impressions: sample selection bias, and very sparse labels. ESMM never fits CVR directly. Over
 * <em>all</em> impressions it fits {@code pCTR} to the click and {@code pCTCVR = pCTR * pCVR} to
 * "clicked and converted"; because the CTR and CVR towers share their body (here a two-task
 * {@link MultiGateExperts}, shared-bottom or MMoE), {@code pCVR} is learned implicitly over the
 * whole impression space. {@link #loss} is the ESMM loss; {@link #clickSpaceLoss} the usual
 * click-space CVR loss (optionally inverse-propensity weighted) for the baselines.
 */
public final class EntireSpaceModel implements Layer {

    private final MultiGateExperts body;

    /**
     * Wraps a two-task body: task 0 is CTR, task 1 is CVR.
     *
     * @param body the shared body with two tasks
     */
    public EntireSpaceModel(MultiGateExperts body) {
        if (body.tasks() != 2) {
            throw new IllegalArgumentException("ESMM needs a two-task body (CTR, CVR)");
        }
        this.body = body;
    }

    /**
     * @param kind the body's architecture
     * @param experts experts of the body
     * @param hidden hidden width
     * @param expertDim expert output width
     * @return a spec for model builders
     */
    public static LayerSpec spec(MultiGateExperts.Kind kind, int experts, int hidden, int expertDim) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(2);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new EntireSpaceModel(new MultiGateExperts(kind, input.dim(-1), 2, experts, hidden, expertDim, 1,
                        random));
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return body.forward(input);
    }

    @Override
    public List<Var> parameters() {
        return body.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return 2;
    }

    /**
     * The ESMM loss over all impressions:
     * {@code BCE(pCTR, click) + BCE(pCTR * pCVR, click * convert)}.
     *
     * @param logits {@code [batch, 2]} from {@link #forward}
     * @param click 0/1 clicks {@code [batch, 1]}
     * @param clickAndConvert 0/1 "clicked and converted" {@code [batch, 1]}
     * @return the scalar loss
     */
    public static Var loss(Var logits, Tensor click, Tensor clickAndConvert) {
        Var ctr = logits.slice(1, 0, 1);
        Var cvr = logits.slice(1, 1, 2);
        Var ctrLoss = ctr.softplus().sub(ctr.mul(click)).mean();
        // log p = -softplus(-a) - softplus(-b); log(1 - p) from p itself, clamped
        Var logP = ctr.neg().softplus().add(cvr.neg().softplus()).neg();
        Var p = logP.exp().clamp(0f, 1f - 1e-6f);
        Var log1mP = p.neg().add(1f).log();
        Var ctcvrLoss = logP.mul(clickAndConvert).add(log1mP.mul(clickAndConvert.neg().add(1f))).mean().neg();
        return ctrLoss.add(ctcvrLoss);
    }

    /**
     * The click-space baseline loss: CTR on all impressions plus CVR on clicked impressions only,
     * each clicked row weighted by {@code weight} (1, or {@code 1 / pCTR} for inverse propensity
     * weighting, normalised to mean 1 over the clicked rows).
     *
     * @param logits {@code [batch, 2]}
     * @param click 0/1 clicks {@code [batch, 1]}
     * @param convert 0/1 conversions {@code [batch, 1]} (used where clicked)
     * @param weight per-row CVR weights {@code [batch, 1]}
     * @return the scalar loss
     */
    public static Var clickSpaceLoss(Var logits, Tensor click, Tensor convert, Tensor weight) {
        Var ctr = logits.slice(1, 0, 1);
        Var cvr = logits.slice(1, 1, 2);
        Var ctrLoss = ctr.softplus().sub(ctr.mul(click)).mean();
        Tensor w = weight.mul(click);
        float total = Math.max(1e-6f, w.sum().item());
        Var cvrLoss = cvr.softplus().sub(cvr.mul(convert)).mul(w).sum().div(total);
        return ctrLoss.add(cvrLoss);
    }

    @Override
    public String toString() {
        return "EntireSpaceModel(" + body + ")";
    }
}
