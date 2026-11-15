package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

/**
 * A whole recurrent pass as one operation with a hand-written backward pass through
 * time, so long sequences cost time proportional to their length. Inputs: the projected
 * inputs {@code xp [batch, steps, gates * hidden]} (bias included), the recurrent weights
 * {@code [hidden, gates * hidden]}, the initial state {@code [batch, hidden]}, then for
 * LSTMs the initial cell {@code [batch, hidden]} or for GRUs with reset-after the
 * recurrent bias of the candidate {@code [hidden]}. Output: every hidden state
 * {@code [batch, steps, hidden]}.
 *
 * <p>One instance runs once: it keeps what the forward pass computed for the backward pass.
 */
final class RecurrentOp implements Op {

    private final Recurrent.Cell cell;
    private final ActivationOp act;
    private final ActivationOp gate;
    private final boolean resetAfter;

    private int steps;
    private int hidden;
    private Tensor[] hPrev;
    private Tensor[] cPrev;
    private Tensor[] cells;
    private Tensor[][] z;       // per step: pre-activations of each gate
    private Tensor[] extra;     // per step: GRU hn (+ bias) or r * h

    RecurrentOp(Recurrent.Cell cell, ActivationOp act, ActivationOp gate, boolean resetAfter) {
        this.cell = cell;
        this.act = act;
        this.gate = gate;
        this.resetAfter = resetAfter;
    }

    @Override
    public String name() {
        return cell.name().toLowerCase(java.util.Locale.ROOT);
    }

    private static Tensor step(Tensor xp, int t, int width) {
        int n = xp.shape().dim(0);
        return xp.slice(1, t, t + 1).reshape(n, width);
    }

    private Tensor part(Tensor m, int k) {
        return m.slice(1, k * hidden, (k + 1) * hidden);
    }

    @Override
    public Tensor forward(Tensor... in) {
        Tensor xp = in[0];
        Tensor wh = in[1];
        Tensor h = in[2];
        int n = xp.shape().dim(0);
        steps = xp.shape().dim(1);
        hidden = wh.shape().dim(0);
        int width = xp.shape().dim(2);
        hPrev = new Tensor[steps];
        z = new Tensor[steps][];
        Tensor[] outs = new Tensor[steps];
        Tensor c = cell == Recurrent.Cell.LSTM ? in[3] : null;
        if (c != null) {
            cPrev = new Tensor[steps];
            cells = new Tensor[steps];
        }
        if (cell == Recurrent.Cell.GRU) {
            extra = new Tensor[steps];
        }
        for (int t = 0; t < steps; t++) {
            Tensor x = step(xp, t, width);
            hPrev[t] = h;
            switch (cell) {
                case RNN -> {
                    Tensor zt = x.add(h.matmul(wh));
                    z[t] = new Tensor[] {zt};
                    h = zt.activation(act, 0);
                }
                case LSTM -> {
                    Tensor zt = x.add(h.matmul(wh));
                    Tensor zi = part(zt, 0);
                    Tensor zf = part(zt, 1);
                    Tensor zg = part(zt, 2);
                    Tensor zo = part(zt, 3);
                    z[t] = new Tensor[] {zi, zf, zg, zo};
                    cPrev[t] = c;
                    c = zf.activation(gate, 0).mul(c).add(zi.activation(gate, 0).mul(zg.activation(act, 0)));
                    cells[t] = c;
                    h = zo.activation(gate, 0).mul(c.activation(act, 0));
                }
                case GRU -> {
                    Tensor xr = part(x, 0);
                    Tensor xu = part(x, 1);
                    Tensor xn = part(x, 2);
                    Tensor zr;
                    Tensor zu;
                    Tensor zn;
                    if (resetAfter) {
                        Tensor hw = h.matmul(wh);
                        zr = xr.add(part(hw, 0));
                        zu = xu.add(part(hw, 1));
                        Tensor hn = part(hw, 2).add(in[3]);
                        extra[t] = hn;
                        zn = xn.add(zr.activation(gate, 0).mul(hn));
                    } else {
                        Tensor ru = h.matmul(wh.slice(1, 0, 2 * hidden));
                        zr = xr.add(part(ru, 0));
                        zu = xu.add(part(ru, 1));
                        Tensor rh = zr.activation(gate, 0).mul(h);
                        extra[t] = rh;
                        zn = xn.add(rh.matmul(wh.slice(1, 2 * hidden, 3 * hidden)));
                    }
                    z[t] = new Tensor[] {zr, zu, zn};
                    Tensor u = zu.activation(gate, 0);
                    Tensor nn = zn.activation(act, 0);
                    h = nn.add(u.mul(h.sub(nn)));
                }
            }
            outs[t] = h.reshape(n, 1, hidden);
        }
        return Tensor.concat(1, outs);
    }

    @Override
    public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
        Tensor wh = in[1];
        int n = grad.shape().dim(0);
        Tensor dWh = Tensor.zeros(wh.shape(), wh.device());
        Tensor dh = Tensor.zeros(Shape.of(n, hidden), wh.device());
        Tensor dc = cell == Recurrent.Cell.LSTM ? Tensor.zeros(Shape.of(n, hidden), wh.device()) : null;
        Tensor dExtra = in.length > 3 && cell == Recurrent.Cell.GRU
                ? Tensor.zeros(in[3].shape(), wh.device()) : null;
        Tensor[] dxp = new Tensor[steps];
        Tensor whT = wh.transpose();
        for (int t = steps - 1; t >= 0; t--) {
            dh = dh.add(step(grad, t, hidden));
            Tensor h = hPrev[t];
            Tensor dz;
            switch (cell) {
                case RNN -> {
                    dz = z[t][0].activationBackward(act, 0, dh);
                    dWh = dWh.add(h.transpose().matmul(dz));
                    dh = dz.matmul(whT);
                }
                case LSTM -> {
                    Tensor zi = z[t][0];
                    Tensor zf = z[t][1];
                    Tensor zg = z[t][2];
                    Tensor zo = z[t][3];
                    Tensor c = cells[t];
                    Tensor o = zo.activation(gate, 0);
                    Tensor dcTotal = dc.add(c.activationBackward(act, 0, dh.mul(o)));
                    Tensor dzo = zo.activationBackward(gate, 0, dh.mul(c.activation(act, 0)));
                    Tensor dzi = zi.activationBackward(gate, 0, dcTotal.mul(zg.activation(act, 0)));
                    Tensor dzg = zg.activationBackward(act, 0, dcTotal.mul(zi.activation(gate, 0)));
                    Tensor dzf = zf.activationBackward(gate, 0, dcTotal.mul(cPrev[t]));
                    dc = dcTotal.mul(zf.activation(gate, 0));
                    dz = Tensor.concat(1, dzi, dzf, dzg, dzo);
                    dWh = dWh.add(h.transpose().matmul(dz));
                    dh = dz.matmul(whT);
                }
                case GRU -> {
                    Tensor zr = z[t][0];
                    Tensor zu = z[t][1];
                    Tensor zn = z[t][2];
                    Tensor u = zu.activation(gate, 0);
                    Tensor nn = zn.activation(act, 0);
                    Tensor r = zr.activation(gate, 0);
                    Tensor dnn = dh.mul(u.neg().add(1));
                    Tensor du = dh.mul(h.sub(nn));
                    Tensor dhPrev = dh.mul(u);
                    Tensor dzn = zn.activationBackward(act, 0, dnn);
                    Tensor dzu = zu.activationBackward(gate, 0, du);
                    if (resetAfter) {
                        Tensor hn = extra[t];
                        Tensor dzr = zr.activationBackward(gate, 0, dzn.mul(hn));
                        Tensor dhn = dzn.mul(r);
                        dExtra = dExtra.add(dhn.sum(0));
                        Tensor dhw = Tensor.concat(1, dzr, dzu, dhn);
                        dWh = dWh.add(h.transpose().matmul(dhw));
                        dh = dhPrev.add(dhw.matmul(whT));
                        dz = Tensor.concat(1, dzr, dzu, dzn);
                    } else {
                        Tensor rh = extra[t];
                        Tensor whn = wh.slice(1, 2 * hidden, 3 * hidden);
                        Tensor drh = dzn.matmul(whn.transpose());
                        Tensor dzr = zr.activationBackward(gate, 0, drh.mul(h));
                        Tensor dru = Tensor.concat(1, dzr, dzu);
                        Tensor wru = wh.slice(1, 0, 2 * hidden);
                        dWh = dWh.add(Tensor.concat(1, h.transpose().matmul(dru), rh.transpose().matmul(dzn)));
                        dh = dhPrev.add(drh.mul(r)).add(dru.matmul(wru.transpose()));
                        dz = Tensor.concat(1, dzr, dzu, dzn);
                    }
                }
                default -> throw new IllegalStateException();
            }
            dxp[t] = dz.reshape(n, 1, dz.shape().dim(1));
        }
        Tensor[] grads = new Tensor[in.length];
        grads[0] = Tensor.concat(1, dxp);
        grads[1] = dWh;
        grads[2] = dh;
        if (in.length > 3) {
            grads[3] = cell == Recurrent.Cell.LSTM ? dc : dExtra;
        }
        return grads;
    }
}
