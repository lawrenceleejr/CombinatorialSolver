"""
JetAssign: per-jet parent assignment for pair-produced resonances, jet 4-vectors only.

PairNet picks one of 70 fixed 3+3 groupings among the leading 7 jets.  With truth
from the parton ancestry of every jet (scripts/generate_pythia.py --ancestry), only
about half of the signal that could sit in the mass peak is reachable that way:
FSR jets, merged quark pairs and decay quarks beyond the 7th jet are lost.  Here each
of the (up to 12) jets is labelled parent 1 / parent 2 / other, groups have free sizes,
and the group four-vectors are sums over the jets of each label.

  * Same encoder as PairNet (canonical frame, momenta / HT, pairwise attention bias).
  * Labels are made unambiguous by convention: parent 1 owns the leading jet that has
    a parent at all (loader: canonical_anc).  The network learns the same convention.
  * Group four-vectors are soft (label probabilities) in training and hard (argmax)
    for the search variables m1, m2, m_avg and the asymmetry.
  * A gate sees the pooled event plus the soft groups (masses, asymmetry, embeddings)
    and gives P(signal); a mass head gives log(M/HT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pairnet import PairBiasEncoderLayer, asinh, canonicalize, cosh, minkowski_m2, to_cartesian


def _group_kin(P, m):
    """ln m, ln pT/m, eta-ish (asinh pz/pT) of a group four-vector (HT units)."""
    pt = torch.sqrt(P[..., 1] ** 2 + P[..., 2] ** 2).clamp(min=1e-6)
    lm = torch.log(m.clamp(min=1e-4))
    rap = asinh(P[..., 3] / pt)
    return torch.stack([lm, torch.log(pt) - lm, rap], -1)


class JetAssign(nn.Module):
    def __init__(self, n_jets=12, d_model=128, nhead=8, num_layers=4, dim_ff=256, dropout=0.1,
                 massless=False, refine=1, pair_head=False):
        super().__init__()
        self.pair_head = pair_head
        self.n_jets, self.d, self.nhead = n_jets, d_model, nhead
        self.massless = massless
        self.refine = refine
        self.embed = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pair_mlp = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, nhead * num_layers))
        self.layers = nn.ModuleList(
            [PairBiasEncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        # per-jet label head; refinement passes see the current soft groups
        self.label_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        n_grp = 2 * 3 + 2  # group kin x2, asym, ln m_avg
        self.ref_in = nn.Linear(d_model + 3 + 2 * d_model + n_grp, d_model)
        self.ref_layers = nn.ModuleList(
            [PairBiasEncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(refine)])
        self.ref_pair = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, nhead * max(refine, 1)))
        self.ref_ln = nn.LayerNorm(d_model)
        self.ref_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))
        if pair_head:
            # P(same parent) for every jet pair and P(not from a parent) per jet; the search groups are the
            # exact best 2-colouring of the parent jets under the pair log-odds (decode)
            self.pair_out = nn.Sequential(nn.Linear(2 * d_model + 4, d_model), nn.GELU(), nn.Linear(d_model, 1))
            self.other_out = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
            k = torch.arange(2 ** (n_jets - 1))
            bits = ((k.unsqueeze(-1) >> torch.arange(n_jets - 1)) & 1).float()
            S = torch.cat([torch.ones(len(k), 1), 1 - 2 * bits], -1)  # (K, N), jet 0 fixed to +1
            self.register_buffer("colourings", S, persistent=False)
        # gate and mass head
        n_ev = 2 * d_model + 2 * d_model + d_model + n_grp + 1
        self.gate_mlp = nn.Sequential(nn.Linear(n_ev, d_model), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.mass_mlp = nn.Sequential(nn.Linear(n_ev, d_model), nn.GELU(),
                                      nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def _jet_features(self, pt, eta, phi, m, mask):
        lpt = torch.log(pt.clamp(min=1e-6))
        e = torch.sqrt((pt * cosh(eta)) ** 2 + m**2)
        le = torch.log(e.clamp(min=1e-6))
        f = torch.stack([lpt, eta, torch.cos(phi), torch.sin(phi), le, m / pt.clamp(min=1e-6), lpt - le], -1)
        return f * mask.unsqueeze(-1)

    def _pair_features(self, p4, pt, eta, phi, mask):
        deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
        dphi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
        dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
        dr = torch.sqrt(deta**2 + dphi**2).clamp(min=1e-4)
        ptmin = torch.minimum(pt.unsqueeze(-1), pt.unsqueeze(-2))
        kt = ptmin * dr
        z = ptmin / (pt.unsqueeze(-1) + pt.unsqueeze(-2)).clamp(min=1e-6)
        m2 = minkowski_m2(p4.unsqueeze(2) + p4.unsqueeze(1)).clamp(min=1e-8)
        f = torch.stack([torch.log(dr), torch.log(kt.clamp(min=1e-6)), torch.log(z.clamp(min=1e-6)),
                         torch.log(m2)], -1)
        pm = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
        return f * pm

    @staticmethod
    def groups(q, p4):
        """q (B, N, 3) label weights -> group four-vectors, masses (HT units), asymmetry."""
        P1 = torch.einsum("bn,bnc->bc", q[..., 0], p4)
        P2 = torch.einsum("bn,bnc->bc", q[..., 1], p4)
        m1 = torch.sqrt(minkowski_m2(P1).clamp(min=1e-8))
        m2 = torch.sqrt(minkowski_m2(P2).clamp(min=1e-8))
        asym = (m1 - m2).abs() / (m1 + m2).clamp(min=1e-6)
        return P1, P2, m1, m2, asym

    def _grp_feats(self, q, p4, x):
        P1, P2, m1, m2, asym = self.groups(q, p4)
        e1 = torch.einsum("bn,bnd->bd", q[..., 0], x)
        e2 = torch.einsum("bn,bnd->bd", q[..., 1], x)
        er = torch.einsum("bn,bnd->bd", q[..., 2], x)
        k1, k2 = _group_kin(P1, m1), _group_kin(P2, m2)
        # symmetric in the two parents
        kin = torch.cat([k1 + k2, (k1 - k2).abs(), asym.unsqueeze(-1),
                         torch.log((0.5 * (m1 + m2)).clamp(min=1e-4)).unsqueeze(-1)], -1)
        return torch.cat([e1 + e2, e1 * e2], -1), er, kin

    def decode(self, pair_logits, other_logit, mask, chunk=1024):
        """Exact best split of the candidate jets (P(other) < 0.5) into two non-empty groups,
        maximising the summed pair log-odds of same-group pairs.  Returns hard labels (B, N, 3)."""
        c = (mask & (other_logit < 0)).float()
        A = pair_logits * c.unsqueeze(-1) * c.unsqueeze(-2)
        A = A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
        S = self.colourings
        best = []
        for i in range(0, len(A), chunk):
            Ab, cb = A[i:i + chunk], c[i:i + chunk]
            sc = torch.einsum("kn,bnm,km->bk", S, Ab, S)
            n1 = cb @ ((1 + S) / 2).T
            ok = (n1 > 0.5) & (cb.sum(-1, keepdim=True) - n1 > 0.5)
            best.append(sc.masked_fill(~ok, -1e9).argmax(-1))
        s = S[torch.cat(best)]  # (B, N)
        g1 = c * (s > 0).float()
        g2 = c * (s < 0).float()
        return torch.stack([g1, g2, mask.float() - g1 - g2], -1)

    def _encode(self, x, pf, mask, layers, pair_mlp):
        B, N, _ = x.shape
        neg = torch.zeros(B, 1, N, device=x.device).masked_fill(~mask.unsqueeze(1), float("-inf"))
        pb = pair_mlp(pf).view(B, N, N, len(layers), self.nhead).permute(3, 0, 4, 1, 2)
        for li, layer in enumerate(layers):
            bias = neg.unsqueeze(1).expand(B, self.nhead, N, N) + pb[li]
            x = layer(x, bias.reshape(B * self.nhead, N, N), None)
        return x

    def forward(self, pt, eta, phi, m, mask):
        """pt/eta/phi/m (B, N) pT-sorted, zero-padded; mask (B, N) bool.

        Returns label_logits (B, N, 3), q (soft labels), hard labels, m1/m2/asym (hard groups, HT
        units), ht, gate_logit, log_m_over_ht."""
        if self.massless:
            m = torch.zeros_like(m)
        pt, eta, phi, m, ht = canonicalize(pt, eta, phi, m, mask)
        p4 = to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1)
        pf = self._pair_features(p4, pt, eta, phi, mask)
        x = self.embed(self._jet_features(pt, eta, phi, m, mask))
        x = self.ln_out(self._encode(x, pf, mask, self.layers, self.pair_mlp)) * mask.unsqueeze(-1)
        lab = self.label_mlp(x)
        outs = [lab]
        if self.refine:  # one refinement pass of `refine` layers that sees the current soft groups
            q = lab.softmax(-1) * mask.unsqueeze(-1)
            sym, er, kin = self._grp_feats(q, p4, x)
            B, N, _ = x.shape
            ctx = torch.cat([sym, kin], -1).unsqueeze(1).expand(B, N, -1)
            h = self.ref_in(torch.cat([x, q, ctx], -1))
            h = self.ref_ln(self._encode(h, pf, mask, self.ref_layers, self.ref_pair)) * mask.unsqueeze(-1)
            lab = lab + self.ref_head(h)
            x = x + h
            outs.append(lab)
        q = lab.softmax(-1) * mask.unsqueeze(-1)
        hard = F.one_hot(lab.argmax(-1), 3).float() * mask.unsqueeze(-1)
        sym, er, kin = self._grp_feats(q, p4, x)
        xm = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        pooled = torch.cat([x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1), xm], -1)
        nj = mask.float().sum(1, keepdim=True) / self.n_jets
        ev = torch.cat([pooled, sym, er, kin, nj], -1)
        extra = {}
        if self.pair_head:
            B, N, d = x.shape
            xi, xj = x.unsqueeze(2).expand(B, N, N, d), x.unsqueeze(1).expand(B, N, N, d)
            pl = self.pair_out(torch.cat([xi * xj, xi + xj, pf], -1)).squeeze(-1)
            pl = 0.5 * (pl + pl.transpose(1, 2))
            ol = self.other_out(x).squeeze(-1)
            extra = {"pair_logits": pl, "other_logit": ol}
            with torch.no_grad():
                hard = self.decode(pl.float(), ol.float(), mask)
        _, _, m1, m2, asym = self.groups(hard, p4)
        _, _, sm1, sm2, _ = self.groups(q, p4)
        lm_soft = torch.log((0.5 * (sm1 + sm2)).clamp(min=1e-4))
        return {
            "label_logits": lab, "label_logits_all": outs, "q": q, "hard": hard,
            "m1": m1, "m2": m2, "asym": asym, "sm1": sm1, "sm2": sm2, "ht": ht,
            "gate_logit": self.gate_mlp(ev).squeeze(-1),
            "log_m_over_ht": lm_soft + self.mass_mlp(ev).squeeze(-1), **extra,
        }


def jetassign_loss(out, labels, is_sig, mask, pmass, bkg_weight=1.0, w_gate=1.0, w_mass=1.0,
                   w_grpmass=0.0, w_win=0.0, w_qcd_pass=0.0, window=(0.85, 1.1), asym_cut=0.1, tau=0.03,
                   tau_a=0.01, w_pair=1.0):
    """labels (B, N, 3): soft truth fractions (parent 1 / parent 2 / other) in canonical order;
    rows of all zeros (padding, unlabelled) are ignored.  Signal events with no labelled jet use only
    the gate and mass terms.

      label CE (signal, every refinement pass)  +  w_gate * gate BCE (QCD weighted by bkg_weight)
      + w_mass * |log m_est - log M| (signal)  + w_grpmass * soft |log m_i - log M| (signal)
      - w_win * smooth fraction of signal with hard-ish groups in the window and A < asym_cut
      + w_qcd_pass * smooth fraction of QCD with A < asym_cut
    """
    sig = is_sig.bool()
    has = labels.sum(-1) > 0.5
    jw = (has & mask & sig.unsqueeze(-1)).float()
    tgt = labels / labels.sum(-1, keepdim=True).clamp(min=1e-6)
    lce = 0.0
    for lab in out["label_logits_all"]:
        ce = -(tgt * F.log_softmax(lab.float(), -1)).sum(-1)
        lce = lce + (ce * jw).sum() / jw.sum().clamp(min=1)
    lce = lce / len(out["label_logits_all"])
    if "pair_logits" in out:
        dom = tgt.argmax(-1)
        par = (dom < 2) & has & mask & sig.unsqueeze(-1)
        pw = (par.unsqueeze(-1) & par.unsqueeze(-2)).float()
        pw = pw * (1 - torch.eye(pw.shape[-1], device=pw.device))
        same = (dom.unsqueeze(-1) == dom.unsqueeze(-2)).float()
        lp = F.binary_cross_entropy_with_logits(out["pair_logits"].float(), same, reduction="none")
        lpair = (lp * pw).sum() / pw.sum().clamp(min=1)
        lo = F.binary_cross_entropy_with_logits(out["other_logit"].float(), (dom == 2).float(), reduction="none")
        lother = (lo * jw).sum() / jw.sum().clamp(min=1)
        lce = lce + w_pair * (lpair + lother)
    w = torch.where(sig, 1.0, bkg_weight)
    lgate = F.binary_cross_entropy_with_logits(out["gate_logit"], sig.float(), weight=w, reduction="sum") / w.sum()
    loss = lce + w_gate * lgate
    logs = {"lce": float(lce.detach()), "lgate": float(lgate.detach())}
    if sig.any():
        lM = torch.log(pmass.clamp(min=1.0) / out["ht"])
        lmass = (out["log_m_over_ht"] - lM).abs()[sig].mean()
        loss = loss + w_mass * lmass
        logs["lmass"] = float(lmass.detach())
        if w_grpmass or w_win:
            sm1, sm2 = out["sm1"], out["sm2"]
            if w_grpmass:
                lg = ((torch.log(sm1.clamp(min=1e-4)) - lM).abs() + (torch.log(sm2.clamp(min=1e-4)) - lM).abs())[sig].mean()
                loss = loss + w_grpmass * lg
            if w_win:
                rel = 0.5 * (sm1 + sm2) / torch.exp(lM)
                a = (sm1 - sm2).abs() / (sm1 + sm2).clamp(min=1e-6)
                pw = torch.sigmoid((asym_cut - a) / tau_a) * torch.sigmoid((rel - window[0]) / tau) \
                    * torch.sigmoid((window[1] - rel) / tau)
                loss = loss - w_win * pw[sig].mean()
    if w_qcd_pass and (~sig).any():
        sm1, sm2 = out["sm1"], out["sm2"]
        a = (sm1 - sm2).abs() / (sm1 + sm2).clamp(min=1e-6)
        loss = loss + w_qcd_pass * torch.sigmoid((asym_cut - a) / tau_a)[~sig].mean()
    return loss, logs


def load_jetassign(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    a = ck["args"]
    net = JetAssign(n_jets=a["n_jets"], d_model=a["d_model"], nhead=a["heads"], num_layers=a["layers"],
                    dim_ff=2 * a["d_model"], dropout=0.0, massless=a.get("massless", False),
                    refine=a.get("refine", 1), pair_head=a.get("pair_head", False))
    net.load_state_dict(ck["state"])
    return net.eval()
