"""
PairNet: interpretation ranker for pair-produced resonances, jet 4-vectors only.

Question the network answers, per event:
  "Of all ways to pick two disjoint groups of k jets from the leading N jets,
   which one looks most like pair production of two equal-mass resonances,
   and does ANY of them look like it at all?"

Design (see docs/REVIEW.md for the reasoning):
  * Inputs are only jet four-vectors (pt, eta, phi, m).  The event is put in a
    canonical frame (leading jet at phi=0, eta>=0) and every momentum scale is
    divided by HT, so the network sees shapes, not absolute masses.
  * Jet tokens are encoded by a Transformer whose attention is biased by
    Lorentz-invariant pairwise features (ln dR, ln kT, ln z, ln m_ij^2), as in
    the Particle Transformer.  These are functions of the 4-vectors only.
  * Every interpretation (G1, G2, rest) is scored by a symmetric scorer built
    from: summed jet embeddings of each group, the group 4-vector kinematics
    (mass, boost, internal pairwise-mass fractions), the inter-group relations
    (mass asymmetry, angular separation) and the leftover jets.
  * One extra "null" logit, from the pooled event, means "no interpretation is
    pair production".  A single softmax over [interpretations..., null] gives
      - the best interpretation (argmax over interpretations), and
      - an event-level signal probability  P(sig) = 1 - p(null).
    QCD is trained to land on null; signal on its true interpretation.  The
    network never has to hide QCD inside asymmetric interpretations.
"""

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


def enumerate_pairings(n_jets: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All unordered pairs of disjoint k-subsets of range(n_jets).

    Returns (g1, g2) index tensors of shape (A, k); g1[a] < g2[a] lexicographically.
    """
    g1, g2 = [], []
    for a in combinations(range(n_jets), k):
        rest = [j for j in range(n_jets) if j not in a]
        for b in combinations(rest, k):
            if a < b:
                g1.append(a)
                g2.append(b)
    return torch.tensor(g1, dtype=torch.long), torch.tensor(g2, dtype=torch.long)


def enumerate_interpretations(n_jets: int, sizes=((3, 3),)) -> list[tuple[tuple, tuple]]:
    """All unordered pairs of disjoint jet sets (G1, G2) with (|G1|, |G2|) in sizes.

    Canonical order: the larger group first; equal sizes ordered lexicographically.
    sizes=((3, 3),) reproduces enumerate_pairings(n, 3).  ((3, 3), (3, 2), (2, 2))
    additionally allows a parent whose two decay quarks merged into one jet.
    """
    out = []
    for sa, sb in sizes:
        sa, sb = max(sa, sb), min(sa, sb)
        for a in combinations(range(n_jets), sa):
            rest = [j for j in range(n_jets) if j not in a]
            for b in combinations(rest, sb):
                if sa == sb and not a < b:
                    continue
                out.append((a, b))
    return out


def asinh(x):
    """asinh written with elementary ops (the TorchScript ONNX exporter has no Asinh)."""
    return torch.sign(x) * torch.log(x.abs() + torch.sqrt(x * x + 1.0))


def sinh(x):
    e = torch.exp(x)
    return 0.5 * (e - 1.0 / e)


def cosh(x):
    e = torch.exp(x)
    return 0.5 * (e + 1.0 / e)


def logaddexp(a, b):
    """Stable log(exp(a) + exp(b)) with elementary ops (ONNX-exportable)."""
    return torch.maximum(a, b) + torch.log1p(torch.exp(-(a - b).abs()))


def to_cartesian(pt, eta, phi, m):
    px, py, pz = pt * torch.cos(phi), pt * torch.sin(phi), pt * sinh(eta)
    e = torch.sqrt(px**2 + py**2 + pz**2 + m**2)
    return torch.stack([e, px, py, pz], dim=-1)


def minkowski_m2(p):
    return p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2


def canonicalize(pt, eta, phi, m, mask):
    """Rotate so jet 0 has phi=0, flip so jet 0 has eta>=0, scale by HT.

    Returns (pt/HT, eta, phi, m/HT, ht).  Exact symmetries of the problem, so the
    network does not need to learn them from augmentation.
    """
    ht = (pt * mask).sum(-1, keepdim=True).clamp(min=1e-6)
    phi = phi - phi[:, :1]
    phi = (phi + torch.pi) % (2 * torch.pi) - torch.pi
    sgn = torch.where(eta[:, :1] < 0, -1.0, 1.0)
    eta = eta * sgn
    return pt / ht, eta, phi, m / ht, ht.squeeze(-1)


class PairBiasEncoderLayer(nn.Module):
    """Pre-LN Transformer layer with an additive per-head pairwise attention bias."""

    def __init__(self, d, nhead, dff, dropout):
        super().__init__()
        self.nhead = nhead
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_pad):
        # bias: (B*H, N, N) float; key_pad folded into bias as -inf
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=bias, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class PairNet(nn.Module):
    def __init__(
        self,
        n_jets: int = 8,
        k: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        use_pair_bias: bool = True,
        use_group_kin: bool = True,
        use_null: bool = True,
        sizes=None,
        gated: bool = False,
        bg_beta: float = 1.0,
        bg_kappa: float = 30.0,
        n_assign: int | None = None,
        gate_interp: bool = False,
        mass_head: bool = False,
    ):
        super().__init__()
        self.n_jets, self.k = n_jets, k
        self.d, self.nhead = d_model, nhead
        self.use_pair_bias = use_pair_bias
        self.use_group_kin = use_group_kin
        self.use_null = use_null or gated
        self.gated = gated
        self.bg_beta, self.bg_kappa = bg_beta, bg_kappa
        self.gate_interp = gate_interp
        self.mass_head = mass_head
        use_null = self.use_null
        # interpretations use the leading n_assign jets; the encoder (context)
        # sees all n_jets
        self.n_assign_jets = n_assign or n_jets

        self.sizes = tuple(tuple(x) for x in (sizes or ((k, k),)))
        interp = enumerate_interpretations(self.n_assign_jets, self.sizes)
        self.n_assign = len(interp)
        M1 = torch.zeros(self.n_assign, n_jets)
        M2 = torch.zeros(self.n_assign, n_jets)
        kmax = max(max(x) for x in self.sizes)
        npair = kmax * (kmax - 1) // 2
        # intra-group jet pairs, padded with (0, 0) and masked
        P1 = torch.zeros(self.n_assign, npair, 2, dtype=torch.long)
        P2 = torch.zeros(self.n_assign, npair, 2, dtype=torch.long)
        PM1 = torch.zeros(self.n_assign, npair)
        PM2 = torch.zeros(self.n_assign, npair)
        for i, (a, b) in enumerate(interp):
            M1[i, list(a)] = 1.0
            M2[i, list(b)] = 1.0
            for P, PM, g in ((P1, PM1, a), (P2, PM2, b)):
                prs = list(combinations(g, 2))
                for j, (x, y) in enumerate(prs):
                    P[i, j, 0], P[i, j, 1] = x, y
                    PM[i, j] = 1.0
        for name, t in (("M1", M1), ("M2", M2), ("P1", P1), ("P2", P2), ("PM1", PM1), ("PM2", PM2)):
            self.register_buffer(name, t, persistent=False)

        n_jet_feat = 7
        self.embed = nn.Sequential(
            nn.Linear(n_jet_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        n_pair_feat = 4
        self.pair_mlp = nn.Sequential(
            nn.Linear(n_pair_feat, 64), nn.GELU(), nn.Linear(64, nhead * num_layers)
        )
        self.layers = nn.ModuleList(
            [PairBiasEncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)

        # group kinematics: ln m/HT, ln(pT/m), eta, k sorted pairwise-mass
        # fractions (k=3: 3 values; k=2: 1 value), ln(m_max_pair/m)
        # + group size flag
        self.n_gk = 4 + npair if use_group_kin else 0
        # inter-group: asym, ln((m1+m2)/HT), dR, |dphi|, |deta|, pT balance,
        # ln(m_sys/HT), rest pT fraction
        self.n_ik = 8 if use_group_kin else 0

        self.group_mlp = nn.Sequential(
            nn.Linear(d_model + self.n_gk, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.rest_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d_model + self.n_ik, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        if use_null:
            # the gate optionally also sees a softmax-pooled summary of the
            # interpretation representations plus the best and log-sum-exp
            # scores ("does a convincing pairing exist?")
            n_in = 2 * d_model + (d_model + 2 if gate_interp else 0)
            self.null_mlp = nn.Sequential(
                nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )
        if mass_head:
            # scale-free mass estimate: log(M/HT) as a learned correction to the
            # probability-weighted log m_avg/HT of the interpretations
            self.mass_mlp = nn.Sequential(
                nn.Linear(3 * d_model + 4, d_model), nn.GELU(),
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
            )

    # ------------------------------------------------------------------ features
    def _jet_features(self, pt, eta, phi, m, mask):
        lpt = torch.log(pt.clamp(min=1e-6))
        e = torch.sqrt((pt * cosh(eta)) ** 2 + m**2)
        le = torch.log(e.clamp(min=1e-6))
        f = torch.stack(
            [lpt, eta, torch.cos(phi), torch.sin(phi), le, m / pt.clamp(min=1e-6), lpt - le], -1
        )
        return f * mask.unsqueeze(-1)

    def _pair_features(self, p4, pt, eta, phi, mask):
        deta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
        dphi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
        dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
        dr = torch.sqrt(deta**2 + dphi**2).clamp(min=1e-4)
        ptmin = torch.minimum(pt.unsqueeze(-1), pt.unsqueeze(-2))
        kt = ptmin * dr
        z = ptmin / (pt.unsqueeze(-1) + pt.unsqueeze(-2)).clamp(min=1e-6)
        pij = p4.unsqueeze(2) + p4.unsqueeze(1)
        m2 = minkowski_m2(pij).clamp(min=1e-8)
        f = torch.stack([torch.log(dr), torch.log(kt.clamp(min=1e-6)), torch.log(z.clamp(min=1e-6)),
                         torch.log(m2)], -1)
        pm = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
        return f * pm, m2

    # ------------------------------------------------------------------ forward
    def forward(self, pt, eta, phi, m, mask):
        """pt/eta/phi/m: (B, N) pT-sorted, zero-padded; mask (B, N) bool.

        Returns dict with
          logits   (B, A+1)  last column is the null logit (if use_null)
          valid    (B, A)    interpretation uses only real jets
          m1, m2   (B, A)    group masses in HT units
          ht       (B,)
        """
        B, N = pt.shape
        pt, eta, phi, m, ht = canonicalize(pt, eta, phi, m, mask)
        p4 = to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1)

        x = self.embed(self._jet_features(pt, eta, phi, m, mask))
        pf, m2pair = self._pair_features(p4, pt, eta, phi, mask)
        neg = torch.zeros(B, 1, N, device=pt.device).masked_fill(~mask.unsqueeze(1), float("-inf"))
        if self.use_pair_bias:
            pb = self.pair_mlp(pf)  # (B, N, N, H*L)
            pb = pb.view(B, N, N, len(self.layers), self.nhead).permute(3, 0, 4, 1, 2)
        for li, layer in enumerate(self.layers):
            bias = neg.unsqueeze(1).expand(B, self.nhead, N, N)
            if self.use_pair_bias:
                bias = bias + pb[li]
            x = layer(x, bias.reshape(B * self.nhead, N, N), None)
        x = self.ln_out(x) * mask.unsqueeze(-1)

        # ---- per-interpretation quantities via membership matmuls
        M1, M2 = self.M1, self.M2
        Mr = 1.0 - M1 - M2
        e1 = torch.einsum("an,bnd->bad", M1, x)
        e2 = torch.einsum("an,bnd->bad", M2, x)
        er = torch.einsum("an,bnd->bad", Mr, x)
        P1 = torch.einsum("an,bnc->bac", M1, p4)
        P2 = torch.einsum("an,bnc->bac", M2, p4)
        Pr = torch.einsum("an,bnc->bac", Mr, p4)
        m1 = torch.sqrt(minkowski_m2(P1).clamp(min=1e-8))
        m2 = torch.sqrt(minkowski_m2(P2).clamp(min=1e-8))

        need = (M1 + M2).sum(-1)
        valid = torch.einsum("an,bn->ba", M1 + M2, mask.float()) > need - 0.5

        if self.use_group_kin:
            gk1 = self._gkin(P1, m1, m2pair, self.P1, self.PM1, self.M1)
            gk2 = self._gkin(P2, m2, m2pair, self.P2, self.PM2, self.M2)
            h1 = self.group_mlp(torch.cat([e1, gk1], -1))
            h2 = self.group_mlp(torch.cat([e2, gk2], -1))
            ik = self._ikin(P1, P2, Pr, m1, m2)
        else:
            h1, h2 = self.group_mlp(e1), self.group_mlp(e2)
            ik = e1.new_zeros(B, self.n_assign, 0)
        hr = self.rest_mlp(er)
        hid = self.score_mlp[:-1](torch.cat([h1 + h2, h1 * h2, hr, ik], -1))  # (B, A, d)
        s = self.score_mlp[-1](hid).squeeze(-1)
        s = s.masked_fill(~valid, -1e4)

        out = {"assign_logits": s, "valid": valid, "m1": m1, "m2": m2, "ht": ht}
        if self.use_null:
            pooled = torch.cat(
                [(x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)), x.max(1).values], -1
            )
            if self.gate_interp or self.mass_head:
                wts = s.softmax(-1).unsqueeze(-1)
                summ = (wts * hid).sum(1)
                nvalid = valid.float().sum(-1, keepdim=True).clamp(min=1)
                lse = torch.logsumexp(s, -1, keepdim=True) - torch.log(nvalid)
                smax = s.max(-1, keepdim=True).values
            if self.mass_head:
                lm = torch.log((0.5 * (m1 + m2)).clamp(min=1e-6))  # (B, A), HT units
                lm_mean = (wts.squeeze(-1) * lm).sum(-1, keepdim=True)
                lm_best = lm.gather(1, s.argmax(-1, keepdim=True))
                corr = self.mass_mlp(torch.cat([pooled, summ, smax, lse, lm_mean, lm_best], -1))
                out["log_m_over_ht"] = (lm_mean + corr).squeeze(-1)
            gate_in = torch.cat([pooled, summ, smax, lse], -1) if self.gate_interp else pooled
            null = self.null_mlp(gate_in)
            if self.gated:
                # gate: P(signal) = sigmoid(-null).  Mixture of the learned signal
                # scorer and a fixed anti-signal choice (high asymmetry, low mass).
                bgs = self.anti_signal_score(m1, m2, valid)
                out["bg_logits"] = (self.bg_kappa * bgs).masked_fill(~valid, -1e4)
                out["gate_logit"] = -null.squeeze(-1)
                ls = F.log_softmax(s, -1)
                lb = F.log_softmax(out["bg_logits"], -1)
                g = out["gate_logit"].unsqueeze(-1)
                out["logits"] = logaddexp(F.logsigmoid(g) + ls, F.logsigmoid(-g) + lb)
            else:
                out["logits"] = torch.cat([s, null], -1)
        else:
            out["logits"] = s
        return out

    def anti_signal_score(self, m1, m2, valid):
        """Per-interpretation 'obviously not signal' score in [0, 1+beta]:
        mass asymmetry + beta * (how low m1+m2 is within this event's range)."""
        asym = (m1 - m2).abs() / (m1 + m2).clamp(min=1e-6)
        msum = m1 + m2
        big = torch.where(valid, msum, torch.full_like(msum, -1e9)).max(-1, keepdim=True).values
        small = torch.where(valid, msum, torch.full_like(msum, 1e9)).min(-1, keepdim=True).values
        low = (big - msum) / (big - small).clamp(min=1e-6)
        return asym + self.bg_beta * low

    def _gkin(self, P, mg, m2pair, pidx, pmask, M):
        """Group kinematics from 4-vectors only."""
        pt = torch.sqrt(P[..., 1] ** 2 + P[..., 2] ** 2).clamp(min=1e-6)
        eta = asinh(P[..., 3] / pt)
        # intra-group pairwise masses as fractions of the group mass^2 (sorted, padded)
        fr = m2pair[:, pidx[..., 0], pidx[..., 1]] * pmask  # (B, A, npair)
        fr = fr / (mg**2).unsqueeze(-1).clamp(min=1e-8)
        fr = fr.sort(-1, descending=True).values
        size = M.sum(-1).expand(mg.shape) - 2.5
        return torch.cat(
            [torch.log(mg).unsqueeze(-1), torch.log(pt / mg).unsqueeze(-1), eta.unsqueeze(-1),
             size.unsqueeze(-1), fr], -1
        )

    @staticmethod
    def _ikin(P1, P2, Pr, m1, m2):
        def ptetaphi(P):
            pt = torch.sqrt(P[..., 1] ** 2 + P[..., 2] ** 2).clamp(min=1e-6)
            return pt, asinh(P[..., 3] / pt), torch.atan2(P[..., 2], P[..., 1])

        pt1, eta1, phi1 = ptetaphi(P1)
        pt2, eta2, phi2 = ptetaphi(P2)
        dphi = (phi1 - phi2 + torch.pi) % (2 * torch.pi) - torch.pi
        deta = eta1 - eta2
        dr = torch.sqrt(dphi**2 + deta**2)
        asym = (m1 - m2).abs() / (m1 + m2).clamp(min=1e-6)
        msys = torch.sqrt(minkowski_m2(P1 + P2).clamp(min=1e-8))
        ptr = torch.sqrt(Pr[..., 1] ** 2 + Pr[..., 2] ** 2)
        bal = (pt1 - pt2).abs() / (pt1 + pt2)
        return torch.stack(
            [asym, torch.log(m1 + m2), dr, dphi.abs(), deta.abs(), bal, torch.log(msys), ptr], -1
        )


def pairnet_loss(out, target, is_sig, sig_weight=1.0, bkg_weight=1.0):
    """Joint softmax over [interpretations..., null].

    target: (B,) interpretation index for reconstructable signal, -1 for signal
            whose true interpretation is not among the candidates, ignored for bkg.
    is_sig: (B,) bool.
    """
    logits = out["logits"]
    logp = F.log_softmax(logits, -1)
    A = out["assign_logits"].shape[1]
    has_null = logits.shape[1] == A + 1
    loss = logits.new_zeros(logits.shape[0])
    rec = is_sig & (target >= 0)
    if rec.any():
        loss[rec] = -logp[rec, target[rec]]
    if has_null:
        unrec = is_sig & (target < 0)
        if unrec.any():
            # signal, but no correct candidate: only demand "not null"
            loss[unrec] = -torch.logsumexp(logp[unrec, :A], -1)
        bkg = ~is_sig
        if bkg.any():
            loss[bkg] = -logp[bkg, A]
    w = torch.where(is_sig, sig_weight, bkg_weight)
    if not has_null:
        w = w * rec.float()
    return (loss * w).sum() / w.sum().clamp(min=1e-6)


def gated_loss(out, target, is_sig, w_proxy=0.5, w_gate=1.0, bkg_weight=1.0, pmass=None, w_mass=1.0):
    """Loss for the gated model.

    target: (B,) exact-truth index for reconstructable signal, or the
            "most signal-like" proxy index (encoded as -2 - idx) for signal whose
            true interpretation is unavailable; ignored for background.
    Signal  -> -log P_mix(target)      (exact: weight 1, proxy: w_proxy)
    QCD     -> -log P_mix(argmax anti-signal score)
    plus a BCE on the gate (signal vs QCD).
    """
    logp = out["logits"]
    B = logp.shape[0]
    exact = is_sig & (target >= 0)
    proxy = is_sig & (target <= -2)
    tgt = torch.where(proxy, -2 - target, target)
    bg_t = out["bg_logits"].argmax(-1)
    tgt = torch.where(is_sig, tgt, bg_t).clamp(min=0)
    nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    w = torch.where(exact, 1.0, torch.where(proxy, w_proxy, 0.0))
    w = torch.where(is_sig, w, torch.full_like(w, bkg_weight))
    lmix = (nll * w).sum() / w.sum().clamp(min=1e-6)
    lgate = F.binary_cross_entropy_with_logits(
        out["gate_logit"], is_sig.float(),
        weight=torch.where(is_sig, 1.0, bkg_weight) * torch.ones(B, device=logp.device),
    )
    loss = lmix + w_gate * lgate
    if "log_m_over_ht" in out and pmass is not None and is_sig.any():
        # L1 (median) regression of log(M/HT) on signal events
        tgt_m = torch.log(pmass.clamp(min=1.0) / out["ht"].clamp(min=1e-6))
        loss = loss + w_mass * (out["log_m_over_ht"] - tgt_m).abs()[is_sig].mean()
    return loss


def load_checkpoint(path):
    """Rebuild a PairNet from a train_pairnet.py checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    sizes = tuple((int(x[0]), int(x[1])) for x in a["sizes"].split(",")) if a.get("sizes") else None
    net = PairNet(
        n_jets=a["n_jets"], k=a["k"], d_model=a["d_model"], nhead=a["heads"],
        num_layers=a["layers"], dim_ff=2 * a["d_model"], dropout=0.0,
        use_pair_bias=not a["no_pair_bias"], use_group_kin=not a["no_group_kin"],
        use_null=not a["no_null"], sizes=sizes, gated=a.get("gated", False),
        bg_beta=a.get("bg_beta", 1.0), n_assign=a.get("n_assign"),
        gate_interp=a.get("gate_interp", False), mass_head=a.get("mass_head", False),
    )
    net.load_state_dict(ck["state"])
    net.eval()
    return net, a
