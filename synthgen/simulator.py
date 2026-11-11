"""Vectorised transaction simulator (LLD 2.2): recurring + Poisson discretionary + episodic flows."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .catalog import BY_CATEGORY, CATALOG, FILL, Category as C
from .config import GenConfig

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_MERCH_IDX = {m.name: i for i, m in enumerate(CATALOG)}


def _sig(x):
    return 1 / (1 + np.exp(-x))


class _Sink:
    """Collects (user, day, merchant, amount) arrays from every flow."""

    def __init__(self):
        self.u, self.d, self.m, self.a = [], [], [], []

    def add(self, u, d, m, a):
        u = np.asarray(u, np.int64)
        if u.size == 0:
            return
        self.u.append(u); self.d.append(np.asarray(d, np.int64))
        self.m.append(np.broadcast_to(np.asarray(m, np.int64), u.shape).copy()); self.a.append(np.asarray(a, float))

    def frame(self):
        return pd.DataFrame({"user_id": np.concatenate(self.u), "day": np.concatenate(self.d),
                             "merchant": np.concatenate(self.m), "amount": np.concatenate(self.a)})


def simulate(p: pd.DataFrame, ev: pd.DataFrame, score_dates: np.ndarray, cfg: GenConfig,
             rng: np.random.Generator) -> pd.DataFrame:
    n, H = len(p), cfg.history_days
    days_before = H - np.arange(H)                                         # (H,) H..1
    dates = score_dates[:, None] - days_before[None, :].astype("timedelta64[D]")   # (n,H)
    dom = (dates - dates.astype("datetime64[M]")).astype(int) + 1          # day-of-month 1..31
    inc_ratio = (p.income_monthly / cfg.median_monthly_income).to_numpy()
    zi, zs, zt, zst = (p[c].to_numpy() for c in ("z_inc", "z_saver", "z_travel", "z_stress"))
    zd, zdisc = p.z_digital.to_numpy(), p.z_disc.to_numpy()
    e_type, e_day = ev.event.to_numpy(), ev.event_day.to_numpy()
    after = lambda t: (e_type == t)[:, None] & (days_before[None, :] <= e_day[:, None])   # (n,H)
    S = _Sink()

    def amount(m_idx, u, scale=None):
        m = CATALOG[m_idx] if np.isscalar(m_idx) else None
        mu = np.array([CATALOG[k].mu for k in np.atleast_1d(m_idx)]) if m is None else m.mu
        sd = np.array([CATALOG[k].sigma for k in np.atleast_1d(m_idx)]) if m is None else m.sigma
        a = np.exp(mu + 0.45 * np.log(inc_ratio[u]) + sd * rng.standard_normal(len(u)))
        return a if scale is None else a * scale

    def pick(cat, u):
        idx = np.array(BY_CATEGORY[cat])
        tiers = np.array([CATALOG[k].tier for k in idx])
        w = np.exp(0.9 * np.outer(zi[u], tiers))
        w /= w.sum(1, keepdims=True)
        r = rng.random(len(u))[:, None]
        return idx[np.minimum((r > w.cumsum(1)).sum(1), len(idx) - 1)]

    def poisson(cat, lam, mult=None, sign=-1.0):
        L = np.repeat(lam[:, None], H, 1) if mult is None else lam[:, None] * mult
        k = rng.poisson(L)
        u, d = np.nonzero(k)
        rep = k[u, d]
        u, d = np.repeat(u, rep), np.repeat(d, rep)
        m = pick(cat, u)
        S.add(u, d, m, sign * amount(m, u))

    # ---------------- recurring
    lost_job = (e_type == "STRESS") & (rng.random(n) < 0.35)
    raise_f = np.where(e_type == "RAISE", rng.uniform(1.15, 1.4, n), 1.0)
    sd = np.minimum(p.salary_day.to_numpy(), 28)
    u, d = np.nonzero(dom == sd[:, None])
    keep = ~(lost_job[u] & (days_before[d] <= e_day[u]))
    u, d = u[keep], d[keep]
    bump = np.where((e_type[u] == "RAISE") & (days_before[d] <= e_day[u]), raise_f[u], 1.0)
    S.add(u, d, _MERCH_IDX["SALARY"], p.income_monthly.to_numpy()[u] * bump * rng.normal(1, 0.02, len(u)))

    rent_day = rng.integers(5, 11, n)
    u, d = np.nonzero((dom == rent_day[:, None]) & p.pays_rent.to_numpy()[:, None])
    S.add(u, d, _MERCH_IDX["ALUGUEL"], -np.exp(7.3 + 0.6 * np.log(inc_ratio[u])) * rng.normal(1, 0.01, len(u)))

    tel = np.where(rng.random(n) < 0.5, _MERCH_IDX["VIVO"], _MERCH_IDX["CLARO"])
    for name, day in (("ENEL", 12), ("SABESP", 18), (None, 20)):
        u, d = np.nonzero(dom == day)
        m = tel[u] if name is None else _MERCH_IDX[name]
        S.add(u, d, m, -amount(m, u))

    subs = np.array(BY_CATEGORY[C.SUBSCRIPTION])
    sub_pop = np.array([5, 4, 3, 2, 1, 2], float); sub_pop /= sub_pop.sum()
    for uu in range(n):
        k = int(p.n_subs.iat[uu])
        if not k:
            continue
        chosen = rng.choice(subs, size=min(k, len(subs)), replace=False, p=sub_pop)
        for m in chosen:
            dd = np.nonzero(dom[uu] == rng.integers(1, 29))[0]
            S.add(np.full(len(dd), uu), dd, m, -np.full(len(dd), np.exp(CATALOG[m].mu)))

    # investing: monthly transfers for savers; RAISE events push savers into investing
    invests = rng.random(n) < _sig(1.6 * zs + 0.5 * zi - 0.6)
    inv_day = np.minimum(sd + 2, 28)
    u, d = np.nonzero(dom == inv_day[:, None])
    active = invests[u] | ((e_type[u] == "RAISE") & (days_before[d] <= e_day[u]) & (zs[u] > -0.7))
    active &= ~(lost_job[u] & (days_before[d] <= e_day[u]))
    u, d = u[active], d[active]
    S.add(u, d, _MERCH_IDX["BROKER"], -p.income_monthly.to_numpy()[u] * rng.uniform(0.08, 0.3, len(u)))
    u, d = np.nonzero((dom == 1) & invests[:, None])
    S.add(u, d, _MERCH_IDX["RENDIMENTO"], amount(_MERCH_IDX["RENDIMENTO"], u) * (1 + np.exp(zs[u])))

    # ---------------- discretionary (Poisson)
    kids = p.has_kids.to_numpy(); car = p.has_car.to_numpy()
    baby_after, car_after, stress_after = after("BABY"), after("NEW_CAR"), after("STRESS")
    poisson(C.GROCERY, 0.18 * (1 + 0.4 * kids))
    poisson(C.RESTAURANT, 0.20 * np.exp(0.35 * zdisc + 0.2 * zi), mult=1 - 0.4 * stress_after)
    poisson(C.DELIVERY, 0.10 * np.exp(0.5 * zd + 0.3 * zdisc), mult=1 + 0.8 * baby_after)
    poisson(C.TRANSPORT, 0.20 * np.exp(0.3 * zd) * np.where(car, 0.5, 1.0))
    poisson(C.FUEL, 0.08 * car)
    poisson(C.FUEL, np.full(n, 0.09), mult=car_after & ~car[:, None])
    poisson(C.SHOPPING, 0.08 * np.exp(0.5 * zdisc + 0.2 * zi), mult=1 + 1.5 * baby_after - 0.5 * stress_after)
    poisson(C.PHARMACY, 0.04 * np.exp(0.02 * (p.age.to_numpy() - 38)), mult=1 + 2.5 * baby_after)
    poisson(C.BABY, np.full(n, 0.10), mult=baby_after)
    poisson(C.AUTO, np.full(n, 0.03), mult=car_after * 1.0 + 0.15 * car[:, None])
    poisson(C.PIX_OUT, 0.12 * np.exp(0.3 * zdisc))
    poisson(C.PIX_IN, 0.06 * np.ones(n), mult=1 + 2.0 * stress_after, sign=+1.0)
    poisson(C.CASH, 0.03 * np.exp(-0.5 * zd))
    # financial stress: fees, quick loans, collections — intensify after a STRESS event
    poisson(C.FEE, 0.010 * np.exp(np.minimum(1.1 * zst, 2.5)), mult=1 + 5 * stress_after)
    poisson(C.LOAN_IN, 0.0015 * np.exp(1.0 * zst), mult=1 + 10 * stress_after, sign=+1.0)
    poisson(C.COLLECTION, 0.001 * np.exp(1.2 * zst), mult=1 + 6 * stress_after)

    # car purchase on the event day
    u = np.nonzero(e_type == "NEW_CAR")[0]
    S.add(u, H - e_day[u], _MERCH_IDX["CONCESSIONARIA"], -amount(_MERCH_IDX["CONCESSIONARIA"], u))

    # ---------------- episodic travel
    n_trips = rng.poisson(np.minimum(H / 365 * 1.5 * np.exp(1.1 * zt + 0.3 * zi - 0.4), 5.0) * np.where(stress_after.any(1), 0.5, 1))
    tu = np.repeat(np.arange(n), n_trips)
    start = rng.integers(0, H, len(tu))
    intl = rng.random(len(tu)) < _sig(0.8 * zi[tu] + 0.6 * zt[tu] - 1.0)
    book = np.clip(start - rng.integers(7, 40, len(tu)), 0, H - 1)
    m = pick(C.TRAVEL_AIR, tu); S.add(tu, book, m, -amount(m, tu) * np.where(intl, 2.5, 1.0))
    m = pick(C.TRAVEL_HOTEL, tu); S.add(tu, np.clip(book + rng.integers(0, 5, len(tu)), 0, H - 1), m, -amount(m, tu))
    k = np.where(intl, rng.integers(3, 9, len(tu)), 0)
    if k.sum():
        au = np.repeat(tu, k)
        ad = np.clip(np.repeat(start, k) + np.concatenate([np.arange(x) for x in k]), 0, H - 1)
        m = np.full(len(au), _MERCH_IDX["ABROAD"]); S.add(au, ad, m, -amount(m, au))

    tx = S.frame()
    tx = tx[(tx.day >= 0) & (tx.day < H)]
    tx["intra"] = rng.random(len(tx))
    tx = tx.sort_values(["user_id", "day", "intra"], kind="stable").drop(columns="intra").reset_index(drop=True)
    tx["date"] = score_dates[tx.user_id.to_numpy()] - (H - tx.day.to_numpy()).astype("timedelta64[D]")
    tx["amount"] = tx.amount.round(2)
    tx["category"] = [CATALOG[k].category.value for k in tx.merchant.to_numpy()]
    tx["description"] = _describe(tx.merchant.to_numpy(), cfg, rng)
    return tx[["user_id", "date", "amount", "description", "category"]]


def _describe(merchants: np.ndarray, cfg: GenConfig, rng: np.random.Generator) -> list:
    pools = {k: np.array(v) for k, v in FILL.items()}
    pre = {k: rng.choice(v, len(merchants)) for k, v in pools.items()}
    nums = rng.integers(1, 9999, len(merchants))
    tsel = rng.random(len(merchants))
    trunc = rng.random(len(merchants)) < cfg.desc_truncate_p
    tlen = rng.integers(8, 19, len(merchants))
    out = []
    for i, mi in enumerate(merchants):
        tpls = CATALOG[mi].templates
        t = tpls[int(tsel[i] * len(tpls))]
        s = _PLACEHOLDER.sub(lambda g: f"{nums[i]:04d}" if g.group(1) == "num" else str(pre[g.group(1)][i]), t)
        out.append(s[: tlen[i]] if trunc[i] else s)
    return out
