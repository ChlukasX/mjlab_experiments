"""Tables from the JSON/npz files of scripts/eval_selfcontact.py (docs/mount_selfcontact).

    uv run python scripts/gen_selfcontact_report.py <dir> <label> [<label> ...]

Per policy: self-contact, coincidences, close calls, arms. Then the comparison, the share of steps
with any self-contact, and the distance between the right leg and the left leg while latched.
"""

import json
import sys

import numpy as np

D, LABELS = sys.argv[1], sys.argv[2:]
TH = (0.0, 0.005, 0.01, 0.02, 0.05)
ARMS = ("shoulder_pitch", "shoulder_roll", "upper_arm", "elbow")
R = {lab: (json.load(open(f"{D}/{lab}.json")), np.load(f"{D}/{lab}_dist.npz")) for lab in LABELS}


def pct(x):
    return "-" if x != x else (f"{100 * x:.1f}%" if x >= 0.001 or x == 0 else f"{100 * x:.2f}%")


def contact_stats(r):
    a = {k: np.array(v) for k, v in r["acc"].items()}
    n = np.array(r["steps_per_phase"])
    c3 = a["cnt"][:3].sum(0)
    out = {"share": c3 / n[:3].sum(), "n3": n[:3].sum(), "n": n}
    out["ph"] = a["cnt"] / np.maximum(n[:, None], 1)
    out["dmean"] = 1000 * a["dsum"][:3].sum(0) / np.maximum(c3, 1)
    out["dmax"] = 1000 * a["dmax"][:3].max(0)
    out["fmean"] = a["fsum"][:3].sum(0) / np.maximum(c3, 1)
    out["fmax"] = a["fmax"][:3].max(0)
    for k in ("f10", "fw"):
        out[k] = a[k][:3].sum(0) / np.maximum(c3, 1)
    out["a"] = a
    return out


def close_stats(z, key):
    d, ph = z[key], z["mph" if key == "mesh" else "ph"]
    m3 = np.isin(ph, (0, 1, 2))
    return {
        "all": np.stack([(d[m3] < t).mean(0) for t in TH], 0),
        "A": (d[ph == 0] < 0.01).mean(0),
        "B": (d[ph == 1] < 0.01).mean(0),
        "C": (d[ph == 2] < 0.01).mean(0),
        "min": d[m3].min(0),
        "n": int(m3.sum()),
        "nA": int((ph == 0).sum()),
        "static": (d[m3].std(0) < 0.003) & (d[m3].mean(0) < 0.04),
    }  # nests at a joint: the distance hardly moves


for lab in LABELS:
    r, z = R[lab]
    pairs, cs = r["pairs"], contact_stats(r)
    n = cs["n"]
    print(f"\n## {lab}  ({r['task'].removeprefix('Mjlab-Piplus-Ball-')})")
    print(
        f"envs {r['num_envs']}, seed {r['seed']}; env-steps behind each share (A / B / C / M / E): "
        + " / ".join(f"{int(x)}" for x in n)
        + f"; t>=3 s (A+B+C): {int(cs['n3'])}; "
        f"primitive distance samples: {len(z['ph'])} (t>=3 s: {int(np.isin(z['ph'], (0, 1, 2)).sum())}, in A: {int((z['ph'] == 0).sum())}); "
        f"mesh samples: {len(z['mph'])} (t>=3 s: {int(np.isin(z['mph'], (0, 1, 2)).sum())}, in A: {int((z['mph'] == 0).sum())})"
    )
    g, sat = r["gait"], np.array(r["sat_ph"])
    hip = r["actuators"].index("r_hip_pitch")
    nab = n[0] + n[1]
    print(
        f"survived {r['survived']:.0%}, held-stance success {r['success']:.0%}, taps {g['taps_per_10s']:.1f} per 10 s, "
        f"stance share of on-ball steps {g['stance_share']:.0%}, support foot - torso {g['yaw_deg_median']:+.0f} deg, "
        f"r_hip_pitch at force limit (A+B) {(sat[0, hip] * n[0] + sat[1, hip] * n[1]) / nab:.0%}; "
        f"force check: external contact force / weight = {r['ext_force_over_weight']:.2f} (weight {r['weight_N']:.0f} N)"
    )
    # 1. contacts
    top = np.argsort(-cs["share"])[:10]
    print(
        "\nSelf-contact, top 10 body pairs (share = env-steps with contact; depth and force over contact env-steps, t>=3 s)\n"
    )
    print(
        "| pair | share t>=3 | A | B | C | M (mount) | depth mean / max mm | normal force mean / max N | F>10 N | F>weight |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in top:
        ph = cs["ph"][:, p]
        print(
            f"| {pairs[p]} | {pct(cs['share'][p])} | {pct(ph[0])} | {pct(ph[1])} | {pct(ph[2])} | {pct(ph[3])} | "
            f"{cs['dmean'][p]:.1f} / {cs['dmax'][p]:.1f} | {cs['fmean'][p]:.0f} / {cs['fmax'][p]:.0f} | "
            f"{pct(cs['f10'][p])} | {pct(cs['fw'][p])} |"
        )
    anyc = np.array(r["acc"]["cnt"])
    # 2. coincidences, phases A+B (latched)
    a, bph = cs["a"], np.array(r["base_ph"])
    print(
        "\nCoincidence in A+B (share of the pair's contact steps with ...; baseline = share of all A+B steps)\n"
    )
    print(
        "| pair | free-foot tap | tap within 0.1 s | r_hip_pitch at force limit | any actuator at limit | r_hip_pitch joint at range limit |"
    )
    print("|---|---|---|---|---|---|")

    def ab(k, p):
        return a[k][:2, p].sum() / max(a["cnt"][:2, p].sum(), 1)

    base = bph[:2].sum(0) / nab
    print(
        f"| baseline (all A+B steps) | {pct(base[0])} | {pct(base[1])} | {pct(base[2])} | {pct(base[3])} | {pct(base[4])} |"
    )
    for p in top[:5]:
        print(
            f"| {pairs[p]} | {pct(ab('tap', p))} | {pct(ab('tapn', p))} | {pct(ab('hipsat', p))} | {pct(ab('anysat', p))} | {pct(ab('hiplim', p))} |"
        )
    f = r["falls"]
    print(
        f"\nFalls: {f['n']} terminated envs ({', '.join(f'{k} {v}' for k, v in f['terms'].items() if v)}); self-contact in the last 0.5 s "
        f"before the fall: any {f['any']}/{f['n']}, with F>10 N {f['sig']}/{f['n']}; baseline (all live t>=3 s steps): any "
        f"{pct(r['base']['any'] / max(r['base']['n'], 1))}, F>10 N {pct(r['base']['sig'] / max(r['base']['n'], 1))}"
    )
    fp, lp = np.array(r["fall_p"]), np.array(r["live_p"])
    print(
        "Contact in the last 0.5 s before a fall, against the same window at a live step: "
        + ", ".join(
            f"{pairs[p]} {int(fp[p])}/{f['n']} ({pct(fp[p] / max(f['n'], 1))}) vs {pct(lp[p] / max(r['base']['n'], 1))}"
            for p in top[:5]
        )
    )
    c = r["corr"]
    print(
        f"Across envs (n {c['n_envs']}), Spearman of taps per 10 s with the share of steps with self-contact: any "
        f"{c['taps_vs_any_contact']:+.2f}, F>10 N {c['taps_vs_f10_contact']:+.2f}; by pair (F>10 N): "
        + ", ".join(f"{k} {v:+.2f}" for k, v in c["by_pair_f10"].items())
    )
    # 3. close calls
    for key, title in (
        ("coll", "collision primitives (what the simulation enforces)"),
        ("mesh", "visual meshes, convex hulls (overestimate)"),
    ):
        cl = close_stats(z, key)
        order = [p for p in np.argsort(-cl["all"][2]) if not cl["static"][p]][:10]
        stat = [pairs[p] for p in np.flatnonzero(cl["static"] & (cl["all"][2] > 0.5))]
        print(
            f"\nClose calls, {title}: share of samples (t>=3 s, n {cl['n']}) with signed distance below ...\n"
        )
        print(
            "| pair | min distance mm | <0 | <5 mm | <1 cm | <2 cm | <5 cm | <1 cm in A | in B | in C |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for p in order:
            s = cl["all"][:, p]
            print(
                f"| {pairs[p]} | {1000 * cl['min'][p]:.1f} | {pct(s[0])} | {pct(s[1])} | {pct(s[2])} | {pct(s[3])} | {pct(s[4])} | "
                f"{pct(cl['A'][p])} | {pct(cl['B'][p])} | {pct(cl['C'][p])} |"
            )
        print(
            f"\nStatic pairs left out (distance std < 3 mm, mean < 4 cm; nest at a joint): {', '.join(stat) or 'none'}"
        )
    # mechanism of the top two pairs (primitive samples, phases A+B)
    pr, jq, af = z["pres"], z["jq"], z["af"]
    jn, an = list(z["joints"]), list(z["actuators"])
    ab_m = np.isin(z["ph"], (0, 1))
    leg = [
        f"{sd}_{n}"
        for sd in ("r", "l")
        for n in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")
    ]
    print(
        "\nMechanism, primitive samples in A+B (n %d): mean joint angle (rad) with the pair in contact / not, and the r_hip_pitch actuator force\n"
        % ab_m.sum()
    )
    print(
        "| pair | state | n | "
        + " | ".join(leg)
        + " | P(contact) in A | in B with tap | in B without tap | r_hip_pitch force at limit | corr(F, |r_hip_pitch force|) |"
    )
    print("|---|---|---|" + "---|" * (len(leg) + 5))
    for p in top[:2]:
        cm = ab_m & pr[:, p]
        for lab, mk in (("contact", cm), ("no contact", ab_m & ~pr[:, p])):
            qs = " | ".join(f"{jq[mk][:, jn.index(n + '_joint')].mean():+.2f}" for n in leg)
            if lab == "contact":
                hf = np.abs(af[cm][:, an.index("r_hip_pitch")])
                pf = z["pf"][cm][:, p].astype(float)
                extra = (
                    f"{pct(pr[z['ph'] == 0][:, p].mean())} | {pct(pr[(z['ph'] == 1) & z['tap']][:, p].mean())} | "
                    f"{pct(pr[(z['ph'] == 1) & ~z['tap']][:, p].mean())} | {pct((hf > 19).mean())} | "
                    f"{np.corrcoef(pf, hf)[0, 1]:+.2f}"
                )
            else:
                extra = " | | | | "
            print(f"| {pairs[p]} | {lab} | {int(mk.sum())} | {qs} | {extra} |")
    # 4. arms
    j = r["joints"]
    print(
        f"\nArm joints in phase A (n {int(j['n'])} env-steps): mean q / std / mean target (rad), share within 0.03 rad of the range limit\n"
    )
    print("| joint | q mean | q std | target mean | at limit | range |")
    print("|---|---|---|---|---|---|")
    for i, nme in enumerate(j["names"]):
        if any(x in nme for x in ARMS):
            print(
                f"| {nme.removesuffix('_joint')} | {j['mean'][i]:+.2f} | {j['std'][i]:.2f} | {j['target'][i]:+.2f} | {pct(j['at_limit'][i])} | "
                f"{j['range'][i][0]:+.2f} .. {j['range'][i][1]:+.2f} |"
            )

# comparison across policies
if len(LABELS) > 1:
    cs_all = {lab: contact_stats(R[lab][0]) for lab in LABELS}
    pairs = R[LABELS[0]][0]["pairs"]
    tot = sum(cs_all[lab]["share"] for lab in LABELS)
    top = np.argsort(-tot)[:6]
    cl = {k: {lab: close_stats(R[lab][1], k) for lab in LABELS} for k in ("coll", "mesh")}
    head = "| pair | metric | " + " | ".join(LABELS) + " |\n|---|---|" + "---|" * len(LABELS)
    print("\n## Comparison\n")
    print(head)
    for p in top:
        rows = (
            ("self-contact share t>=3", lambda lab: pct(cs_all[lab]["share"][p])),
            (
                "mean / max force N",
                lambda lab: f"{cs_all[lab]['fmean'][p]:.0f} / {cs_all[lab]['fmax'][p]:.0f}",
            ),
            (
                "mean / max depth mm",
                lambda lab: f"{cs_all[lab]['dmean'][p]:.1f} / {cs_all[lab]['dmax'][p]:.1f}",
            ),
            (
                "share in A | B",
                lambda lab: f"{pct(cs_all[lab]['ph'][0, p])} | {pct(cs_all[lab]['ph'][1, p])}",
            ),
            ("primitive < 1 cm", lambda lab: pct(cl["coll"][lab]["all"][2, p])),
            ("mesh < 0", lambda lab: pct(cl["mesh"][lab]["all"][0, p])),
        )
        for name, fn in rows:
            print(f"| {pairs[p]} | {name} | " + " | ".join(fn(lab) for lab in LABELS) + " |")
    print("\nGait (same rollouts)\n")
    print("| metric | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS))
    for name, fn in (
        ("survived", lambda r: f"{r['survived']:.0%}"),
        ("held-stance success", lambda r: f"{r['success']:.0%}"),
        ("taps per 10 s", lambda r: f"{r['gait']['taps_per_10s']:.1f}"),
        ("stance share", lambda r: f"{r['gait']['stance_share']:.0%}"),
        ("support foot - torso deg", lambda r: f"{r['gait']['yaw_deg_median']:+.0f}"),
        (
            "r_hip_pitch at force limit (A+B)",
            lambda r: (
                f"{(np.array(r['sat_ph'])[:2, r['actuators'].index('r_hip_pitch')] * np.array(r['steps_per_phase'])[:2]).sum() / sum(r['steps_per_phase'][:2]):.0%}"
            ),
        ),
    ):
        print(f"| {name} | " + " | ".join(fn(R[lab][0]) for lab in LABELS) + " |")


# Share of sampled env-steps (t>=3 s) with any self-contact, by group.
print("\n## Any self-contact (primitive samples, t>=3 s)\n")
print(
    "| policy | samples | any contact | any, F>10 N | any, F>weight | without torso\\|r_thigh, F>10 N | feet and calves, F>10 N | arms and head, F>10 N | in A | in B |"
)
print("|---|---|---|---|---|---|---|---|---|---|")
for lab in LABELS:
    z = R[lab][1]
    pairs, pr, pf, ph = list(z["pairs"]), z["pres"], z["pf"].astype(float), z["ph"]
    weight = R[lab][0]["weight_N"]
    m, big = np.isin(ph, (0, 1, 2)), pr & (z["pf"].astype(float) > 10)
    th = pairs.index("torso|r_thigh")
    arm = [
        i
        for i, p in enumerate(pairs)
        if any(k in p for k in ("wrist", "elbow", "head", "upper_arm", "shoulder"))
    ]
    foot = [i for i, p in enumerate(pairs) if ("ankle" in p or "calf" in p) and i not in arm]
    ex = np.ones(len(pairs), bool)
    ex[th] = False

    def f(mask, a):
        return f"{a[mask].mean():.1%}"

    print(
        f"| {lab} | {int(m.sum())} | {f(m, pr.any(1))} | {f(m, big.any(1))} | {f(m, (pr & (pf > weight)).any(1))} | "
        f"{f(m, big[:, ex].any(1))} | {f(m, big[:, foot].any(1))} | {f(m, big[:, arm].any(1))} | "
        f"{f(ph == 0, big.any(1))} | {f(ph == 1, big.any(1))} |"
    )

# Right leg against left leg while latched: the closest pair of the group in each sample.
LEG = {
    "foot-foot": ("ankle_roll",),
    "any right-left leg pair (calf, ankle)": ("calf", "ankle_pitch", "ankle_roll"),
}
print("\n## Right leg against left leg (collision primitives; distance mm, negative = overlap)\n")
print("| policy | group | phase | n | median | p5 | <0 | <5 mm | <1 cm | <2 cm |")
print("|---|---|---|---|---|---|---|---|---|---|")
for lab in LABELS:
    z = R[lab][1]
    pairs = list(z["pairs"])
    for grp, names in LEG.items():
        idx = [
            i
            for i, p in enumerate(pairs)
            if p.split("|")[0].startswith("r_")
            and p.split("|")[1].startswith("l_")
            and p.split("|")[0][2:] in names
            and p.split("|")[1][2:] in names
        ]
        if grp == "foot-foot":
            idx = [i for i in idx if pairs[i] == "r_ankle_roll|l_ankle_roll"]
        for pname, phs in (("A", (0,)), ("B", (1,))):
            x = z["coll"][np.isin(z["ph"], phs)][:, idx].min(1) * 1000
            print(
                f"| {lab} | {grp} | {pname} | {len(x)} | {np.median(x):.1f} | {np.percentile(x, 5):.1f} | {(x < 0).mean():.0%} | "
                f"{(x < 5).mean():.0%} | {(x < 10).mean():.0%} | {(x < 20).mean():.0%} |"
            )
