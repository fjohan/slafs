#!/usr/bin/env python3
import argparse, math, random
from collections import defaultdict
import pandas as pd
import numpy as np

def largest_remainder(targets_float):
    floors = np.floor(targets_float).astype(int)
    remainder = targets_float - floors
    k = int(round(targets_float.sum())) - floors.sum()
    order = np.argsort(-remainder)  # biggest remainders first
    alloc = floors.copy()
    for i in order[:max(k, 0)]:
        alloc[i] += 1
    return alloc

def reallocate_shortages(desired, capacities):
    """Cap by capacity; redistribute shortages to bins with spare capacity."""
    desired = desired.astype(int).copy()
    capacities = capacities.astype(int)
    desired = np.minimum(desired, capacities)
    deficit = int(desired.sum())  # current placed
    target = int(capacities.sum())  # not the right target here, will be passed outside
    return desired  # (redistribution handled in the wrapper)

def stratified_take(df_class, bins, per_bin, rng):
    """Sample per_bin[i] from df_class items in bins[i]. If some bins lack items, redistribute."""
    # capacity per bin for this class
    cap = df_class.groupby(bins).size().reindex(range(len(per_bin)), fill_value=0).values
    need = per_bin.copy()

    # Cap by capacity
    need = np.minimum(need, cap)
    # Redistribute remaining count to fill total N
    short = per_bin.sum() - need.sum()
    if short > 0:
        spare = cap - need
        while short > 0 and spare.sum() > 0:
            # distribute one by one to bins with most spare
            idx = np.argsort(-spare)
            allocated_any = False
            for i in idx:
                if spare[i] > 0:
                    need[i] += 1
                    spare[i] -= 1
                    short -= 1
                    allocated_any = True
                    if short == 0:
                        break
            if not allocated_any:
                break  # no place to put more
    # Now actually sample
    picks = []
    for i, take in enumerate(need):
        if take <= 0: 
            continue
        pool = df_class[bins == i]
        if len(pool) <= take:
            picks.append(pool)  # take all
        else:
            picks.append(pool.sample(n=int(take), random_state=rng))
    if picks:
        return pd.concat(picks, ignore_index=True)
    return df_class.iloc[0:0]

def main():
    ap = argparse.ArgumentParser(description="Frequency-stratified sampling for animate vs inanimate.")
    ap.add_argument("--tsv", required=True, help="Input TSV with columns: writtenForm, lemgram, frequency, animacy, path")
    ap.add_argument("--n", type=int, required=True, help="Number of items PER CLASS (animate/inanimate)")
    ap.add_argument("--bins", type=int, default=10, help="Number of quantile bins on log10(frequency); default 10")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--out-prefix", default="sampled", help="Output prefix (default: sampled)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    df = pd.read_csv(args.tsv, sep="\t", dtype={"writtenForm":str, "lemgram":str,
                                                "frequency":np.int64, "animacy":str, "path":str})
    # Keep only animate / inanimate
    df = df[df["animacy"].isin(["animate", "inanimate"])].copy()
    df = df[df["frequency"] > 0].copy()

    # Log-frequency (guards against huge skew)
    df["_logf"] = np.log10(df["frequency"])

    # Shared quantile bins over ALL rows (animate + inanimate)
    # Labels 0..B-1 (low -> high)
    B = args.bins
    df["_bin"] = pd.qcut(df["_logf"], q=B, labels=False, duplicates="drop")
    # Ensure bins are 0..(B'-1) after duplicates=drop
    # If small dataset collapses to fewer bins
    unique_bins = sorted(df["_bin"].dropna().unique())
    remap = {b:i for i,b in enumerate(unique_bins)}
    df["_bin"] = df["_bin"].map(remap)
    B_eff = len(unique_bins)

    # Target per-bin proportions from the combined distribution
    overall_counts = df["_bin"].value_counts().reindex(range(B_eff), fill_value=0).values.astype(float)
    overall_props  = overall_counts / overall_counts.sum()
    desired_per_bin = largest_remainder(overall_props * args.n)  # length B_eff

    # Split classes
    anim_df = df[df["animacy"] == "animate"].copy()
    inan_df = df[df["animacy"] == "inanimate"].copy()

    # Bin assignments per class
    anim_bins = anim_df["_bin"].values
    inan_bins = inan_df["_bin"].values

    # For each class, sample according to desired_per_bin with capacity-aware redistribution
    def sample_for_class(df_class, class_bins):
        # capacities available per bin
        cap = pd.Series(class_bins).value_counts().reindex(range(B_eff), fill_value=0).values
        need = desired_per_bin.copy()
        need = np.minimum(need, cap)
        short = args.n - need.sum()
        if short > 0:
            spare = cap - need
            # greedily allocate remaining to bins with most spare
            order = np.argsort(-spare)
            for i in order:
                if short <= 0: break
                add = min(short, spare[i])
                if add > 0:
                    need[i] += add
                    spare[i] -= add
                    short -= add
        # if still short (class too small), just take as many as exist
        take_total = min(args.n, int(cap.sum()))
        adjust = need.sum() - take_total
        if adjust > 0:
            # trim from bins with the most assigned, one by one
            order = np.argsort(-need)
            for i in order:
                if adjust <= 0: break
                if need[i] > 0:
                    need[i] -= 1
                    adjust -= 1
        # build bin Series aligned with df_class
        return stratified_take(df_class, df_class["_bin"].values, need, rng)

    anim_sample = sample_for_class(anim_df, anim_bins)
    inan_sample = sample_for_class(inan_df, inan_bins)

    # Final tidy columns and write
    cols = ["writtenForm", "lemgram", "frequency", "animacy", "path"]
    anim_sample[cols].to_csv(f"{args.out_prefix}_animate.tsv", sep="\t", index=False)
    inan_sample[cols].to_csv(f"{args.out_prefix}_inanimate.tsv", sep="\t", index=False)

    # Quick summaries
    def summarize(name, df_s):
        s = df_s.copy()
        s["_bin"] = s["_bin"].astype(int)
        bybin = s.groupby("_bin")["lemgram"].count()
        print(f"\n{name}: n={len(s)}")
        print("per-bin counts:\n" + bybin.to_string())
        print("freq stats:", s["frequency"].describe(percentiles=[.1,.25,.5,.75,.9]).to_dict())

    summarize("ANIMATE", anim_sample)
    summarize("INANIMATE", inan_sample)

if __name__ == "__main__":
    main()


