import json
import numpy as np
import argparse
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import csv

SIERPINSKI_D_S = 2 * np.log(3) / np.log(5)


# -----------------------------------------------------------
# Run directory (no overwrite)
# -----------------------------------------------------------

def make_run_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"RUN_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# -----------------------------------------------------------
# Graph construction
# -----------------------------------------------------------

def sierpinski_gasket_edges(level: int) -> Tuple[int, List[Tuple[int, int]]]:

    if level < 0:
        raise ValueError("level must be >= 0")

    if level == 0:
        return 3, [(0,1),(1,2),(0,2)]

    N_prev, edges_prev = sierpinski_gasket_edges(level-1)

    def offset(edges, off):
        return [(u+off,v+off) for u,v in edges]

    offT = 0
    offL = N_prev
    offR = 2*N_prev

    edges = []
    edges += offset(edges_prev,offT)
    edges += offset(edges_prev,offL)
    edges += offset(edges_prev,offR)

    parent = list(range(3*N_prev))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a,b):
        ra,rb = find(a),find(b)
        if ra != rb:
            parent[rb] = ra

    A,B,C = 0,1,2

    union(offT+B, offL+A)
    union(offT+C, offR+A)
    union(offL+C, offR+B)

    rep_to_new = {}
    new_id = 0

    def new(x):
        nonlocal new_id
        r = find(x)
        if r not in rep_to_new:
            rep_to_new[r] = new_id
            new_id += 1
        return rep_to_new[r]

    edge_set = set()

    for u,v in edges:
        nu,nv = new(u),new(v)
        if nu == nv:
            continue
        if nu > nv:
            nu,nv = nv,nu
        edge_set.add((nu,nv))

    return new_id, sorted(edge_set)


# -----------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------

def adjacency_from_edges(N:int,edges)->np.ndarray:

    A = np.zeros((N,N),dtype=float)

    for u,v in edges:
        A[u,v]=1
        A[v,u]=1

    return A


def compute_laplacian(adj):

    deg = np.sum(adj,axis=1)
    L = np.diag(deg) - adj

    eigenvalues,_ = np.linalg.eigh(L)

    return np.sort(eigenvalues)


# -----------------------------------------------------------
# Capacity sweep
# -----------------------------------------------------------

def default_capacities(N:int)->List[int]:

    base=[3,5,8,13,21]
    fracs=[0.1,0.2,0.35,0.5,0.7,0.85,1.0]

    caps = base + [max(3,int(f*N)) for f in fracs]

    return sorted(set([c for c in caps if 2<=c<=N]))


# -----------------------------------------------------------
# Experiment
# -----------------------------------------------------------

def run_experiment(levels:List[int],n_tau:int=90):

    data = {}
    summary_rows=[]

    for level in levels:

        print(f"Level {level}")

        N,edges = sierpinski_gasket_edges(level)

        adj = adjacency_from_edges(N,edges)

        eigenvalues = compute_laplacian(adj)

        lam_max = eigenvalues[-1] if eigenvalues[-1]>0 else 1

        taus = np.logspace(-3,np.log10(5e3/lam_max),n_tau)

        caps = default_capacities(N)

        # vectorized matrix
        exp_matrix = np.exp(-np.outer(taus,eigenvalues))

        P_full = np.mean(exp_matrix,axis=1)

        level_data = {
            "N":N,
            "taus":taus.tolist(),
            "eigenvalues":eigenvalues.tolist(),
            "P_full":P_full.tolist(),
            "capacities":{}
        }

        for C in caps:

            P_C = np.mean(exp_matrix[:,:C],axis=1)

            tau_crit = float("inf")
            if C-1 < len(eigenvalues) and eigenvalues[C-1] > 0:
                tau_crit = 1/eigenvalues[C-1]

            level_data["capacities"][C] = {
                "P_C":P_C.tolist(),
                "tau_crit":tau_crit
            }

            summary_rows.append([
                level,
                N,
                C,
                C/N,
                tau_crit
            ])

        data[level]=level_data

    return data,summary_rows


# -----------------------------------------------------------
# Save
# -----------------------------------------------------------

def save_outputs(run_dir,data,summary):

    with open(run_dir/"data.json","w") as f:
        json.dump(data,f,indent=2)

    with open(run_dir/"summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["level","N","C_obs","C_ratio","tau_crit"])
        w.writerows(summary)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output",default="./results")
    parser.add_argument("--levels",type=int,nargs="*",default=[2,3,4])
    parser.add_argument("--n_tau",type=int,default=90)

    args=parser.parse_args()

    base = Path(args.output)

    run_dir = make_run_dir(base)

    print("Run directory:",run_dir)

    data,summary = run_experiment(
        levels=args.levels,
        n_tau=args.n_tau
    )

    save_outputs(run_dir,data,summary)

    manifest={
        "levels":args.levels,
        "n_tau":args.n_tau,
        "spectral_dimension":float(SIERPINSKI_D_S)
    }

    with open(run_dir/"manifest.json","w") as f:
        json.dump(manifest,f,indent=2)

    print("Finished")

    return 0


if __name__=="__main__":
    sys.exit(main())
