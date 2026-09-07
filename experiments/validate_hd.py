"""Validate the high-dimensional protocol before spending cluster time."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.hd import (rstar_ambient, rstar_test, random_projections,
                            pca_projection, certified_rstar_bound, ensemble_test)

rng = np.random.default_rng(0)

print("=" * 72)
print("1. CALIBRATION of the ambient r* test (must hold Type-I at nominal level)")
print("   null: both clouds i.i.d. from the SAME distribution, in R^512")
ps = []
for rep in range(300):
    Z = rng.normal(size=(120, 512))
    r = rstar_test(Z[:60], Z[60:], B=199, rng=rep)
    ps.append(r["p_sep"])
ps = np.array(ps)
for a in (0.01, 0.05, 0.10):
    print(f"     alpha={a:<5} empirical rejection = {(ps <= a).mean():.3f}"
          f"   (binomial se {np.sqrt(a*(1-a)/len(ps)):.3f})")

print()
print("=" * 72)
print("2. POWER: separated classes in high dimension should be certified")
print("   two Gaussians in R^512 at increasing separation, n=60 each")
for gap in (0.0, 0.5, 1.0, 2.0, 4.0):
    mu = np.zeros(512); mu[0] = gap
    X = rng.normal(size=(60, 512)); Y = rng.normal(size=(60, 512)) + mu
    r = rstar_test(X, Y, B=199, rng=1)
    print(f"     gap={gap:<4} r*={r['rstar_obs']:.3f}  sep_index={r['separation_index']:.3f}"
          f"  p_sep={r['p_sep']:.3f}")

print()
print("=" * 72)
print("3. TIGHTNESS: how much does the projected tier lose, vs K projections?")
print("   ambient r* vs best-of-K projected bound (d0=5, R^512)")
mu = np.zeros(512); mu[0] = 3.0
X = rng.normal(size=(60, 512)); Y = rng.normal(size=(60, 512)) + mu
pooled = np.vstack([X, Y])
for K in (1, 4, 16, 64):
    projs = random_projections(512, 5, K, seed=7)
    b = certified_rstar_bound(X, Y, projs)
    print(f"     K={K:<3} bound={b['rstar_bound']:.3f}  ambient={b['rstar_ambient']:.3f}"
          f"  tightness={b['tightness']:.3f}")
pca = pca_projection(pooled, 5)
b = certified_rstar_bound(X, Y, [pca])
print(f"     PCA   bound={b['rstar_bound']:.3f}  ambient={b['rstar_ambient']:.3f}"
      f"  tightness={b['tightness']:.3f}   <- signal is in the top PC here")

print()
print("=" * 72)
print("4. EXACTNESS of the ensemble profile test (label-independent ensemble)")
print("   null data, K=4 projections, aggregate=mean")
projs = random_projections(64, 5, 4, seed=3)
ps = []
t0 = time.time()
for rep in range(60):
    Z = rng.normal(size=(80, 64))
    r = ensemble_test([Z[:40], Z[40:]], projs, B=49, rng=rep)
    ps.append(r["p_down"])
ps = np.array(ps)
for a in (0.05, 0.10):
    print(f"     alpha={a:<5} empirical rejection (p_down) = {(ps <= a).mean():.3f}"
          f"   (se {np.sqrt(a*(1-a)/len(ps)):.3f})")
print(f"     [{time.time()-t0:.0f}s for 120 reps x 99 perms x 4 projections]")

print()
print("=" * 72)
print("5. COST: ambient tier on a foundation-model-sized problem")
for D in (768, 4096):
    X = rng.normal(size=(200, D)); Y = rng.normal(size=(200, D))
    t0 = time.time(); rstar_test(X, Y, B=199, rng=0); dt = time.time() - t0
    print(f"     D={D:<5} n=200/class, B=199 permutations: {dt:.2f}s per pair")
rng=np.random.default_rng(0)

print("A. Why ambient r* fails: distance concentration vs class signal")
print("   isotropic N(0,I) in R^D, classes separated by `gap` in one coordinate")
print(f"   {'D':>5} {'gap':>5} {'ambient r*':>11} {'r* if gap=0':>12} {'signal/noise':>13}")
for D in (8,64,256,768):
    for gap in (0.0,4.0):
        mu=np.zeros(D); mu[0]=gap
        X=rng.normal(size=(80,D)); Y=rng.normal(size=(80,D))+mu
        r=rstar_ambient(X,Y)
        X0=rng.normal(size=(80,D)); Y0=rng.normal(size=(80,D))
        r0=rstar_ambient(X0,Y0)
        if gap>0: print(f"   {D:>5} {gap:>5} {r:>11.3f} {r0:>12.3f} {r/r0:>13.3f}")

print()
print("B. Tightness of the projection bound vs D and vs data anisotropy")
print("   'isotropic': signal in 1 coord, noise in all D")
print("   'low-id'   : realistic representation -- spectrum decaying like 1/k")
print(f"   {'D':>5} {'kind':>10} {'PCA bound':>10} {'ambient':>9} {'tightness':>10}")
for D in (64,256,768):
    # isotropic
    mu=np.zeros(D); mu[0]=4.0
    X=rng.normal(size=(80,D)); Y=rng.normal(size=(80,D))+mu
    b=certified_rstar_bound(X,Y,[pca_projection(np.vstack([X,Y]),5)])
    print(f"   {D:>5} {'isotropic':>10} {b['rstar_bound']:>10.3f} {b['rstar_ambient']:>9.3f} {b['tightness']:>10.3f}")
    # low intrinsic dimension: eigenvalues ~ 1/k, class signal in top directions
    scale=1.0/np.arange(1,D+1)
    Xl=rng.normal(size=(80,D))*scale; Yl=rng.normal(size=(80,D))*scale
    Yl[:,0]+=1.5*scale[0]*4
    b=certified_rstar_bound(Xl,Yl,[pca_projection(np.vstack([Xl,Yl]),5)])
    print(f"   {D:>5} {'low-id':>10} {b['rstar_bound']:>10.4f} {b['rstar_ambient']:>9.4f} {b['tightness']:>10.3f}")
