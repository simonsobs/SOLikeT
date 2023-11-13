r"""
.. module:: foreground_marginalization

:Synopsis: Perform the foreground marginalization for the TTTEEE CMB power spectra for SO.
:Authors: Hidde Jense.
"""
import numpy as np
import tqdm
import os
import sys
from soliket.mflike.ForegroundMarginalizer import ForegroundMarginalizer
from soliket import BandPass, Foreground
from soliket.mflike.theoryforge_MFLike import TheoryForge_MFLike

"""
You can modify the code below here!
"""

outdir = "fgmarge"

os.makedirs(outdir, exist_ok = True)

bandpass = BandPass({
    "read_from_sacc" : True,
})

fgmodel = Foreground()

thfo = TheoryForge_MFLike()

fgmarge = ForegroundMarginalizer({
    # Put your nondefault settings in here.
    "input_file" : "LAT_simu_sacc_00000.fits",
    "cov_Bbl_file" : "data_sacc_w_covar_and_Bbl.fits",
    "data_folder" : "ForegroundMarginalizer/v0.8",
    "bandpass" : bandpass,
    "foregrounds" : fgmodel,
    "theoryforge" : thfo,
    "lmax_extract" : {
        "tt" : 7000,
        "te" : 7000,
        "ee" : 7000
    }
})

# Hard prior bounds for the parameters.
param_ranges = {
    "a_tSZ" : [ 0.0, 10.0 ],
    "a_kSZ" : [ 0.0, 10.0 ],
    "a_p" : [ 0.0, 50.0 ],
    "a_c" : [ 0.0, 50.0 ],
    "beta_p" : [ 1.8, 2.2 ],
    "beta_c" : [ 2.0, 2.4 ],
    "a_s" : [ 0.0, 10.0 ],
    "a_gtt" : [ 0.0, 50.0 ],
    "a_gte" : [ 0.0, 1.0 ],
    "a_gee" : [ 0.0, 1.0 ],
    "a_pste" : [ -1.0, 1.0 ],
    "a_psee" : [ 0.0, 1.0 ],
    "xi" : [ 0.0, 0.2 ],
}

fixed_params = {
    "T_d" : 9.6,
    "cal_LAT_93" : 1,
    "cal_LAT_145" : 1,
    "cal_LAT_225" : 1,
    "calT_LAT_93" : 1,
    "calT_LAT_145" : 1,
    "calT_LAT_225" : 1,
    "calE_LAT_93" : 1,
    "calE_LAT_145" : 1,
    "calE_LAT_225" : 1,
    "alpha_LAT_93" : 0,
    "alpha_LAT_145" : 0,
    "alpha_LAT_225" : 0,
    "bandint_shift_LAT_93": 0,
    "bandint_shift_LAT_145": 0,
    "bandint_shift_LAT_225": 0,
    "calG_all": 1,
}

# The starting point, initial covmat proposal, and names of all parameters.
starting_point = np.array([ 3.3, 1.6, 6.9, 2.1, 4.9, 2.1, 3.1, 8.7, 0.1, 0.01, 0.01, 0.0, 0.1 ])
proposal = np.diag([ 0.7, 1.0, 0.2, 0.1, 0.5, 0.1, 0.5, 0.45, 0.07, 0.06, 0.05, 0.05, 0.05 ]) / 100.0
param_names = [ "a_tSZ", "a_kSZ", "a_p", "beta_p", "a_c", "beta_c", "a_s", "a_gtt", "a_gte", "a_gee", "a_psee", "a_pste", "xi" ]

update_proposal_every = 100


def update_proposal(data, Nsplits = 4):
    """
    A helper function to increase the convergence rate - it updates the covariance matrix
    based on the current accepted samples.
    """
    # Returns R-1, new proposal
    weights = data[:,0].astype(int)
    samples = data[:,1:]
    
    print(f"Computing new proposal over {weights.shape[0]:d} data points.")
    
    cut = int(weights.shape[0]) // (Nsplits+1)
    
    if cut == 0:
        print(f"\tNot enough data points. Skipping for now.")
        return np.inf, None
    
    ranges = [ (i * cut, (i+1) * cut) for i in range(1, Nsplits+1) ]
    print(f"Splits are {ranges}")
    
    Ns = np.array([ np.sum(weights[f:l]) for f,l in ranges ])
    
    means = np.array([ np.average(samples[f:l,:], weights = weights[f:l], axis = 0) for f,l in ranges ])
    covs = np.array([ np.cov(samples[f:l,:].T, fweights = weights[f:l]) for f, l in ranges ])
    
    mean_of_covs = np.average(covs, weights = Ns, axis = 0)
    cov_of_means = np.atleast_2d(np.cov(means.T, fweights = Ns))
    
    d = np.sqrt(np.diag(cov_of_means))
    corr_of_means = (cov_of_means / d).T / d
    norm_mean_of_covs = (mean_of_covs / d).T / d
    
    try:
        L = np.linalg.cholesky(norm_mean_of_covs)
    except np.linalg.LinAlgError:
        print(f"Mean of covs is non-deomposable. Skipping for now.")
        return np.inf, None
    
    Linv = np.linalg.inv(L)
    
    try:
        eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
    except np.linalg.LinAlgError:
        print(f"Correlation matrix has no eigenvalues. Skipping for now.")
        return np.inf, None
    
    Rminus1 = max(np.abs(eigvals))
    
    # We can now update proposal -> sqrt(mean of covs)
    s = np.diag(np.sqrt(np.diag(mean_of_covs)))
    sinv = np.linalg.inv(s)
    r = sinv @ mean_of_covs @ sinv
    Lprime = np.linalg.cholesky(r)
    new_proposal = s @ Lprime
    
    return Rminus1, new_proposal


def logprior(**params):
    """
    Any log(prior) function you want, given the parameters.
    """
    prior = { k : -1e10 for k in params.keys() }
    
    for p in params.keys():
        if p in param_ranges:
            pmin, pmax = param_ranges[p]
            if pmin < params[p] and params[p] < pmax:
                prior[p] = 0.0
    
    prior["a_gtt"] -= ((params["a_gtt"] - 2.79) / (2.0 * 0.45)) ** 2.0
    prior["a_gte"] -= ((params["a_gte"] - 0.36) / (2.0 * 0.04)) ** 2.0
    prior["a_gee"] -= ((params["a_gee"] - 0.13) / (2.0 * 0.03)) ** 2.0
    prior["a_s"] -= ((params["a_s"] - 3.1) / (2.0 * 0.4)) ** 2.0
    
    return sum([ prior[k] for k in prior ])


weight = 0
accepted = 0

last_update = 0
burn_in = 0
Rminus1_last = np.inf
Rminus1_best = 30.0
accepted_samples = None

current_point = starting_point.copy()
pbar = tqdm.tqdm(range(300000))

param_dict = { k : v for k, v in zip(param_names, current_point) }

fgmarge.make_mapping_matrix(**param_dict, **fixed_params)

print("Starting chain...")

np.savetxt(f"{outdir}/leff.txt", fgmarge.mapping_ls)

with open(f"{outdir}/samples.txt", "w") as fp:
    fp.write("# weight\tchi2\t" + "\t".join([ "%s" % s for s in param_names ]) + "\n")

with open(f"{outdir}/extracted.txt", "w") as fp:
    fp.write("# TT - TE - EE\n")

with open(f"{outdir}/progress.txt", "w") as fp:
    fp.write("# Samples\tAccepted\tR-1\n")

np.savetxt(f"{outdir}/covmat.txt", proposal @ proposal.T, header = "#" + " ".join(param_names))

for n in pbar:
    param_dict = { k : v for k, v in zip(param_names, current_point) }
    
    # If calibration is fixed, you can comment out this line to speed up the code.
    fgmarge.make_mapping_matrix(**param_dict, **fixed_params)
    
    fg_vec = fgmarge.get_theory_vector(None, **param_dict, **fixed_params)
    cl_vec = fgmarge.extract(fg_vec)
    
    with open(f"{outdir}/extracted.txt", "a") as fp:
        fp.write("\t".join([ "%.8e" % i for i in cl_vec ]) + "\n")
    
    weight += 1
    
    oldprior = logprior(**param_dict)
    oldlike = fgmarge.loglike(None, cl_vec, **param_dict, **fixed_params)
    oldpost = oldprior + oldlike
    
    new_point = current_point + proposal @ np.random.normal(loc = 0.0, scale = 1.0, size = current_point.shape)
    newparams = { k : v for k, v in zip(param_names, new_point) }
    newprior = logprior(**newparams)
    newlike = 0.0
    newpost = newprior
    accept = False
    
    if newprior > -1e8:
        newlike = fgmarge.loglike(None, cl_vec, **newparams, **fixed_params)
        newpost = newlike + newprior
        
        rat = newpost - oldpost
        
        accept = (newpost > oldpost) or (np.exp(rat) > np.random.random())
    
    if accept:
        with open(f"{outdir}/samples.txt", "a") as fp:
            fp.write(f"{weight:d}\t")
            fp.write(f"{-2.0 * newlike:.5f}\t")
            fp.write("\t".join([ "%.5f" % i for i in current_point ]))
            fp.write("\n")
        
        if burn_in > 0:
            burn_in -= 1
        else:
            old_sample = np.atleast_2d(np.concatenate(([weight], current_point)))
            
            if accepted_samples is None:
                accepted_samples = old_sample
            else:
                accepted_samples = np.vstack(( accepted_samples, old_sample ))
            
            last_update += 1
            if last_update >= update_proposal_every:
                print(f"Proposal update at {accepted+1:6d} accepted samples.")
                last_update = 0
                weights = accepted_samples[:,0]
                m = int(0.3 * sum(weights))
                Rminus1, new_proposal = update_proposal(accepted_samples[np.cumsum(weights) >= m,:])
                Rminus1_last = Rminus1
                
                if Rminus1 < Rminus1_best:
                    print(f"\tAccepted new proposal matrix.")
                    proposal = new_proposal
                    Rminus1_best = Rminus1
                    np.savetxt(f"{outdir}/covmat.txt", proposal @ proposal.T, header = "#" + " ".join(param_names))
                
                print(f"\tAccepted samples: {accepted+1}")
                print(f"\tR-1 = {Rminus1}")
                sys.stdout.flush()
                
                with open(f"{outdir}/progress.txt", "a") as fp:
                    fp.write(f"{n+1:7d}\t{accepted+1:7d}\t{Rminus1:e}\t{Rminus1_best:e}\n")
        
        current_point = new_point.copy()
        accepted += 1
        weight = 0
    
    pbar.set_description(f"Acc rate: {(accepted+1)/(n+1):7.1%}, R-1: {Rminus1_last:.2e}")

if weight > 0:
    with open("samples.txt", "a") as fp:
        fp.write(f"{weight:d}\t")
        fp.write(f"{-2.0 * oldlike:.5f}\t")
        fp.write("\t".join([ "%.5f" % i for i in current_point ]))
        fp.write("\n")
