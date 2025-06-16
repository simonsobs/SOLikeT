import cobaya
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sacc

from cobaya.yaml import yaml_load_file
from cobaya.install import install
from cobaya.model import get_model

# read in the cobaya info
info = yaml_load_file('run_mflike_fiducial.yaml')

fiducial_cosmo = yaml_load_file('params_cosmo_smooth_fiducial.yaml')
fiducial_fg = yaml_load_file('params_mflikefg_smooth_fiducial.yaml')
fiducial_sys = yaml_load_file('params_mflikesyst_smooth_fiducial.yaml')

fiducial_params = {**fiducial_cosmo, **fiducial_fg, **fiducial_sys}
info['params'] = fiducial_params

# ensure all components are installed
# install(info)

# force model computation at fiducial parameters
model = get_model(info)
model.loglikes()

# extract CMB + FG spectra
Dl = model.components[2].get_Cl(ell_factor=True)
fg_totals = model.components[3].get_fg_totals()

mflike = model.components[0]
dls = {s: Dl[s][mflike.l_bpws] for s, _ in mflike.lcuts.items()}
#combine CMB and FG and add svstematics
DlsObs = mflike.get_modified_theory(dls, fg_totals, **fiducial_params)

# construct a sacc file
ps_dic = {}
ps_vec = np.zeros_like(model.components[0].data_vec)

for m in model.components[0].spec_meta:
    p = m["pol"]
    ids = m["ids"]
    w = m["bpw"]
    t1 = m["t1"]
    t2 = m["t2"]
    
    #print(p, ids, t1, t2)
    if (t1 + "x" + t2 not in ps_dic.keys ()):
        ps_dic[t1 + "x" + t2] = {"lbin": m["leff"]}
        
    dls_obs = DlsObs[p, t2, t1] if m["hasYX_xsp"] else DlsObs[p, t1, t2]
    for i, nonzero, weights in zip (m["ids"], w.nonzeros, w.sliced_weights):
        ps_vec[i] = weights @ dls_obs[nonzero]
        
    ps_dic[t1 + "x" + t2].update({p: ps_vec[ids]})

namedir = './data/'

for k in ps_dic.keys():
    namefile = "Dl_" + k + "_auto_00000.dat"
    ell = ps_dic[k]["lbin"]
    tt = ps_dic[k]["tt"]
    te = ps_dic[k]["te"]
    ee = ps_dic[k]["ee"]
    tbebbb = np.zeros(len(ell))
    np.savetxt(namedir + namefile, 
        np.column_stack((ell, tt, te, tbebbb, te, tbebbb, ee, tbebbb, tbebbb, tbebbb)))

data = {}
sim_suffix = "00000"
for spec_name in ps_dic.keys():
    na, nb = spec_name.split("x")
    data[na, nb] = {}
    spec = np.loadtxt("%s/Dl_%s_auto_%s.dat" % (namedir, spec_name, sim_suffix), 
        unpack=True)
    ps = {"lbin": spec[0],
          "TT": spec[1],
          "TE": spec[2],
          "TB": spec[3],
          "ET": spec[4],
          "BT": spec[5],
          "EE": spec[6],
          "EB": spec[7],
          "BE": spec[8],
          "BB": spec[9]}
    data[na, nb] = ps

exp_freq = ['LAT_93', 'LAT_145', 'LAT_225']
pols = ["T", "E", "B"]
map_types = {"T": "0", "E": "e", "B": "b"}


def get_x_iterator():
    for id_efa, efa in enumerate(exp_freq):
        for id_efb, efb in enumerate(exp_freq):
            if (id_efa > id_efb): continue
            for ipa, pa in enumerate(pols):
                if (efa == efb):
                    polsb = pols[ipa:]
                else:
                    polsb = pols
                for pb in polsb:
                    yield (efa, efb, pa, pb)
                    print(efa, efb, pa, pb)

spec_sacc = sacc.Sacc()

for exp_f in exp_freq:
    print("%s_s0" % (exp_f))

    my_data_bandpasses = {"nu": np.array([float(exp_f.split("_")[1])]), 
                          "b_nu": np.array([1.])}
    my_data_beams = {"l": np.arange(10000), "bl": np.ones(10000)}

    # CMB temperature
    spec_sacc.add_tracer("NuMap", "%s_s0" % (exp_f),
        quantity="cmb_temperature", spin=0,
        nu=model.components[0].bands[exp_f + "_s0"]["nu"],
        bandpass=model.components[0].bands[exp_f + "_s0"]["bandpass"],
        ell=my_data_beams["l"],
        beam=my_data_beams["bl"])

    # CMB polarization
    spec_sacc.add_tracer("NuMap", "%s_s2" % (exp_f),
        quantity="cmb_polarization", spin=2,
        nu=model.components[0].bands[exp_f + "_s2"]["nu"],
        bandpass=model.components[0].bands[exp_f + "_s2"]["bandpass"],
        ell=my_data_beams["l"],
        beam=my_data_beams["bl"])

for id_x, (efa, efb, pa, pb) in enumerate(get_x_iterator()):
    if pa == "T":
        ta_name = "%s_s0" % (efa)
    else:
        ta_name = "%s_s2" % (efa)

    if pb == "T":
        tb_name = "%s_s0" % (efb)
    else:
        tb_name = "%s_s2" % (efb)

    if pb == "T":
        cl_type = "cl_" + map_types[pb] + map_types[pa]
    else:
        cl_type = "cl_" + map_types[pa] + map_types[pb]

    lbin = data[efa, efb]["lbin"]
    cb = data[efa, efb][pa + pb]

    spec_sacc.add_ell_cl(cl_type, ta_name, tb_name, lbin, cb)

spec_sacc.save_fits("%s/data_sacc_smooth_%s.fits" % (namedir, sim_suffix), overwrite=True)
