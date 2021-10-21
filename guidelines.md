# Bringing a new likelihood into SOLikeT

The following lays out what needs to be done to bring a new likelihood (e.g., **newlike**)
into the **soliket** framework.  Note that at present not all of these have yet been done
as stated below for the currently implemented likelihoods, but this will come.

* Is **newlike** Gaussian or Poisson?  If so, great; if not, then we need to do some
prep work to implement a generic version of the new likelihood form 
into **SOLikeT**, alongside `GaussianLikelihood` and `PoissonLikelihood`.

* Test **newlike** *in situ*:
    * Is there a minimal installation procedure available for the original likelihood? 
      (i.e., can a new user install & use **newlike** without heavy dependencies?) 
    * Is there a minimal test suite for the orignal implementation?  A bare minimum test is a demo
    `test_newlike.yaml` file specifying a test model, and corresponding
    `test_newlike.py` test file that loads that model and compares the computed likelihood to a fiducial value.
    (See, e.g., `soliket/tests/test_mflike.py` or `soliket/tests/test_lensing.py`.)

* Port **newlike** into **soliket**:
    * Put `test_newlike.yaml` and `test_newlike.py` files into `soliket/tests`.  These should be the same 
    as the original versions of these files, with the sole exception that paths to the referenced likelihoods
    should be `soliket.newlike` instead of the original paths.  When these tests pass,
    then we know the likelihood is correctly implemented.
    * Create a submodule for the new likelihood in `soliket/newliket`, and copy the original code over into this.

* Once ported code passes tests, refactor **newlike** to follow **soliket** conventions:
    * Rewrite likelihood code so as to inherit from `GaussianLikelihood`, implementing `_get_data()`, `_get_cov()`,
    `_get_theory()` methods, etc.
    * Also, if there is substantial data to be used by the likelihood, have it also extend 
    `_InstallableLikelihood` (see `soliket.mflike` and `soliket.lensing` for examples).
    * Factor out all cosmological/astrophysical calculations necessary to compute the "theory vector" into separate standalone
    `Theory` objects (current example of this is the `Foreground` object in `soliket.mflike`.)
    * If any of your new modules require physical constants, please hack the constants.py module in soliket (if you need to add new entries) and import it as needed. This would avoid inconsistent definitions of potentially shared quantities. Don't re-define constants in your own modules.

Development regarding **newlike** in the soliket repository should either happen in a branch named `dev-newlike`.  
Submit a pull request and request review to merge to master.
