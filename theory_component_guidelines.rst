Contributor Guidelines
======================

How to add your theory component in solike
---------------

Basic ingredients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Your theory calculator must inherit from the cobaya theory class. It must have 3 main blocks of functions:

inizialization (``inizialize``);

requirements (``get_requirement``, ``must_provide``);

calculations (``get_X``, ``calculate``).

In what follows, we will use the structure of ``mflike`` as a concrete example of how to build these 3 blocks. The new version of ``mflike`` in SOLikeT splits the original mflike in 4 blocks: one cobaya-likelihood component (``mflike``); three cobaya-theory components.

The 3 theory components are:
	1. ``TheoryForge``: this is where raw theory (CMB spectra) is mixed and modified with instrumental and non-cosmological effects
	2. ``Foreground``: this is where the foreground (fg) spectra are computed
	3. ``BandPass``: this is where bandpasses are built (either analytically or read from file)


Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can either assign params in inizialize or do that via a dedicated yaml. You can in general do all the calculations that need to be done once for all.

Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here you need to write what external elements are needed by your theory block to perform its duties. These external elements will be computed and provided by some other external module (e.g., another Theory class).
In our case, ``mflike`` must tell us that it needs a dictionary of cmb+fg spectra. This is done by letting the get_requirement function return a dictionary which has the name of the needed element as a key. For example, if the cmb+fg spectra dict is called ``cmbfg_dict``, the get_requirement function should 

::

   return {"cmbfg_dict":{}}

The key is a dict itself. It can be empty, if no params need to be passed to the external Theory in charge of computing cmbfg_dict.
It might be possible that, in order to compute ``cmbfg_dict``, we should pass to the specific Theory component some params known by ``mflike`` (e.g., frequency channel). This is done by filling the above empty dict:

::

   {"cmbfg_dict": {"param1": param1_value, "param2": param2_value, etc}}

If this happens, then the external Theory block (in this example, ``TheoryForge``) must have a ``must_provide`` function.
``must_provide`` tells the code 
	1. what values should be assigned to the parameters needed to compute the element required from the Theory block. The required elements are stored in the ``**requirements`` dictionary which is the input of ``must_provide``.
	In our example, ``TheoryForge`` will assign to ``param1`` the ``param1_value`` passed from ``mflike`` via the ``get_requirement`` in ``mflike`` (and so on). For example:
	::

		must_provide(self, **requirements):
   			if "cmbfg_dict" in requirements:
    			self.param1 = requirements["cmbfg_dict"]["param1"]

	if this is the only job of ``must_provide``, then the function will not return anything


	2. if needed, what external elements are needed by this specific theory block to perform its duties. In this case, the function will return a dictionary of (empty or not) dictionaries which are the requirements of the specific theory block. Note this can be also done via ``get_requirement``. However, if you need to pass some params read from the block above to the new requirements, this can only be done with ``must_provide``. For example, ``TheoryForge`` needs ``Foreground`` to compute the fg spectra, which we store in a dict called ``fg_dict``. We also want ``TheoryForge`` to pass to ``Foreground`` ``self.param1``. This is done as follows:
	::

		must_provide(self, **requirements):
    		if “cmbfg_dict” etc etc
      			...
    		return {“fg_dict”: {“param1_fg”: self.param1}}

	Of course, ``Foreground`` will have a similar call to ``must_provide``, where we assign to ``self.param1_fg`` the value passed from ``TheoryForge`` to ``Foreground``.

Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In each Theory class, you need at least 2 functions:

	1. 
	::
		get_X(self, any_other_param):
   			return self.current_state[“X”]

	where "X" is the name of the requirement computed by that class (in our case, it is ``cmbfg_dict`` in ``TheoryForge``, ``fg_dict`` in ``Foreground``).
	"any_other_param" is an optional param that you may want to apply to ``current_state["X"]`` before returning it. E.g., it could be a rescaling amplitude. 
	This function is called by the Likelihood or Theory class that has "X" as its requirement, via the ``self.provider.get_X(any_other_param)`` call.

	2. 
	::
		calculate(self, **state, want_derived=False/True, **params_values_dict):
   			do actual calculations, that could involve the use of some of the     **params_value_dict, and might also compute derived params (if want_derived=True)
   			state[“X”] = result of above calculations



