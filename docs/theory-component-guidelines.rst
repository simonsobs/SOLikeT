Contributor Guidelines
======================

How to add your theory component in soliket
-------------------------------------------

Basic ingredients
^^^^^^^^^^^^^^^^^
Your theory calculator must inherit from the cobaya theory class. It must have 3 main blocks of functions:

initialization (``initialize``);

requirements (``get_requirements``, ``must_provide``);

calculations (``get_X``, ``calculate``).

In what follows, we will use the structure of ``mflike`` as a concrete example of how to build these 3 blocks. mflike splits the original mflike into 2 blocks:
one cobaya-likelihood component (``mflike``), and a cobaya-theory component ``BandpowerForeground`` to calculate the foreground model.


Initialization
^^^^^^^^^^^^^^
You can either assign params in initialize or do that via a dedicated yaml. You can in general do all the calculations that need to be done once for all.

Requirements
^^^^^^^^^^^^
Here you need to write what external elements are needed by your theory block to perform its duties. These external elements will be computed and provided by some other external module (e.g., another Theory class).
In our case, ``mflike`` must tell us that it needs a dictionary of cmb and fg spectra. This is done by letting the get_requirement function return a dictionary which has the name of the needed element as a key. For example, if the cmb+fg spectra dict is called ``fg_totals``, the get_requirement function should be::

   return {"fg_totals":{}}

The key is a dict itself. It can be empty, if no params need to be passed to the external Theory in charge of computing fg_totals.
It might be possible that, in order to compute ``fg_totals``, we should pass to the specific Theory component some params known by ``mflike`` (e.g., frequency channel). This is done by filling the above empty dict::

   {"fg_totals": {"param1": param1_value, "param2": param2_value, etc}}

If this happens, then the external Theory block (in this example, ``BandpowerForeground``) must have a ``must_provide`` function. ``must_provide`` tells the code:

   1. The values which should be assigned to the parameters needed to compute the element required from the Theory block. The required elements are stored in the
   ``**requirements`` dictionary which is the input of ``must_provide``.
   In our example, ``BandpowerForeground`` will assign to ``param1`` the ``param1_value`` passed from ``mflike`` via the ``get_requirement`` in ``mflike`` (and so on). For example:
   ::

        must_provide(self, **requirements):
           if "fg_totals" in requirements:
              self.param1 = requirements["fg_totals"]["param1"]

   if this is the only job of ``must_provide``, then the function will not return anything

   2. If required, what external elements are needed by this specific theory block to perform its duties. In this case, the function will return a dictionary of dictionaries which are the requirements of the specific theory block. These dictionaries do not have to necessarily contain content (they can be empty instances of the dictionary), but must be included if expected. Note this can be also done via ``get_requirement``. However, if you need to pass some params read from the block above to the new requirements, this can only be done with ``must_provide``.

Calculation
^^^^^^^^^^^
In each Theory class, you need at least 2 functions:

   1. A get function:
   ::

      get_X(self, any_other_param):
         return self.current_state[“X”]

   where "X" is the name of the requirement computed by that class (in our case, it is ``fg_totals`` in ``BandpowerForeground``). ``any_other_param`` is an optional param that you may want to apply to ``current_state["X"]`` before returning it. E.g., it could be a rescaling amplitude. This function is called by the Likelihood or Theory class that has ``X`` as its requirement, via the ``self.provider.get_X(any_other_param)`` call.

   2. A calculate function:
   ::

      calculate(self, **state, want_derived=False, **params_values_dict):
         state[“X”] = result of above calculations

   which will do actual calculations, that could involve the use of some of the ``**params_value_dict``, and might also compute derived params (if ``want_derived=True``).
