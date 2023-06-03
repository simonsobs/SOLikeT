# This class is taken from Pixell, written by Sigurd Naess. We don't need all of pixell
# for this, so for now we just take the Hankel transform class.
import numpy as np


class RadialFourierTransform:
    def __init__(self, lrange=None, rrange=None, n=512, pad=256):
        """Construct an object for transforming between radially
        symmetric profiles in real-space and fourier space using a
        fast Hankel transform. Aside from being fast, this is also
        good for representing both cuspy and very extended profiles
        due to the logarithmically spaced sample points the fast
        Hankel transform uses. A cost of this is that the user can't
        freely choose the sample points. Instead one passes the
        multipole range or radial range of interest as well as the
        number of points to use.

        The function currently assumes two dimensions with flat geometry.
        That means the function is only approximate for spherical
        geometries, and will only be accurate up to a few degrees
        in these cases.

        Arguments:
        * lrange = [lmin, lmax]: The multipole range to use. Defaults
          to [0.01, 1e6] if no rrange is given.
        * rrange = [rmin, rmax]: The radius range to use if lrange is
                not specified, in radians. Example values: [1e-7,10].
                Since we don't use spherical geometry r is not limited to 2 pi.
        * n: The number of logarithmically equi-spaced points to use
                in the given range. Default: 512. The Hankel transform usually
                doesn't need many points for good accuracy, and can suffer if
                too many points are used.
        * pad: How many extra points to pad by on each side of the range.
          Padding is useful to get good accuracy in a Hankel transform.
          The transforms this function does will return padded output,
                which can be unpadded using the unpad method. Default: 256
        """
        if lrange is None and rrange is None:
            lrange = [0.1, 1e7]
        if lrange is None:
            lrange = [1 / rrange[1], 1 / rrange[0]]
        logl1, logl2 = np.log(lrange)
        logl0 = (logl2 + logl1) / 2
        self.dlog = (logl2 - logl1) / n
        i0 = (n + 1) / 2 + pad
        self.ell = np.exp(logl0 + (np.arange(1, n + 2 * pad + 1) - i0) * self.dlog)
        self.r = 1 / self.ell[::-1]
        self.pad = pad

    def real2harm(self, rprof):
        """Perform a forward (real -> harmonic) transform, taking us from the
        provided real-space radial profile rprof(r) to a harmonic-space profile
        lprof(l). rprof can take two forms:
        1. A function rprof(r) that can be called to evalute the profile at
           arbitrary points.
        2. An array rprof[self.r] that provides the profile evaluated at the
           points given by this object's .r member.
        The transform is done along the last axis of the profile.
        Returns lprof[self.ell]. This includes padding, which can be removed
        using self.unpad"""
        import scipy.fft

        try:
            rprof = rprof(self.r)
        except TypeError:
            pass
        lprof = 2 * np.pi * scipy.fft.fht(rprof * self.r, self.dlog, 0) / self.ell
        return lprof

    def harm2real(self, lprof):
        """Perform a backward (harmonic -> real) transform, taking us from the
        provided harmonic-space radial profile lprof(l) to a real-space profile
        rprof(r). lprof can take two forms:
        1. A function lprof(l) that can be called to evalute the profile at
           arbitrary points.
        2. An array lprof[self.ell] that provides the profile evaluated at the
           points given by this object's .l member.
        The transform is done along the last axis of the profile.
        Returns rprof[self.r]. This includes padding, which can be removed
        using self.unpad"""
        import scipy.fft

        try:
            lprof = lprof(self.ell)
        except TypeError:
            pass
        rprof = scipy.fft.ifht(lprof / (2 * np.pi) * self.ell, self.dlog, 0) / self.r
        return rprof

    def unpad(self, *arrs):
        """Remove the padding from arrays used by this object. The
        values in the padded areas of the output of the transform have
        unreliable values, but they're not cropped automatically to
        allow for round-trip transforms. Example:
                r = unpad(r_padded)
                r, l, vals = unpad(r_padded, l_padded, vals_padded)"""
        if self.pad == 0:
            res = arrs
        else:
            res = tuple([arr[..., self.pad: -self.pad] for arr in arrs])
        return res[0] if len(arrs) == 1 else res
