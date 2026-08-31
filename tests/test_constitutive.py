"""Characterization test for ddms.numeric.constitutive.

c_G has no file I/O and no plotting side effects, so it can be exercised
directly. This is a "golden master" test: it pins the function's current
output for a fixed input so that future refactors of the numeric package
can be checked for unintended behavior changes, without requiring us to
independently re-derive the underlying physics.
"""
from types import SimpleNamespace

import pytest

from ddms.numeric.constitutive import c_G


def test_c_G_shear_modulus_temperature_dependence():
	prm = SimpleNamespace(T=298.0, u0=27.0e9, Tm=933.0, theta=0.5)

	out = c_G(prm)

	assert out.G == pytest.approx(24_028_434_524.572903, rel=1e-9)


def test_c_G_at_melting_point_is_zero():
	prm = SimpleNamespace(T=933.0, u0=27.0e9, Tm=933.0, theta=0.5)

	out = c_G(prm)

	assert out.G == pytest.approx(0.0, abs=1e-6)
