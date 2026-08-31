"""Regression tests for the dependency-light math helpers in ddms.numeric.processing.

These cover the pure strain/stress-curve utilities only (no Abaqus/DAMASK I/O),
since those are the only functions in this module that can be exercised
without simulation output files or a running solver.
"""
import numpy as np
import pytest

from ddms.numeric.processing import (
	get_YS,
	get_UTS,
	get_engineering,
	get_logarithmic,
	get_uniform_stress,
	smoothing,
)


def test_get_YS_offset_method_on_bilinear_curve():
	# Elastic-perfectly-plastic curve: stress = E*strain up to the knee,
	# then flat at YSy. The 0.2% offset line runs from (0.002, 0) with
	# slope E, so on a perfectly flat plastic branch it only catches up
	# to the curve once it reaches that plateau height, at
	# strain = 0.002 + YSy/E.
	E = 70000.0		# MPa
	true_YSy = 300.0	# MPa
	knee_strain = true_YSy / E
	expected_YSx = 0.002 + true_YSy / E

	strain = np.linspace(0, 0.05, 5000)
	stress = np.where(strain <= knee_strain, E * strain, true_YSy)

	YSx, YSy = get_YS(strain, stress, E=E)

	assert YSy == pytest.approx(true_YSy, rel=1e-2)
	assert YSx == pytest.approx(expected_YSx, rel=1e-2)


def test_get_UTS_returns_curve_up_to_peak():
	strain = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
	stress = np.array([0.0, 100.0, 200.0, 150.0, 50.0])	# peak at index 2

	uts_strain, uts_stress = get_UTS(strain, stress)

	assert list(uts_strain) == [0.0, 0.01]
	assert list(uts_stress) == [0.0, 100.0]


def test_engineering_logarithmic_are_inverses():
	true_strain = np.linspace(0.0, 0.2, 50)
	true_stress = 500.0 * true_strain + 100.0

	eng_strain, eng_stress = get_engineering(true_strain, true_stress)
	round_trip_strain, round_trip_stress = get_logarithmic(eng_strain, eng_stress)

	np.testing.assert_allclose(round_trip_strain, true_strain, atol=1e-10)
	np.testing.assert_allclose(round_trip_stress, true_stress, atol=1e-8)


def test_get_uniform_stress_interpolates_linear_curve():
	strain = np.linspace(0, 0.1, 11)
	stress = 1000.0 * strain

	out = get_uniform_stress(strain, stress, l=0.0, r=0.1, num_pt=5)

	expected = 1000.0 * np.linspace(0, 0.1, 5)
	np.testing.assert_allclose(out, expected, atol=1e-8)


def test_smoothing_is_moving_average():
	data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

	out = smoothing(data, window_width=2)

	np.testing.assert_allclose(out, [1.5, 2.5, 3.5, 4.5])
