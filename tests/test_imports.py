"""Import smoke tests.

Catches accidental reintroduction of import-time side effects (forcing a GUI
matplotlib backend, requiring torch to import the DAMASK/Abaqus side, etc.)
that would otherwise only surface when someone runs a script on a machine
without every optional dependency installed.
"""
import importlib


def test_ddms_top_level_imports_without_optional_deps():
	importlib.import_module('ddms')


def test_ddms_numeric_imports():
	importlib.import_module('ddms.numeric')
	importlib.import_module('ddms.numeric.processing')
	importlib.import_module('ddms.numeric.constitutive')
	importlib.import_module('ddms.numeric.visualize')
