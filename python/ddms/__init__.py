# `numeric` (damask/orix) and `surrogate` (torch/torch_geometric) pull in
# disjoint, heavy dependency sets - importing `ddms` used to require both to
# be installed even if you only needed one. Import the submodule you need
# directly, e.g. `import ddms.numeric` or `import ddms.surrogate`.