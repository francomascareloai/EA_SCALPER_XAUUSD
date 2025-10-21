@echo off
python -m mypy --strict --warn-redundant-casts --warn-unused-ignores --warn-unreachable --disallow-any-generics --disallow-untyped-defs --disallow-incomplete-defs --check-untyped-defs --disallow-untyped-decorators --no-implicit-optional --warn-return-any --no-implicit-reexport --strict-optional src tests
