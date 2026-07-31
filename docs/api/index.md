# API Reference

fluxopt keeps a deliberately narrow public surface: what you import from the
top-level `fluxopt` package is the API, and nothing else is.

{%
   include-markdown "../../README.md"
   start="<!--public-api-start-->"
   end="<!--public-api-end-->"
%}

## Import style

Every public name is available from the package root, so a single import is
enough:

```python
import fluxopt as fx

fx.Flow(carrier='heat', size=100)
```

Importing from the module a name happens to live in — `fluxopt.elements.Flow`
— works today but is not covered by the stability policy. Prefer `fx.Flow`.

## Module pages

The pages below document the modules backing the public surface. Modules that
are not listed (validation, contracts, constraint builders, the benchmark
harness) are implementation detail: they carry no compatibility guarantee and
are intentionally left out of this reference.
