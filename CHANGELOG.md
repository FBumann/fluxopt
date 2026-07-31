# Changelog

## [0.2.0](https://github.com/fluxopt/fluxopt/compare/v0.1.0...v0.2.0) (2026-07-31)


### ⚠ BREAKING CHANGES

* flat time axis with time_period coord and ragged periods ([#213](https://github.com/fluxopt/fluxopt/issues/213))
* **effects:** reject time-varying contribution_from into lump-bearing sources ([#258](https://github.com/fluxopt/fluxopt/issues/258))
* derive qualified flow ids at build time ([#246](https://github.com/fluxopt/fluxopt/issues/246))
* drop effect rate bounds, collapse temporal domain, nest data-model containers ([#242](https://github.com/fluxopt/fluxopt/issues/242))
* objective API rework — build_model, guards, rename ([#223](https://github.com/fluxopt/fluxopt/issues/223))

### refac

* derive qualified flow ids at build time ([#246](https://github.com/fluxopt/fluxopt/issues/246)) ([013a3f0](https://github.com/fluxopt/fluxopt/commit/013a3f0080623400820ccceb2d429d0be220c8ec))
* objective API rework — build_model, guards, rename ([#223](https://github.com/fluxopt/fluxopt/issues/223)) ([255601a](https://github.com/fluxopt/fluxopt/commit/255601a64e5a65556d355f0af90990039b82f641))


### Features

* **benchmark:** add abstract multi-period stress reference system ([#260](https://github.com/fluxopt/fluxopt/issues/260)) ([7433ff0](https://github.com/fluxopt/fluxopt/commit/7433ff05deae2f64fe2d579f6f87c66e3a872478))
* drop effect rate bounds, collapse temporal domain, nest data-model containers ([#242](https://github.com/fluxopt/fluxopt/issues/242)) ([841a69f](https://github.com/fluxopt/fluxopt/commit/841a69fda6331f71d510c1c6122aa04d0366781d))
* flat time axis with time_period coord and ragged periods ([#213](https://github.com/fluxopt/fluxopt/issues/213)) ([a10833c](https://github.com/fluxopt/fluxopt/commit/a10833c5d713ef4c4d90e7cc4bc09985e4f57e84))
* user-runnable benchmark with realistic reference systems ([#257](https://github.com/fluxopt/fluxopt/issues/257)) ([f0f049e](https://github.com/fluxopt/fluxopt/commit/f0f049ecd56203386d1795e122a3f5026ba1d4e2))


### Bug Fixes

* **effects:** reject time-varying contribution_from into lump-bearing sources ([#258](https://github.com/fluxopt/fluxopt/issues/258)) ([78a737d](https://github.com/fluxopt/fluxopt/commit/78a737db044fa4ae844da2b2a58ae393ba9825de))
* **status:** correct build gating, reject profile combo, warn on prior-dt assumption ([#251](https://github.com/fluxopt/fluxopt/issues/251)) ([26c5db1](https://github.com/fluxopt/fluxopt/commit/26c5db1fb13ddc213fc1b16f3f1316e09ce5a610))


### Performance Improvements

* **effects:** fold Leontief inverse into term coefficients, not expressions ([#270](https://github.com/fluxopt/fluxopt/issues/270)) ([dfb5f31](https://github.com/fluxopt/fluxopt/commit/dfb5f3165cfda540ce2fb180c05ec7e7bee38c91))

## [0.1.0](https://github.com/fluxopt/fluxopt/compare/v0.0.8...v0.1.0) (2026-07-21)


### ⚠ BREAKING CHANGES

* fail-fast FlowSystem validation, elements on BaseModel ([#229](https://github.com/fluxopt/fluxopt/issues/229))
* **effects:** objective_effects no longer accepts a list; penalty is minimized by default (was: tracked but ignored).
* unify field naming to quantity_min/max grammar ([#204](https://github.com/fluxopt/fluxopt/issues/204))

### refac

* unify field naming to quantity_min/max grammar ([#204](https://github.com/fluxopt/fluxopt/issues/204)) ([3459c6c](https://github.com/fluxopt/fluxopt/commit/3459c6cb7d465458fe557070a6161c93bc03e258))


### Features

* **effects:** allow per-period values for periodic bounds ([#205](https://github.com/fluxopt/fluxopt/issues/205)) ([d8dfc4d](https://github.com/fluxopt/fluxopt/commit/d8dfc4de603f42c8a4e48672b679a5a8b612db19))
* **effects:** weighted objective with penalty auto-inclusion ([#210](https://github.com/fluxopt/fluxopt/issues/210)) ([20cef8f](https://github.com/fluxopt/fluxopt/commit/20cef8fe0a4691055c0b5bdfa293e56fbf321b58))
* element round-trip (to_dict/from_dict) + ProfileRef ([#227](https://github.com/fluxopt/fluxopt/issues/227)) ([16a229a](https://github.com/fluxopt/fluxopt/commit/16a229a222d9dc55886e3d443fdfaff387c4c3a2))
* fail-fast FlowSystem validation, elements on BaseModel ([#229](https://github.com/fluxopt/fluxopt/issues/229)) ([2673131](https://github.com/fluxopt/fluxopt/commit/2673131857ab8090c0a7e8e07d459b178a0c21a5))
* **flow:** add flow_hours and load_factor bounds ([#206](https://github.com/fluxopt/fluxopt/issues/206)) ([3dd45a7](https://github.com/fluxopt/fluxopt/commit/3dd45a71d2030ee1050537401f3962a8f04c68f7))
* **flow:** add ramp rate limits ([#207](https://github.com/fluxopt/fluxopt/issues/207)) ([785875b](https://github.com/fluxopt/fluxopt/commit/785875b398059319c478f6141822ff9d57571d3a))
* FlowSystem declarative front door (YAML + data) ([#228](https://github.com/fluxopt/fluxopt/issues/228)) ([916fb6c](https://github.com/fluxopt/fluxopt/commit/916fb6c43b313a2eb9d0df69da7eff2c3d4ea78a))
* pydantic element layer (validation + JSON Schema) ([#226](https://github.com/fluxopt/fluxopt/issues/226)) ([a3b2f50](https://github.com/fluxopt/fluxopt/commit/a3b2f50489de02c235ae523524ceaef07c073ea0))
* **stats:** add StatsAccessor.summary() headline KPI table ([#188](https://github.com/fluxopt/fluxopt/issues/188)) ([c3540f9](https://github.com/fluxopt/fluxopt/commit/c3540f94c304931f090dec4157e82020d9bb46c5))
* **storage:** add final level bounds and prevent_simultaneous ([#208](https://github.com/fluxopt/fluxopt/issues/208)) ([643fede](https://github.com/fluxopt/fluxopt/commit/643fede6de28b15485f7f38ac85959f180b0b4b3))


### Bug Fixes

* clarify netcdf read error for non-ASCII paths on Windows ([#195](https://github.com/fluxopt/fluxopt/issues/195)) ([c1c2ffe](https://github.com/fluxopt/fluxopt/commit/c1c2ffebe8107af4ba8e07d54785098b6bd03281))
* keep release-please below 1.0 with bump-minor-pre-major ([#232](https://github.com/fluxopt/fluxopt/issues/232)) ([fe539d3](https://github.com/fluxopt/fluxopt/commit/fe539d32d1b7aa0f5e93bc90760a490513529ac6))
* pass pd.Index coords for linopy 0.8 compatibility ([#194](https://github.com/fluxopt/fluxopt/issues/194)) ([9179a36](https://github.com/fluxopt/fluxopt/commit/9179a365c4948652b29339d0d9ada13765dad011))
* reject unknown flow short_ids in conversion_factors ([#212](https://github.com/fluxopt/fluxopt/issues/212)) ([de7ff7a](https://github.com/fluxopt/fluxopt/commit/de7ff7a10560c2cd53a6492e1e5e92248ff925c7))
* weight flow-hour aggregates by timestep duration ([#211](https://github.com/fluxopt/fluxopt/issues/211)) ([4320efe](https://github.com/fluxopt/fluxopt/commit/4320efe56fe5fe61f49c6f6739beab95015d8d5a))

## [0.0.8](https://github.com/FBumann/fluxopt/compare/v0.0.7-alpha.0...v0.0.8) (2026-05-31)


### Features

* **api:** allow period-varying flow profiles via DataFrame; rename TimeSeries → Variate ([#165](https://github.com/FBumann/fluxopt/issues/165)) ([a8b535c](https://github.com/FBumann/fluxopt/commit/a8b535cd5ed0c4f4b3cd065136c594e5ad7c2d9f))


### Bug Fixes

* exclude all highspy 1.14.x versions ([#181](https://github.com/FBumann/fluxopt/issues/181)) ([5046e46](https://github.com/FBumann/fluxopt/commit/5046e466260b7c85070a355d8d504a13303d8736))


### Miscellaneous Chores

* release 0.0.8 ([d375ec5](https://github.com/FBumann/fluxopt/commit/d375ec59c02f50a5c9183ca5b91ec4d5e5c93f02))

## [0.0.7-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.6-alpha.0...v0.0.7-alpha.0) (2026-04-29)


### Features

* add component-level status on Storage ([#145](https://github.com/FBumann/fluxopt/issues/145)) ([a536e4f](https://github.com/FBumann/fluxopt/commit/a536e4f3e40801392271e9881f57276bec2f260a))
* cache per-contributor effect contributions in Result ([#140](https://github.com/FBumann/fluxopt/issues/140)) ([2f9b8d0](https://github.com/FBumann/fluxopt/commit/2f9b8d048541f01686269b6418b6b81f4bc7f82a))
* piecewise conversion via linopy add_piecewise_formulation ([#147](https://github.com/FBumann/fluxopt/issues/147)) ([bf25e39](https://github.com/FBumann/fluxopt/commit/bf25e3999add3ff9a3bff056bf9af558499404fb))


### Bug Fixes

* **ci:** let config control prerelease default, dispatch overrides ([#111](https://github.com/FBumann/fluxopt/issues/111)) ([aadf16c](https://github.com/FBumann/fluxopt/commit/aadf16c6602d763dfd51da8856d240219248f38b))
* exclude highspy 1.14.0 due to MIP presolve bug and temporarilly the nevest xarray version ([#129](https://github.com/FBumann/fluxopt/issues/129)) ([2f753e4](https://github.com/FBumann/fluxopt/commit/2f753e4c8883cfdc99dfd12a0d1446638dda266a))
* include Investment costs in compute_effect_contributions ([#135](https://github.com/FBumann/fluxopt/issues/135)) ([6bc2d4b](https://github.com/FBumann/fluxopt/commit/6bc2d4b76352066794193a66d1373018a3aa98ce))
* scale effect per-hour bounds by timestep duration ([#127](https://github.com/FBumann/fluxopt/issues/127)) ([3473d9c](https://github.com/FBumann/fluxopt/commit/3473d9c1d3776d8f64bb24038ccebea4404b3368))
* use xarray-native comparison in effect_contributions validation ([#139](https://github.com/FBumann/fluxopt/issues/139)) ([e34d7f7](https://github.com/FBumann/fluxopt/commit/e34d7f7264a5f0c5643884459f52c0184831a394))

## [0.0.6-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.5-alpha.0...v0.0.6-alpha.0) (2026-03-14)


### Features

* make all effect parameters period-aware ([#94](https://github.com/FBumann/fluxopt/issues/94)) ([e0f8e66](https://github.com/FBumann/fluxopt/commit/e0f8e66e6237751f14322aeb88677a1c2a82d469))

## [0.0.5-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.4-alpha.0...v0.0.5-alpha.0) (2026-03-14)


### Features

* add Investment for multi-period build-timing optimization ([#85](https://github.com/FBumann/fluxopt/issues/85)) ([fe70bf8](https://github.com/FBumann/fluxopt/commit/fe70bf87133def20288484e9ea658bea43a9e770))

## [0.0.4-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.3-alpha.0...v0.0.4-alpha.0) (2026-03-13)


### Features

* add multi-period optimization ([#81](https://github.com/FBumann/fluxopt/issues/81)) ([2df9100](https://github.com/FBumann/fluxopt/commit/2df9100f226db05e4a5530262dce184ce7de55bf))

## [0.0.3-alpha.0](https://github.com/FBumann/fluxopt/compare/v0.0.2-alpha.0...v0.0.3-alpha.0) (2026-03-11)


### Features

* prepare core for package ecosystem ([#59](https://github.com/FBumann/fluxopt/issues/59)) ([0c77bd0](https://github.com/FBumann/fluxopt/commit/0c77bd06882d6b154c87649a4a75b52e0edb8a79))
* replace Bus with Carrier ([#60](https://github.com/FBumann/fluxopt/issues/60)) ([c3035e1](https://github.com/FBumann/fluxopt/commit/c3035e18e565d75f8c0c2de1220e3a0936826236))
* require explicit carriers in optimize() ([#65](https://github.com/FBumann/fluxopt/issues/65)) ([e6b27d6](https://github.com/FBumann/fluxopt/commit/e6b27d61e5e3029f661b9eeb8eacbb1fb54eec28))
* require explicit carriers in optimize() and ModelData.build() ([#64](https://github.com/FBumann/fluxopt/issues/64)) ([f07df79](https://github.com/FBumann/fluxopt/commit/f07df79bf0c016d256ba14d85c4f3978f09eda5f))

## Changelog
