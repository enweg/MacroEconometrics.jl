# MacroEconometrics

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://enweg.github.io/MacroEconometrics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://enweg.github.io/MacroEconometrics.jl/dev/)
[![Build Status](https://github.com/enweg/MacroEconometrics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/enweg/MacroEconometrics.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Roadmap

**Basics**

- [x] VAR model
  - [x] Simulation
  - [ ] Bayesian
    - [ ] Minnesota
    - [ ] SSVS
    - [x] Indpendent Normal-Wishart
  - [ ] Frequentist
    - [x] LS
    - [ ] Bootstrap CI
  - [ ] Forecasting

- [ ] SVAR model
  - [ ] Simulation
  - [ ] Estimation --> Direct 
  - [ ] Identification
    - [ ] Cholesky
    - [ ] Short-run restrictions
    - [ ] Long-run restrictions
    - [ ] Heteroskedastic
    - [ ] Sign
    - [ ] Proxy VAR

- [ ] Impulse Response Functions
  - [ ] Reduced Form
    - [ ] From VAR
  - [ ] Structural
    - [ ] From SVAR
    - [ ] Identification from VAR
    - [ ] Instrumental Variables

- [ ] Local Projections
- [ ] Visualisation
- [ ] Data Wrangling
- [ ] Replicability
- [ ] FEVD
  - [ ] SVAR
- [ ] Historical Decomposition
  - [ ] SVAR
- [ ] Forecast Scenarios
  - [ ] SVAR
- [ ] Policy Counterfactuals


## Basic Structure

- Every macroeconometric model should just contain the most basic structure
  - It should contain only those elements that one would also have to write
    down, abstracting away from how it is estimated.
  - Parts that need to be estimated should be a subtype of `Estimated` which
    should either be Frequentist or Bayesian
  - If a part is fixed (because it was used in a simulation), then it should be
    of type `Fixed`. These parts are as if we would completely know them
- IRFs, Forecasts, FEVD, ... should all use the `Estimated` type of parts of
  their results depend on estimations.  