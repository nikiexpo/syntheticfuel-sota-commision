# Source material

The PDFs in this directory are copyrighted and are not committed. They are listed here so
the modelling choices in `sfp/models/` can be traced to their sources.

## Subsystem models

- **Cormos, A.-M. & Simon, A. (2013).** "Dynamic modelling of CO2 capture by
  calcium-looping cycle." *Chemical Engineering Transactions* **35**, 421-426.
  DOI 10.3303/CET1335070. — 1D fast-fluidisation carbonator/calciner after Kunii &
  Levenspiel, including sorbent capacity decay with cycle number. Post-combustion; see the
  ambient-partial-pressure caveat in `docs/ASSUMPTIONS.md`.

- **Görgün, H. (2006).** "Dynamic modelling of a proton exchange membrane (PEM)
  electrolyzer." *International Journal of Hydrogen Energy* **31**(1), 29-38.
  DOI 10.1016/j.ijhydene.2005.04.001. — Control-oriented four-ancillary model (anode,
  cathode, membrane, voltage) with hydrogen storage dynamics. Lacks a stack thermal
  state, which this project adds.

- **Moioli, E., Gallandat, N. & Züttel, A. (2019).** "Model based determination of the
  optimal reactor concept for Sabatier reaction in small-scale applications over
  Ru/Al2O3." *Chemical Engineering Journal* **375**, 121954.
  DOI 10.1016/j.cej.2019.121954. — Three-zone reactor with axial heat-transfer
  optimisation; the basis for the lumped hotspot-constrained Sabatier model.

- **El Sibai, A. et al. (2016).** "Model-based optimal Sabatier reactor design for
  power-to-gas applications." *Energy Technology*.

- **Shakouri Kalfati, M. & Abdulla, A. (2025).** "An open-source dynamic model for direct
  air capture of carbon dioxide using solid sorbents." *Carbon Capture Science &
  Technology* **17**, 100516. — Solid-sorbent TVSA on zeolites, a *different* capture
  route from calcium looping. Used for its ambient temperature/humidity dependence
  structure and its siting methodology, not for the sorbent chemistry.

## Control and uncertainty

- **Multi-stage scenario-based oil production optimisation.** — Scenario-tree robust NMPC
  with non-anticipativity on the first stage; the structure of the M6 planner.

- **Importance subsampling for power system planning under multi-year demand and weather
  uncertainty.** — Principled selection of representative and extreme weather scenarios,
  used to keep the multistage planner tractable.

- **Robust co-design NMPC.** — Simultaneous design and control under uncertainty.

## Not in this directory (standard references used directly in code)

- Spencer, J. W. (1971). Fourier series for declination and the equation of time.
- Kasten, F. & Young, A. T. (1989). Revised optical air mass tables.
- Ineichen, P. & Perez, R. (2002). A new airmass independent formulation for the Linke
  turbidity coefficient.
- Liu, B. Y. H. & Jordan, R. C. (1960). Isotropic sky diffuse transposition.
- Faiman, D. (2008). Assessing the outdoor operating temperature of photovoltaic modules.
- Grasa, G. & Abanades, J. C. (2006). CO2 capture capacity of CaO in long series of
  carbonation/calcination cycles.
- Jordan, D. C. & Kurtz, S. R. (2013). Photovoltaic degradation rates.
