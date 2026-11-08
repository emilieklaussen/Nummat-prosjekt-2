# Endringer i `prosjekt2_sortert.ipynb`

`prosjekt2_sortert.ipynb` er en sortert kopi av `prosjekt2_samlet.ipynb`.

**Ingen logikk er endret.** Det eneste som er gjort, er:
1. cellene er flyttet slik at like resultater står samlet,
2. variabel- og funksjonsnavn som ble brukt av flere personer, har fått personens navn som endelse,
3. det er lagt til markdown-overskrifter, og de tre gamle navneoverskriftene er fjernet.

All kode, alle kommentarer og alt output er ellers som før.

## Oppdatering: Amandas nye RK4 (bare numpy)

Amandas commit «ny rk4 bare med numpy» er lagt inn. Hennes 25 celler i `prosjekt2_samlet.ipynb` er byttet ut med cellene fra den nye `RK4.ipynb`, og cellene har samme plass i den sorterte filen som før, siden de svarer én-til-én til de gamle. Hva som er nytt hos Amanda:
- Modellfunksjonene og `ODEs` bruker numpy i stedet for JAX, og `@jax.jit` er fjernet.
- `RK4_jit` heter nå `RK4`.
- `ODEs_scipy` er fjernet, og `solve_ivp` bruker `ODEs` og `B_2D`, `grad_U_2D`, `B_tokamak` direkte.

Amandas nye funksjonsnavn `RK4` kolliderte med Selines `RK4`. Begge har derfor fått endelse: `RK4_amanda` og `RK4_seline`. Se tabellene under.

## 1. Ny rekkefølge


**1. Importer**

- *Emilie (Boris)*
  - kode: `import numpy as np`
- *Amanda (RK4)*
  - kode: `import numpy as np`
- *Seline (FOSMM)*
  - kode: `from scipy.integrate import solve_ivp`

**2. Modellproblemer**

- *Emilie (Boris)*
  - markdown: `## FRA AMANDA:`
  - kode: `#Tokamak magentic field`
- *Amanda (RK4)*
  - markdown: `#### Model problems`
  - kode: `# 2D charged particle`
  - kode: `# Tokamak magnetic field`

**3. Implementering av metodene**

- *Emilie (Boris)*
  - markdown: `# Implementation of Boris method`
  - kode: `def ODEs_emilie(t, y, B, U, Ugrad):`
- *Amanda (RK4)*
  - markdown: `### Classical Runge-Kutta 4 on the extended system`
  - kode: `#defining the numerical approximation of the derivative`
- *Seline (FOSMM)*
  - markdown: `Implementation of Fourth-order symmetric multistep method:`
  - kode: `def starting_values(y0, t_min, t_max, U_grad, B):`

**4. Task A: Kjøring og baneplott**

- *Emilie (Boris)*
  - kode: `N_emilie = int(10**5/ 0.1)`
  - kode: `x_array_boris_toko, v_array_boris_toko, _ = boris_method(ODEs_emili...`
  - markdown: `## Plotting trajectories over time`
  - kode: `plt.title("A: Trajectories for 2D charged particle using Boris")`
  - kode: `plt.title("A: Trajectories for Tokomak magnetic field using Boris")`
  - kode: `R_boris = np.sqrt(x_array_boris_toko[0]**2 + x_array_boris_toko[1]**2)`
- *Amanda (RK4)*
  - markdown: `## A - Plotting trajectories and error over long time`
  - kode: `# Parameters`
  - kode: `#Plotting the trajectories of the two first coordinates of the solu...`
  - kode: `# Parameters`
  - kode: `#Plotting the trajectories of the two first coordinates of the solu...`
- *Seline (FOSMM)*
  - markdown: `Task A on model 3.1 with Fourth-order symmetric multistep method:`
  - kode: `x01 = np.array([0.0, 1.0, 0.1])`
  - markdown: `Plotting for model 3.1:`
  - kode: `plt.title("A: Trajectory of 2D particle using FOSMM")`
  - markdown: `Task A on model 3.2 with Fourth-order symmetric multistep method:`
  - kode: `#Tokamak`
  - markdown: `Plotting for model 3.2:`
  - kode: `plt.title("A: Trajectory in Tokamak magnetic field in x-y plane usi...`
- *Seline (RK4)*
  - markdown: `slett etterpå`
  - kode: `#runge kutte for å verifisere`

**5. Task A: Referanseløsninger og feil over tid**

- *Emilie (Boris)*
  - kode: `T_emilie = 10**5`
  - kode: `sol_toko = solve_ivp(ODEs_emilie, t_span_emilie, initial_condition_...`
  - kode: `x1_emilie,x2_emilie,x3_emilie = sol.y[0:3]`
  - kode: `error_toko_boris = np.linalg.norm(x_array_boris_toko - sol_toko.y[0...`
- *Amanda (RK4)*
  - markdown: `-------------------------------------------------------------------...`
  - kode: `#Testing for a shorter time interval because`
  - kode: `#Plotting the error over time for the RK4 method`
  - kode: `#Plotting the error over time for both 2D paticle model`
- *Seline (FOSMM)*
  - markdown: `Compute referance`
  - kode: `t_span_seline = (0, T_max)`
  - markdown: `Computing and plotting error:`
  - kode: `def error(x, xref):`
- *Seline (RK4)*
  - kode: `error1_RK = error(x1RK, sol_2D_seline.y[:3])`

**6. Task A: Energi**

- *Emilie (Boris)*
  - kode: `def energy_change_boris(x,v, initial_values, U):`
  - kode: `plt.plot(t_eval_emilie[:2000], energy_change_boris_2D[:2000])`
- *Amanda (RK4)*
  - markdown: `---------------------------------------------------------------`
  - kode: `#Plotting the change in energy againt time`
- *Seline (FOSMM)*
  - markdown: `Calculating energy conservation:`
  - kode: `def energy_seline(x, v, U):`
  - markdown: `compare to theorem: h^-7 = 10^7 so should hold for the entire time ...`
  - markdown: `compare results`
  - kode: `plt.plot(t_eval_seline[:700*10], energy_change(x1_seline[:,:700*10]...`
  - kode: `plt.plot(t_eval_seline[:700*10], x1_seline[1,:700*10])`
  - markdown: `SAMMENLIKN ALLE RESULTATER MED FORVENTNINGER! se oppgaveteksten`
- *Seline (RK4)*
  - kode: `plt.title("Change in energy with RK4 with h = 0.1")`

**7. Task B: Konvergens og work-precision**

- *Emilie (Boris)*
  - kode: `def error_boris_decrease():`
  - kode: `error_2D_emilie, error_toka, time_2d, time_toka = error_boris_decre...`
  - kode: `plt.title("Global errors for different step sizes with Boris method")`
  - kode: `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))`
- *Amanda (RK4)*
  - markdown: `## Task B) Order verification and work precision diagrams`
  - kode: `#Paramaters`
  - kode: `#Tokamak`
  - kode: `#Solve_ivp for all three step sizes`
  - kode: `#Comuting the error in the last step`
  - kode: `#plotting the error vs. time with the RK4 method`
  - kode: `#The same only this time in two subplots`
- *Seline (FOSMM)*
  - markdown: `Task B) FOSMM`
  - kode: `T_seline = 1e4`
  - kode: `#CONST = 10**3`
  - markdown: `as we can see, the errors grow asymptotically as h**4`
  - kode: `plt.title("Global errors vs times for different step sizes with FOS...`
  - markdown: `which method most efficient for a given tolerrance?`
- *Seline (RK4 og sammenligning FOSMM/RK4)*
  - kode: `TRK = 1e4`
  - markdown: `since the plots of the error have a steeper slope than the h**4 plo...`

Selines RK4-celler (merket «slett etterpå») er fordelt på seksjon 4–7 etter hva de viser, under overskriften «Seline (RK4)».

## 2. Nye og fjernede markdown-celler

- **Lagt til:** én overskrift for hver seksjon (`# 1. Importer`, …, `# 7. Task B: Konvergens og work-precision`) og en underoverskrift for hver person i hver seksjon, for eksempel `## Emilie (Boris)`. Øverst står en ny tittelcelle, som erstatter den gamle `# Project 2: Numerical Integration For Charged Particles`, med en henvisning til denne filen.
- **Fjernet:** de tre overskriftene `# Emilie`, `# Amanda` og `# Seline`, fordi hver persons kode nå er spredt over flere seksjoner. Personen står i stedet i underoverskriftene.

## 3. Navn som er endret

Navnene fikk endelsen `_emilie`, `_amanda` eller `_seline` i alle celler som tilhører den personen. Det gjelder både definisjonen og alle steder navnet brukes. Navn på keyword-argumenter, som `t_eval=` i `solve_ivp`, er *ikke* endret, bare verdiene som sendes inn. Lokale variabler og parametre inne i funksjoner med samme navn har også fått endelsen. Det endrer ikke hva funksjonene gjør.

Personnavn er brukt i stedet for metodenavn som `_boris`, fordi Emilie allerede har en variabel som heter `t_eval_boris`, og fordi Seline har både FOSMM- og RK4-kode.

### Emilie

| Gammelt navn | Nytt navn | Celler der navnet er endret (første linje i cellen) |
|---|---|---|
| `A_2D` | `A_2D_emilie` | `#Tokamak magentic field` |
| `A_tokamak` | `A_tokamak_emilie` | `#Tokamak magentic field` |
| `B_2D` | `B_2D_emilie` | `#Tokamak magentic field`; `N_emilie = int(10**5/ 0.1)`; `T_emilie = 10**5`; `def error_boris_decrease():` |
| `B_tokamak` | `B_tokamak_emilie` | `#Tokamak magentic field`; `x_array_boris_toko, v_array_boris_toko, _ = b`; `sol_toko = solve_ivp(ODEs_emilie, t_span_emil`; `def error_boris_decrease():` |
| `N` | `N_emilie` | `def ODEs_emilie(t, y, B, U, Ugrad):`; `N_emilie = int(10**5/ 0.1)`; `x_array_boris_toko, v_array_boris_toko, _ = b`; `T_emilie = 10**5` |
| `ODEs` | `ODEs_emilie` | `def ODEs_emilie(t, y, B, U, Ugrad):`; `N_emilie = int(10**5/ 0.1)`; `x_array_boris_toko, v_array_boris_toko, _ = b`; `T_emilie = 10**5`; `sol_toko = solve_ivp(ODEs_emilie, t_span_emil`; `def error_boris_decrease():` |
| `T` | `T_emilie` | `T_emilie = 10**5`; `def error_boris_decrease():` |
| `U_2D` | `U_2D_emilie` | `#Tokamak magentic field`; `N_emilie = int(10**5/ 0.1)`; `T_emilie = 10**5`; `def energy_change_boris(x,v, initial_values, `; `def error_boris_decrease():` |
| `U_tokamak` | `U_tokamak_emilie` | `#Tokamak magentic field`; `x_array_boris_toko, v_array_boris_toko, _ = b`; `sol_toko = solve_ivp(ODEs_emilie, t_span_emil`; `def energy_change_boris(x,v, initial_values, `; `def error_boris_decrease():` |
| `error_2D` | `error_2D_emilie` | `error_2D_emilie, error_toka, time_2d, time_to`; `plt.title("Global errors for different step s`; `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=` |
| `grad_U_2D` | `grad_U_2D_emilie` | `#Tokamak magentic field`; `N_emilie = int(10**5/ 0.1)`; `T_emilie = 10**5`; `def error_boris_decrease():` |
| `grad_U_tokamak` | `grad_U_tokamak_emilie` | `#Tokamak magentic field`; `x_array_boris_toko, v_array_boris_toko, _ = b`; `sol_toko = solve_ivp(ODEs_emilie, t_span_emil`; `def error_boris_decrease():` |
| `h` | `h_emilie` | `def ODEs_emilie(t, y, B, U, Ugrad):`; `T_emilie = 10**5`; `def error_boris_decrease():`; `plt.title("Global errors for different step s` |
| `h_i` | `h_i_emilie` | `def error_boris_decrease():`; `plt.title("Global errors for different step s` |
| `initial_condition_2D` | `initial_condition_2D_emilie` | `T_emilie = 10**5`; `def energy_change_boris(x,v, initial_values, `; `def error_boris_decrease():` |
| `t_eval` | `t_eval_emilie` | `T_emilie = 10**5`; `sol_toko = solve_ivp(ODEs_emilie, t_span_emil`; `x1_emilie,x2_emilie,x3_emilie = sol.y[0:3]`; `error_toko_boris = np.linalg.norm(x_array_bor`; `def energy_change_boris(x,v, initial_values, `; `plt.plot(t_eval_emilie[:2000], energy_change_`; `def error_boris_decrease():` |
| `t_span` | `t_span_emilie` | `def ODEs_emilie(t, y, B, U, Ugrad):`; `T_emilie = 10**5`; `sol_toko = solve_ivp(ODEs_emilie, t_span_emil`; `def error_boris_decrease():` |
| `v0_2D` | `v0_2D_emilie` | `#Tokamak magentic field`; `N_emilie = int(10**5/ 0.1)`; `T_emilie = 10**5`; `def error_boris_decrease():` |
| `x0_2D` | `x0_2D_emilie` | `#Tokamak magentic field`; `N_emilie = int(10**5/ 0.1)`; `T_emilie = 10**5`; `def error_boris_decrease():` |
| `x1` | `x1_emilie` | `x1_emilie,x2_emilie,x3_emilie = sol.y[0:3]` |
| `x2` | `x2_emilie` | `x1_emilie,x2_emilie,x3_emilie = sol.y[0:3]` |
| `x3` | `x3_emilie` | `x1_emilie,x2_emilie,x3_emilie = sol.y[0:3]` |

### Amanda

| Gammelt navn | Nytt navn | Celler der navnet er endret (første linje i cellen) |
|---|---|---|
| `A_2D` | `A_2D_amanda` | `# 2D charged particle` |
| `A_tokamak` | `A_tokamak_amanda` | `# Tokamak magnetic field` |
| `B_2D` | `B_2D_amanda` | `# 2D charged particle`; `# Parameters`; `#Testing for a shorter time interval because`; `#Plotting the error over time for the RK4 met`; `#Paramaters`; `#Solve_ivp for all three step sizes` |
| `B_tokamak` | `B_tokamak_amanda` | `# Tokamak magnetic field`; `# Parameters`; `#Plotting the error over time for the RK4 met`; `#Tokamak`; `#Solve_ivp for all three step sizes` |
| `ODEs` | `ODEs_amanda` | `#defining the numerical approximation of the `; `#Testing for a shorter time interval because`; `#Plotting the error over time for the RK4 met`; `#Solve_ivp for all three step sizes` |
| `RK4` | `RK4_amanda` | `#defining the numerical approximation of the `; `# Parameters`; `# Parameters`; `#Paramaters`; `#Tokamak` |
| `T` | `T_amanda` | `# Parameters`; `# Parameters`; `#Plotting the error over time for the RK4 met`; `#Paramaters`; `#Solve_ivp for all three step sizes` |
| `U_2D` | `U_2D_amanda` | `# 2D charged particle`; `#Plotting the change in energy againt time` |
| `U_tokamak` | `U_tokamak_amanda` | `# Tokamak magnetic field`; `#Plotting the change in energy againt time` |
| `energy` | `energy_amanda` | `#Plotting the change in energy againt time` |
| `error_2D` | `error_2D_amanda` | `#Plotting the error over time for both 2D pat` |
| `grad_U_2D` | `grad_U_2D_amanda` | `# 2D charged particle`; `# Parameters`; `#Testing for a shorter time interval because`; `#Plotting the error over time for the RK4 met`; `#Paramaters`; `#Solve_ivp for all three step sizes` |
| `grad_U_tokamak` | `grad_U_tokamak_amanda` | `# Tokamak magnetic field`; `# Parameters`; `#Plotting the error over time for the RK4 met`; `#Tokamak`; `#Solve_ivp for all three step sizes` |
| `h` | `h_amanda` | `#defining the numerical approximation of the `; `# Parameters`; `# Parameters`; `#Plotting the error over time for the RK4 met`; `#Paramaters` |
| `h_i` | `h_i_amanda` | `#Paramaters`; `#Tokamak`; `#Solve_ivp for all three step sizes`; `#Comuting the error in the last step` |
| `initial_condition_2D` | `initial_condition_2D_amanda` | `#Testing for a shorter time interval because`; `#Plotting the error over time for the RK4 met`; `#Plotting the change in energy againt time`; `#Solve_ivp for all three step sizes` |
| `sol_2D` | `sol_2D_amanda` | `#Plotting the error over time for the RK4 met`; `#Plotting the error over time for both 2D pat` |
| `t_eval` | `t_eval_amanda` | `#Plotting the error over time for the RK4 met`; `#Plotting the error over time for both 2D pat`; `#Plotting the change in energy againt time` |
| `t_span` | `t_span_amanda` | `#Plotting the error over time for the RK4 met` |
| `time` | `time_amanda` | `# Parameters`; `# Parameters` |
| `v0_2D` | `v0_2D_amanda` | `# 2D charged particle`; `# Parameters`; `#Testing for a shorter time interval because`; `#Plotting the error over time for the RK4 met`; `#Plotting the change in energy againt time`; `#Paramaters` |
| `x0_2D` | `x0_2D_amanda` | `# 2D charged particle`; `# Parameters`; `#Testing for a shorter time interval because`; `#Plotting the error over time for the RK4 met`; `#Plotting the change in energy againt time`; `#Paramaters` |
| `x3` | `x3_amanda` | `#Plotting the trajectories of the two first c` |

### Seline

| Gammelt navn | Nytt navn | Celler der navnet er endret (første linje i cellen) |
|---|---|---|
| `N` | `N_seline` | `def starting_values(y0, t_min, t_max, U_grad,`; `#runge kutte for å verifisere`; `t_span_seline = (0, T_max)`; `T_seline = 1e4`; `TRK = 1e4` |
| `RK4` | `RK4_seline` | `#runge kutte for å verifisere`; `TRK = 1e4` |
| `T` | `T_seline` | `T_seline = 1e4` |
| `energy` | `energy_seline` | `def energy_seline(x, v, U):` |
| `h` | `h_seline` | `def starting_values(y0, t_min, t_max, U_grad,`; `x01 = np.array([0.0, 1.0, 0.1])`; `#Tokamak`; `#runge kutte for å verifisere`; `t_span_seline = (0, T_max)`; `error1_RK = error(x1RK, sol_2D_seline.y[:3])`; `def energy_seline(x, v, U):`; `T_seline = 1e4`; `TRK = 1e4` |
| `sol_2D` | `sol_2D_seline` | `t_span_seline = (0, T_max)`; `def error(x, xref):`; `error1_RK = error(x1RK, sol_2D_seline.y[:3])` |
| `t_eval` | `t_eval_seline` | `t_span_seline = (0, T_max)`; `def error(x, xref):`; `def energy_seline(x, v, U):`; `plt.plot(t_eval_seline[:700*10], energy_chang`; `plt.plot(t_eval_seline[:700*10], x1_seline[1,`; `T_seline = 1e4`; `TRK = 1e4` |
| `t_span` | `t_span_seline` | `t_span_seline = (0, T_max)`; `T_seline = 1e4`; `TRK = 1e4` |
| `x1` | `x1_seline` | `x01 = np.array([0.0, 1.0, 0.1])`; `plt.title("A: Trajectory of 2D particle using`; `def error(x, xref):`; `def energy_seline(x, v, U):`; `plt.plot(t_eval_seline[:700*10], energy_chang`; `plt.plot(t_eval_seline[:700*10], x1_seline[1,` |
| `x2` | `x2_seline` | `#Tokamak`; `plt.title("A: Trajectory in Tokamak magnetic `; `def error(x, xref):`; `def energy_seline(x, v, U):` |

**Hvorfor:**
- Modellfunksjonene og startverdiene (`B_2D`, `A_2D`, `U_2D`, `grad_U_2D`, `B_tokamak`, `A_tokamak`, `U_tokamak`, `grad_U_tokamak`, `x0_2D`, `v0_2D`) fantes både hos Emilie og hos Amanda.
- `time`: Amanda brukte `time = np.linspace(...)`, som overskrev `time`-modulen som Emilies `boris_method` bruker til tidtaking. Bare *variabelen* er endret. `import time` og `import time as pytime` er som før.
- `ODEs`: Emilie og Amanda hadde hver sin funksjon med samme navn, med fem og fire argumenter.
- `RK4`: Amanda (ny versjon) og Seline hadde hver sin RK4-funksjon med samme navn og ulike argumenter.
- `energy`: Amanda og Seline hadde ulike funksjoner med samme navn og ulike argumenter.
- `sol_2D`: Amanda og Seline brukte samme navn på ulike referanseløsninger.
- Parametre og hjelpevariabler (`T`, `h`, `h_i`, `N`, `t_eval`, `t_span`, `error_2D`, `initial_condition_2D`, `x1`, `x2`, `x3`) ble satt på nytt av flere personer med ulike verdier, og ville overskrevet hverandre etter sorteringen.

## 4. Navn som fortsatt er felles (bevisst ikke endret)

- `np`, `plt`, `solve_ivp`, `time` (modulen): samme import hos alle, og derfor ufarlige.
- `i`, `fig`, `_`: midlertidige variabler som alltid settes på nytt rett før de brukes, i samme celle.

## 5. Hvordan det er testet

- En statisk sjekk av den nye rekkefølgen viser at ingen funksjon eller variabel brukes før den er definert.
- Emilies, Amandas (ny versjon) og Selines deler ble kjørt hver for seg, og den sorterte notebooken ble kjørt i sin helhet, alt med kortere sluttid T. Alle resultatvariabler (baner, feil, energi, referanseløsninger) er identiske. Bare tidsmålingene varierer, som ventet.
- Output i notebooken er fra deres tidligere kjøringer, og cellene er ikke kjørt på nytt. Kjøringsnumrene står derfor ikke i rekkefølge. Kjør gjerne hele notebooken på nytt fra toppen (*Run All*).

## 6. Ikke gjort, ifølge valg A

Funksjonene som er definert flere ganger, er *ikke* slått sammen. Det gjelder modellproblemene (tre versjoner), høyresiden av ODE-en (`ODEs_emilie`, `ODEs_amanda`, `ODE1`), referanseløsningene, energifunksjonene, de to RK4-implementeringene (`RK4_amanda` og `RK4_seline`) og Amandas doble work-precision-plott.

---

# Branch `FirstGenChange`: felles funksjoner og farger

Disse endringene finnes bare i branchen `FirstGenChange`, ikke på `main`.

## A. Én felles versjon av modellproblemene, ODE-en og energien

I seksjon 2 (Modellproblemer) står det nå to celler som alle tre bruker:

1. **Felles modellproblemer:** `B_2D`, `A_2D`, `U_2D`, `grad_U_2D`, `x0_2D`, `v0_2D`, `B_tokamak`, `A_tokamak`, `U_tokamak`, `grad_U_tokamak`, `x0_tokamak` og `v0_tokamak`.
2. **Felles funksjoner:** `ODEs(t, y, B, grad_U)`, `energy(x, v, U)` og `energy_change(x, v, U)`.

**Disse er fjernet og erstattet med de felles versjonene:**

| Hvem | Fjernet | Erstattet med |
|---|---|---|
| Emilie | `B_2D_emilie`, `A_2D_emilie`, `U_2D_emilie`, `grad_U_2D_emilie`, `x0_2D_emilie`, `v0_2D_emilie`, `B_tokamak_emilie`, `A_tokamak_emilie`, `U_tokamak_emilie`, `grad_U_tokamak_emilie`, `x0_tokomak`, `v0_tokomak` | de felles navnene uten endelse |
| Emilie | `ODEs_emilie(t, y, B, U, Ugrad)` | `ODEs(t, y, B, grad_U)`. U-argumentet ble aldri brukt, så `args=(B, U, grad_U)` i `solve_ivp` er endret til `args=(B, grad_U)`, også inne i `boris_method`. |
| Amanda | Hele tokamak-cellen, i tillegg til `B_2D_amanda`, `A_2D_amanda`, `U_2D_amanda`, `grad_U_2D_amanda`, `B_2D_np`, `grad_U_2D_np`, `B_tokamak_np`, `x0_2D_amanda`, `v0_2D_amanda` og markdown-cellen «Model problems» | de felles navnene. `_np`-funksjonene var kopier og ble ikke brukt. |
| Amanda | `ODEs_amanda` | `ODEs` (identisk) |
| Seline | `B1`, `A1`, `U1`, `U_grad1`, `x01`, `v01`, `B2`, `A2`, `U2`, `U_grad2`, `x02`, `v02` | de felles navnene. `y01` og `y02` finnes fortsatt, men lages nå fra `x0_2D`/`v0_2D` og `x0_tokamak`/`v0_tokamak`. |
| Seline | `ODE1` | `ODEs` (identisk) |
| Seline | `energy_seline`, `energy_change` (flyttet) | `energy`, `energy_change` i fellescellen |
| Amanda | `energy_amanda(y, U)` | `energy(x, v, U)`. `E0_2D`, `E0_tokamak`, `e_n_2D` og `e_n_tokamak` regnes ut som før, bare med den felles funksjonen. |
| Emilie | Energiformelen inne i `energy_change_boris` | `energy(...)`. Selve `energy_change_boris` er beholdt, fordi Boris har v på halve steg og trenger egen behandling. |

**Merk:** `A_tokamak` bruker nå `arctan2(x[1], x[0])`, slik Seline gjorde. Emilie og Amanda brukte `arctan(x[1]/x[0])`, som gir feil verdi når x[0] < 0. A brukes bare av FOSMM, så dette endrer ingen resultater for Boris og RK4.

**Ikke slått sammen:** referanseløsningene, fordi de har ulike toleranser og sluttider, og en sammenslåing ville endret resultatene. Det samme gjelder Selines egen RK4 (`RK4_seline`, merket «slett etterpå») og feilfunksjonene.

## B. Farger

| Metode | Farger |
|---|---|
| Boris (Emilie) | blå: `cornflowerblue` (2D), `royalblue` (Tokamak), `steelblue` (R–x₃-plott), `navy` (referanselinje) |
| RK4 (Amanda og Selines RK4) | lilla: `mediumorchid` (2D), `darkviolet` (Tokamak), `mediumpurple` (R–x₃-plott), `rebeccapurple` (referanselinje), `mediumvioletred` (2D i work-precision, som før) |
| FOSMM (Seline) | grønn: `mediumseagreen` (2D), `darkgreen` (Tokamak), `seagreen` (R–z-plott og x₂-komponent), `olivedrab` (referanse- og teorilinje) |

I sammenligningsplottet FOSMM/RK4 er FOSMM-kurvene grønne og RK4-kurvene lilla.

## C. Testing

Notebooken er kjørt med kortere sluttid T og sammenlignet med versjonen på `main`. 90 av 91 resultatvariabler er bitvis identiske. Den siste, `energy_change_boris_toko`, avviker med 8·10⁻²², som er ren avrunding.

---

# Task C: `TaskC` og `borisC` slått sammen i `main`

- **Nytt:** cellene fra `TaskC` (RK4) og `borisC` (Boris) står etter hverandre under overskriften `# Task C`, med underoverskriftene `## Task C RK4 method` og `## Task C Boris method`. Output fra deres egne kjøringer er beholdt.
- **Fjernet:** én tom kodecelle fra `TaskC`.
- **Nye navn**, så Task C ikke overskriver variabler fra task A. Logikken er ikke endret:
  - i RK4-delen: `R` → `R_RK4_C` og `x3` → `x3_RK4_C`
  - i Boris-delen: `R_boris` → `R_boris_C` og `T` → `T_boris_C`
- **To nye plottceller** under `## Task C: Sammenligning av RK4 og Boris`:
  1. Baner i 2×2-rutenett: $(x_1, x_2)$ og $(R, x_3)$ for RK4 og Boris, som figur 6 i He mfl. (2015).
  2. Relativ energifeil $(H_n - H_0)/|H_0|$ for RK4 og Boris i samme plott, og Boris alene ved siden av, som figur 7(a) og 7(b).
- **Outputet til de nye cellene** er laget ved å kjøre bare definisjonscellene og Task C-cellene, ikke task A og B. Resultat: RK4 ender på −0.99 i relativ energifeil, og Boris holder seg under 8·10⁻¹⁴. Det stemmer med artikkelen.

---

# Branch `rydding-navn`: navn, felles parametre og figurnummer

## Navn uten personnavn
Alle variabler og funksjoner er nå oppkalt etter metode, modell eller oppgave. `_2D` og `_tok` står for modellen, og `_A`, `_B` og `_C` for oppgaven.

| Før | Nå |
|---|---|
| `RK4_amanda`, `h_amanda`, `h_emilie`, `h_seline`, `N_emilie`, `N_seline` (inne i metodene) | `RK4_step`, `h`, `N` |
| `x_array_boris`, `x_array_boris_toko`, `v_array_boris(_toko)` | `x_boris_2D`, `x_boris_tok`, `v_boris_2D`, `v_boris_tok` |
| `x_arr_2D`, `x_arr_tokamak`, `y_arr_2D`, `y_arr_tokamak` | `x_RK4_2D`, `x_RK4_tok`, `y_RK4_2D`, `y_RK4_tok` |
| `x1_seline`, `v1`, `x2_seline`, `v2` | `x_FOSMM_2D`, `v_FOSMM_2D`, `x_FOSMM_tok`, `v_FOSMM_tok` |
| `R`, `x3_amanda`, `R_boris`, `z` | `R_RK4_tok`, `z_RK4_tok`, `R_boris_tok`, `z_boris_tok` |
| `error_boris_2d`, `error_toko_boris`, `error_2D_amanda`, `error_tokamak`, `error1_FOSMM`, `error2_FOSMM` | `err_boris_2D`, `err_boris_tok`, `err_RK4_2D`, `err_RK4_tok`, `err_FOSMM_2D`, `err_FOSMM_tok` |
| `energy_change_boris_2D/_toko`, `e_n_2D`, `e_n_tokamak` | `dE_boris_2D`, `dE_boris_tok`, `dE_RK4_2D`, `dE_RK4_tok` |
| `initial_condition_2D_emilie/_amanda`, `initial_condition_toko/_tokamak`, `y01`, `y02` | `y0_2D`, `y0_tok`, `y0_FOSMM_2D`, `y0_FOSMM_tok` |
| Task B: `error_2D_emilie_T1` … `T4`, `error_2D_RK4_T1` …, `e1`, `e2`, `t1`, `t2` | Lister med én oppføring per T: `err_boris_2D_B[k]`, `err_RK4_tok_B[k]`, `err_FOSMM_2D_B[k]`, `time_boris_2D_B[k]` osv., der `k` er indeksen i `T_B` |
| Overskriftene `## Emilie (Boris)`, `## Amanda (RK4)`, `## Seline (FOSMM)` | `## Boris`, `## RK4`, `## FOSMM` |

## Ett sett parametre per oppgave, i stedet for mange like
- **Task A:** `T_A = 1e5`, `h_A = 0.1`, `N_A`, tidsaksen `t_A` (med `t_A_boris = t_A[1:]`) og startverdiene står i én ny celle øverst i task A. Disse er fjernet: `T_emilie`, `h_emilie`, `N_emilie`, `T_amanda`, `h_amanda`, `n_steps` (definert to ganger), `time_amanda`, `t_eval_amanda`, `T_max`, `h_seline`, `t_eval_seline` og `t_eval_emilie`.
- **Task B:** `T_B = [1e4, 1e3, 1e2, 10]` og `h_B = [0.1, 0.05, 0.025]` står i én ny celle. Disse er fjernet: `T_emilie`, `T_RK4`, `T_seline_vals`, `h_i_emilie` (definert to ganger), `h_i_amanda` (definert to ganger), `h_arr` og `h_p`.

## Én referanseløsning per modell i task A
Før ble det regnet ut tre referanseløsninger per modell i task A, én for hver metode. Nå er det én: `sol_A_2D` og `sol_A_tok`, på tidsaksen `t_A`, med `rtol=1e-12` og `atol=1e-14`. Boris sammenlignes med `sol_A_*.y[:, 1:]`, fordi `boris_method` ikke lagrer $x_0$.

**Merk:** Boris og RK4 brukte før `rtol=1e-10`. Nå bruker de samme, strengere toleranse som FOSMM. Feilverdiene i task A endrer seg derfor litt for Boris og RK4. I testen var det relative avviket mellom $10^{-8}$ og $10^{-4}$, og det er ikke synlig i plottene.

## Fjernet fordi det var dobbelt
- Cellene som kjørte RK4 for task B med T = 1e4 én gang til (`y_arr_2D_i`, `y_arr_tokamak_i`, `sol_2D_i`, `sol_tokamak_i`, `error_2D_i`, `error_tokamak_i`). Resultatene ble ikke brukt i noen plott, og `error_RK4_decrease` regner ut det samme.
- Testkjøringen av `solve_ivp` på et kort intervall i task A.
- Notatet «FIKS SÅ VI BRUKER SAMME ENERGIFUNKSJON …». Det er gjort: alle tre bruker `energy`.

## Rettede feil
- **Konvergensplottene i task B:** RK4-panelet viste Boris sine Tokamak-feil (`error_toka_T1` …) i stedet for RK4 sine (`error_toka_RK4_T1` …). Nå brukes `err_RK4_tok_B[k]`.
- **Task C:** `t_eval_c_boris = np.arange(1, n_steps_C + 1)*h_C` i stedet for `np.linspace(0, T_C, n_steps_C)`, fordi `boris_method` lagrer posisjonene ved $t_1, \dots, t_N$.

## Figurnummer
Alle 26 figurer har fått «Figure N:» foran tittelen (`fig.suptitle` eller `plt.title`), nummerert i rekkefølge.

## Testing
Både den gamle (`main`) og den nye notebooken er kjørt med kortere tider. Alle baner, energier og alle resultater i task B og C er identiske, eller avviker bare med avrundingsfeil under $10^{-16}$ relativt. De eneste forskjellene er feilene i task A for Boris og RK4, som skyldes den nye referanseløsningen. Notebooken er ikke kjørt med fulle tider, så det meste av outputet er fra før. Cellene som er skrevet om eller lagt til, har ikke output før dere kjører dem.

## Impulscellen (ikke endret, se forslag i chatten)
Bare variabelnavnene er oppdatert. Logikken er som før, og cellen gir fortsatt feil.
