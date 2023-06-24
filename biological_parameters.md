# Literature estimates of biological parameters from the model

### 1) Mitochondrial death rate in neurons (mu)

**Reported values for half life:**

- 20-25 days ([Frontiers | The relationship of alpha-synuclein to mitochondrial dynamics and quality control](https://doi.org/10.3389/fnmol.2022.947191))

- 7-28 days ([Mitochondria and ageing: winning and losing in the numbers game - Passos - 2007 - BioEssays - Wiley Online Library](https://doi.org/10.1002/bies.20634))

- 0.25 days ([DOI: 10.1126/sciadv.abf6580](https://doi.org/10.1126/sciadv.abf6580))

----

The outermost region (0.25-28 days) is chosen, and converted to rates/day using ln(2)/half-life, with some approximate rounding

#### Corresponding death rates:

|                     | min        | max      |
| ------------------- | ---------- | -------- |
| **death rate (mu)** | 0.025 /day | 2.5 /day |

### 2) Axonal transport hopping rate (gammas)

**Reported values:**

- 0.1 μm/sec minimum, 0.5 μm/sec typical for anterograde, 0.25 μm/sec typical for retrograde ([Frontiers | Dynamics of Mitochondrial Transport in Axons](https://doi.org/10.3389/fncel.2016.00123))

- 1.02 μm/sec peak for anterograde, 1.41 μm/sec peak for retrograde, 87% immotile (https://doi.org/10.1038/nmeth1055)

----

**Converted to μm /day, this is:**

|                   | minimum      | typical       | maximum        |
| ----------------- | ------------ | ------------- | -------------- |
| anterograde speed | 8640 μm /day | 43200 μm /day | 121824 μm /day |
| retrograde speed  | 8640 μm /day | 21600 μm /day | 88128 μm /day  |

The hopping rates corresponding to this speed depend on the total length of the axon, and the number of nodes used to represent the axon. the hopping rate is calculated as: (distance between nodes)/(speed).

**Hopping rates for nodes separated by 100 μm:**

|                       | minimum   | typical  | maximum   |
| --------------------- | --------- | -------- | --------- |
| **gamma anterograde** | 86.4 /day | 432 /day | 1218 /day |
| **gamma retrograde**  | 86.4 /day | 216 /day | 881 /day  |

Accounting for the fact that a large proportion (>80%) are immotile, the average speed is scaled accordingly.

#### Immotility adjusted Hopping rates for nodes separated by 100 μm:

|                       | minimum    | typical   | maximum    |
| --------------------- | ---------- | --------- | ---------- |
| **gamma anterograde** | 17.28 /day | 86.4 /day | 243.6 /day |
| **gamma retrograde**  | 17.28 /day | 43.2 /day | 176.2 /day |



If the distance between nodes is larger or smaller, scale these values accordingly (e.g. For nodes separated by 500 μm, minimum value would be 3.46 /day )

### 3) birth rate ()
