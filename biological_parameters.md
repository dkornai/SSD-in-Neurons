# Literature estimates of biological parameters from the model

### 1) Mitochondrial death rate in neurons (mu)

**Reported values for half life:**

- 20-25 days (rat) ([The relationship of alpha-synuclein to mitochondrial dynamics and quality control](https://doi.org/10.3389/fnmol.2022.947191))

- 7-28 days (rat) ([Mitochondria and ageing: winning and losing in the numbers game](https://doi.org/10.1002/bies.20634))

- 0.25 days (mice) ([Longitudinal tracking of neuronal mitochondria delineates PINK1/Parkin-dependent mechanisms of mitochondrial recycling and degradation](https://doi.org/10.1126/sciadv.abf6580))

- 1.5 days (flies) ([Modeling mitochondrial dynamics during in vivo axonal elongation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022519308004724?casa_token=rPmBGT5oZ3MAAAAA:pgTQ5LZelA085SZirV5r8i1VHXZUGCex1t-3iyHx8rMUPyqlIyGASAe_Y1IDSg-y3qAklEuuXg))

- 2 days (mouse axon terminal) ([Imaging axonal transport of mitochondria in vivo](https://doi.org/10.1038/nmeth1055))

----

The outermost region (0.25-28 days) is chosen, and converted to rates/day using ln(2)/half-life, with some approximate rounding

#### Corresponding death rates:

|                     | min        | max      |
| ------------------- | ---------- | -------- |
| **death rate (mu)** | 0.025 /day | 2.5 /day |

### 2) Axonal transport hopping rate (gammas)

**Reported values:**

- 0.1 μm/sec minimum, 0.5 μm/sec typical for anterograde, 0.25 μm/sec typical for retrograde ([Dynamics of Mitochondrial Transport in Axons](https://doi.org/10.3389/fncel.2016.00123))

- 1.02 μm/sec peak for anterograde, 1.41 μm/sec peak for retrograde, 87% immotile, 1.9/1 anterograde/retrograde flux ([Imaging axonal transport of mitochondria in vivo](https://doi.org/10.1038/nmeth1055))

- 74% immotile, 2.1/1 anterograde/retrograde flux ([Mitostasis in Neurons: Maintaining Mitochondria in an Extended Cellular Architecture](https://doi.org/10.1016/j.neuron.2017.09.055 "Persistent link using digital object identifier"))

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

A large proportion (~75-85%) are immotile, and motility is uneven between anterograde and retrograde populations with a relative flux of about 2/1. Thus overall, ~8% are motile in the retrograde direction, and ~16% are motile in the anterograde direction

#### Immotility adjusted Hopping rates for nodes separated by 100 μm:

|                       | minimum   | typical   | maximum    |
| --------------------- | --------- | --------- | ---------- |
| **gamma anterograde** | 13.8 /day | 69.1 /day | 194.9 /day |
| **gamma retrograde**  | 6.9 /day  | 17.3 /day | 70.5 /day  |

If the distance between nodes is larger or smaller, scale these values accordingly (e.g. For nodes separated by 500 μm, minimum value would be 3.46 /day )

### 3) birth rate ()

# mitochondrial distribution

| parameter            | ratio in soma | ratio in axon | ratio in dendrite |
| -------------------- | ------------- | ------------- | ----------------- |
| mitochondrial volume | 0.41          | 0.15          | 0.44              |
| cell volume          | 0.39          | 0.23          | 0.38              |

Source: [A Quantitative Study on the Distribution of Mitochondria in the Neuropil of the Juvenile Rat Somatosensory Cortex]("https://academic.oup.com/cercor/article/28/10/3673/5060262")

# soma outflow mitochondrial flux

Number of mt in soma: 

100 mt/μm^3 density of mt, 4% of soma volume is mitochondria ([3D neuronal mitochondrial morphology in axons, dendrites, and somata of the aging mouse hippocampus](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8423436/)) -> ~4mt/μm^3 of soma volume

approx. 7.5 μm diameter of soma -> ~7000 mt in the soma

outflow is 6mt/h -> ~8500 mt/day 

-> Outflow rate is ~1.2/day. If the soma is represented by multiple nodes, outflow rate must be adjusted upwards 

at the axonic terminal, a stable population of 350 mt is maintained with an inflow if 180 mt/day and an outflow of 90 mt/day ([Imaging axonal transport of mitochondria in vivo](https://doi.org/10.1038/nmeth1055))

# interplay of death rates and transport speeds in the non-terminal regions of the arbor

typical mt trafficking speed is 0.65 μm/sec -> ~56000 μm/day 

motile fraction is 0.25 -> effective averge speed is ~14000 μm/day

**Death rates at intermediate nodes should be representative of the amount of time it takes for mt to travel there**

e.g. death rate at a given node is death_rate*(edge_length/mt_transport_speed)
