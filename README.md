# Semi-dynamic traffic assignment with residual demand for SimCenter R2D

![Bay Area Congestion](/images/alameda_congestion.png)

### Features
* Quasi-equilibrium traffic assignment
* Efficient routing for millions of trips using [contraction hierarchy](https://github.com/UDST/pandana/blob/dev/examples/shortest_path_example.py) and priority-queue based Dijkstra algorithm [sp](https://github.com/cb-cities/sp)
* Temporal dynamics with residual demand, with time step of a few minutes
* Compatible with road network retrieved from [OSMnx](https://github.com/gboeing/osmnx)

### Use cases
* Calculating network traffic flow for small and large road networks (10 to 1,000,000 links) at sub-hourly time steps
* Visualizing traffic congestion dynamics throughout the day
* Assessing regional mobility and resilience with disruptions (e.g., road closure, seismic damages)

### Input Data
* Road Network
  * nodes.csv
  * edges.csv
  * closed_edges.csv (designed for disruptive events)
* Demand
*   OD.csv

### Output Data
* Travel time
* Travel route
