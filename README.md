# Semi-dynamic traffic assignment with residual demand for SimCenter R2D

![Bay Area Congestion](bayarea_congestion.jpg)

### Features
* Quasi-equilibrium traffic assignment
* Efficient routing for millions of trips using [contraction hierarchy](https://github.com/UDST/pandana/blob/dev/examples/shortest_path_example.py) and priority-queue based Dijkstra algorithm [sp](https://github.com/cb-cities/sp)
* Temporal dynamics with residual demand, with time step of a few minutes
* Compatible with road network retrieved from [OSMnx](https://github.com/gboeing/osmnx)

### Use cases
* Calculating network traffic flow for small and large road networks (10 to 1,000,000 links) at sub-hourly time steps
* Visualizing traffic congestion dynamics throughout the day
* Analyzing traffic-induced carbon emissions (emission factor model)
* Assessing regional mobility and resilience with disruptions (e.g., road closure, seismic damages) 