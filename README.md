A fork of [HOOMD-blue](https://glotzerlab.engin.umich.edu/hoomd-blue/) for shape-targeted alchemical Monte Carlo.

## Additional required submodule 
- [Clustering](https://github.com/Lallapallooza/clustering)

## Usage

Shape-targeted alchemical MC:
```python
import hoomd

# Create the initial state from a file
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
sim.create_state_from_gsd(filename='init.gsd')

# Set up an MC Integrator of convex polyhedra
mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0.1, default_a=0.1)
sim.operations.integrator = mc

# Set up the targeted shape move class
targeted_move=hoomd.hpmc.shape_move.Targeted1A5(target_1=0.6,
																								target_2=0.5)
shape_update=hoomd.hpmc.update.Shape(shape_move=targeted_move,
																		 trigger=hoomd.trigger.Periodic(10))
sim.operations.updates.append(shape_update)

# Run the simulation
sim.run(1e5)
```

## License

**HOOMD-blue** is available under the [3-clause BSD license](LICENSE).
