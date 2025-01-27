# Roadmap for Template NN

- [Roadmap](#roadmap-for-template-nn)
    - [Vision](#vision)
    - [Development](#development)
        - [Versioning](#versioning)
        - [Content](#content)
    - [Milestones](#milestones)
        - [Model Templates](#model-templates)
        - [Optimisers](#optimisers)
        - [Integration](#integration)
        - [Hybridisation](#hybridisation)

## Vision

The vision of Template NN is to provide an intuitive yet flexible framework that accelerates the prototyping of various
neural networks, enabling developers to focus on innovation rather than the intricacies of model implementation.

Template NN was created to address the challenges of quickly iterating on different neural network models without being
hindered by repetitive setup tasks. It now aims to offer the same flexibility and efficiency to the broader machine
learning community.

## Development

This project follows a release cycle structured around "big, medium, and small" milestones. For versioning, we define
the following pattern for the 0.1.x series:

### Versioning

- **Big releases**: Occur when `x % 3 == 0` (e.g., 0.1.0, 0.1.3, 0.1.6, ...)
- **Medium releases**: Occur when `x % 3 == 1` (e.g., 0.1.1, 0.1.4, 0.1.7, ...)
- **Small releases**: Occur when `x % 3 == 2` (e.g., 0.1.2, 0.1.5, 0.1.8, ...)

### Content

- A *Big* release typically introduces new features being added (e.g. a new neural network architecture template or a
  new optimiser).
- A *Medium* release builds on the previous one by adding related features (e.g. extra arguments or new ways to use
  existing features).
- A *Small* release focuses on improving the project through tasks like writing tests, documentation updates, code refactoring, or
  small improvements without adding significant new functionality.

## Milestones

### Model Templates

This milestone focuses on creating a set of flexible and reusable templates for various neural network architectures.
The goal is to make it easy for users to quickly prototype different models without starting from scratch.

**Planned Models**

| Model Name                 | Description                                                            | Status    |
|----------------------------|------------------------------------------------------------------------|-----------|
| **Feedforward NN (FNN)**   | A basic fully connected neural network for general purposes.           | Completed |
| **Convolutional NN (CNN)** | A neural network designed for image processing and feature extraction. | Planned   |
| **Recurrent NN (RNN)**     | A network suited for sequence-based tasks.                             | Planned   |
| **Transformers**           | A model architecture based on attention mechanisms.                    | Planned   |
| **GANs**                   | A generative adversarial network for data generation tasks.            | Planned   |

### Optimisers

This milestone will develop custom optimisers that are compatible with the model templates. The aim is to provide more
control over training by incorporating various optimisation techniques that are not available in standard PyTorch
libraries.

**Planned Optimisers**

| Optimiser Name                         | Description                                                                                                                                                                     | Status  |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| **Ant Colony Optimisation (ACO)**      | Based on the foraging behavior of ants, this optimiser uses a population of solutions to iteratively improve results through pheromone-based communication.                     | Planned |
| **Artificial Bee Colony (ABC)**        | Modeled after the foraging behavior of honeybees, this optimiser simulates the search for food sources to find optimal solutions.                                               | Planned |
| **Cuckoo Search**                      | Inspired by the parasitic behavior of cuckoo birds, where the optimiser exploits randomness and greediness to find optimal solutions.                                           | Planned |
| **Genetic Algorithm (GA)**             | Mimics the process of natural selection, using crossover and mutation to evolve better solutions over generations.                                                              | Planned |
| **Grey Wolf Optimisation (GWO)**       | Inspired by the hunting behavior of grey wolves, this optimiser simulates the social hierarchy and leadership dynamics within a pack to guide the search for optimal solutions. | Planned |
| **Harris Hawk Optimiser (HHO)**        | Mimics the cooperative hunting behavior of Harris hawks, where individuals switch between exploration and exploitation during the search process.                               | Planned |
| **Particle Swarm Optimisation (PSO)**  | Inspired by the social behavior of birds flocking or fish schooling, this optimiser uses a population of candidate solutions to explore the search space.                       | Planned |
| **Whale Optimisation Algorithm (WOA)** | Based on the bubble-net hunting strategy of humpback whales, this optimiser uses exploration and exploitation strategies to find global optima.                                 | Planned |

### Integration

This milestone will focus on integrating the custom optimisers with the model templates, ensuring smooth
interoperability between the two. The goal is to allow users to seamlessly combine different models and optimisers for
more efficient experimentation.

### Hybridisation

The hybridisation milestone aims to explore and implement hybrid models that combine different types of neural networks.
The goal is to enable more advanced use cases by allowing users to easily integrate and experiment with various model
architectures.