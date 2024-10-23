
# Activity guided Diffusion for Sensor Data Generation
<!-- 
[![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://link-to-documentation) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/username/projectname)](https://img.shields.io/github/v/release/username/projectname?style=flat-square)

## Description
This project aims to [provide a brief overview of what the project does, its goals, or core functionalities]. If you're unsure about the specifics yet, mention that it's in active development with planned features.

- [Documentation](https://link-to-documentation)

## Installation

To get started, follow the installation steps:

1. Add the dependency to your `config.yml` or similar file:

```yaml
dependencies:
  project_name:
    github: username/project_name
```

2. Run the following command to install dependencies:

```bash
install-command
```

## Usage

Here's a basic example of how to use this project. Adapt the following code snippet to your needs:

```python
# Example usage of the project
import project_name

# Initialize the module
module = project_name.Module()

# Example function
result = module.some_function(data)

# Output result
print(result)
```

Add more detailed usage examples and explanations of the available API functions or methods.

## Links

- [Documentation](https://link-to-documentation)
- [Related Project](https://link-to-related-project) -->

<!-- ## License

This project is licensed under the MIT License - see the LICENSE file for details.

You can now add the actual project name, documentation links, and usage details. Let me know if you need further assistance! -->

# Public dataset -- ETTh1

| Model name       | Context length    | MSE            | Description                           |
| ---------------- | ----------------- | -------------- | --------------------------------------|
| TimeDiff         | 1440              | 0.407          | default from the paper                |
| TimeDiff         | 200               | 0.45           | Set the parameter into much shorter   |
| Ours             | 1440              | /              | need to change the model structure    |
| Ours             | 200               | 0.472          | temporal result                       |



# Sensor dataset -- Forecasting with TimeDiff

| Model name       | Context length    | MSE            | Description                           |
| ---------------- | ----------------- | -------------- | --------------------------------------|
| TimeDiff         | 200               | 0.927          | Result on Validation set              |
| Ours             | 200               | 1.0583         | Result on Validation set              |


# Sensor dataset -- Generating with Diffusion-TS
<!-- 
| **Metric**              |    **Models**    | **walking forward** | **running forward** | **jumping**       | **sleeping**      |
|-------------------------|------------------|---------------------|---------------------|-------------------|-------------------|
| **Context-FID Score**   | **Diffusion-TS** | 0.253 ± 0.044       | 0.321 ± 0.070       |  0.611 ± 0.123    | 0.023 ± 0.010     |
| (Lower the Better)      | **Ours**         | **0.193 ± 0.045**   | **0.297 ± 0.072**   | **0.340 ± 0.030** | **0.008 ± 0.003** |

| **Metric**              |    **Models**    | **walking forward** | **running forward** | **jumping**       | **sleeping**      |
|-------------------------|------------------|---------------------|---------------------|-------------------|-------------------|
| **Correlational Score** | **Diffusion-TS** | **0.071 ± 0.017**   | **0.060 ± 0.002**   | 0.075 ± 0.019     | 0.210 ± 0.028     |
| (Lower the Better)      | **Ours**         | 0.115 ± 0.003       | 0.179 ± 0.014       | **0.074 ± 0.004** | **0.108 ± 0.013** |

| **Metric**              |    **Models**  | **walking forward** | **running forward** | **jumping**   | **sleeping**  |
|-------------------------|----------------|---------------------|---------------------|---------------|---------------|
| **Discriminative Score**| Diffusion-TS   | **0.006**           | 0.147               | 0.116         | 0.116         |
| (Lower the Better)      | Ours           | 0.101               | 0.103               | 0.300         | 0.116         |

| **Metric**              |    **Models**  | **walking forward** | **running forward** | **jumping**   | **sleeping**  |
|-------------------------|----------------|---------------------|---------------------|---------------|---------------|
| **Predictive Score**    | Diffusion-TS   | **0.006**           | 0.147               | 0.116         | 0.116         |
| (Lower the Better)      | Ours           | 0.101               | 0.103               | 0.300         | 0.116         | -->


|                         | **SensorDiff** | | | | **Diffusion-Ts** | | | |
|-------------------------|----------|-|-|-|------------------|-|-|-|
|                         | Context<br>FID | Correlation<br>Score |  Discriminative<br>Score | Predictive<br>Score |Context<br>FID | Correlation<br>Score | Discriminative<br>Score | Predictive<br>Score |
| **elevatordown**        | 0.012 ± .004 | 0.107 ± .026 | 0.013 ± .011 | 0.017 ± .001 | 0.014 ± .004 | 0.230 ± .005 | 0.011 ± .013 | 0.017 ± .001 |
| **elevatorup**          | 0.005 ± .001 | 0.056 ± .013 | 0.007 ± .008 | 0.014 ± .000 | 0.013 ± .004 | 0.248 ± .103 |0.003 ± .005 | 0.015 ± .001 |
| **jumping**             | 0.124 ± .021 | 0.058 ± .006 | 0.056 ± .023 | 0.031 ± .000 | 0.191 ± .037 | 0.146 ± .022 |0.025 ± .013 | 0.031 ± .001 |
| **runningforward**      | 0.118 ± .023 | 0.048 ± .012 | 0.145 ± .011 | 0.040 ± .001 | 0.037 ± .003 | 0.182 ± .032 |0.000 ± .000 | 0.038 ± .002 |
| **sitting**             | 0.003 ± .001 | 0.060 ± .011 | 0.054 ± .045 | 0.013 ± .000 | 0.038 ± .012 | 0.248 ± .030 |0.000 ± .000 | 0.012 ± .000 |
| **sleeping**            | 0.001 ± .000 | 0.052 ± .010 | 0.005 ± .008 | 0.012 ± .000 | 0.006 ± .001 | 0.288 ± .091 |0.003 ± .008 | 0.012 ± .001 |
| **standing**            | 0.024 ± .001 | 0.069 ± .016 | 0.024 ± .003 | 0.017 ± .000 | 0.024 ± .007 | 0.227 ± .052 |0.001 ± .003 | 0.016 ± .001 |
| **walkingdownstairs**   | 0.067 ± .017 | 0.059 ± .012 | 0.105 ± .007 | 0.039 ± .001 | 0.173 ± .024 | 0.173 ± .030 |0.025 ± .009 | 0.039 ± .002 |
| **walkingforward**      | 0.054 ± .012 | 0.137 ± .010 | 0.088 ± .005 | 0.033 ± .003 | 0.027 ± .006 | 0.010 ± .030 |0.000 ± .000 | 0.031 ± .003 |
| **walkingleft**         | 0.092 ± .015 | 0.050 ± .004 | / | / | 0.072 ± .005 | 0.177 ± .028 |0.000 ± .000 | 0.040 ± .002 |
| **walkingright**        | 0.069 ± .019 | 0.053 ± .009 | 0.043 ± .027 | 0.034 ± .002 | 0.047 ± .007 | 0.165 ± .051 |0.000 ± .000 | 0.032 ± .001 |
| **walkingupstairs**     | 0.068 ± .034 | 0.056 ± .004 | 0.118 ± .006 | 0.043 ± .001 | 0.052 ± .009 | 0.155 ± .023 |0.006 ± .006 | 0.042 ± .001 |
