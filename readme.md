

# SARLens

Welcome to the Project Name repository. This project focuses on [brief description of the project and its purpose].

## Features

- Feature 1: [Description of feature 1]
- Feature 2: [Description of feature 2]
- Feature 3: [Description of feature 3]

## Installation

To get started with the project, follow these installation steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/sirbastiano/AutoFocusNet.git
   cd AutoFocusNet
   ```



2. **Create a new conda environment:**

   ```sh
   conda create -n autofocus python=3.11
   conda activate autofocus
   ```

3. **Install the project dependencies:**

   Clone & Install the repo from avalentino for the s1isp decoder:
   
   ```sh
   git clone https://github.com/sirbastiano/s1isp.git
   cd s1isp 
   python3 -m pip install --editable .
   cd ..
   pip install . -e
   ```

## Usage

Provide examples or steps to use the project. Include code snippets if necessary.

```python
# Example usage
from SARLens import main_function

result = main_function(input_data)
print(result)
```

## Requirements

The following Python packages are required and will be installed automatically:

- `asf_search==7.1.2`
- `pandas==2.2.2`
- `rasterio==1.2.10`
- `torch==2.2.0`

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to open an issue or contact us at [roberto.delprete@unina.it].

