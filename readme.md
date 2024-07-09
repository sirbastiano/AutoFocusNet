

# SARLens

Welcome to the SARLens repository. This project focuses on Sentinel-1 Level-0 (L0) products.

## Features

- Decoding: [Uses s1isp for decoding S-1 L0]
- Focusing: [Uses a coarse Range-Doppler Algorithm for focusing]
- Plotting: [Uses cdf for estimating the plotting params]

## Installation

To get started with the project, follow these installation steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/sirbastiano/AutoFocusNet.git
   cd AutoFocusNet
   ```



2. **Create a new conda environment:**

   ```sh
   conda create -n sarlib python=3.9
   conda activate sarlib
   ```

3. **Install the project dependencies:**

   - Install pytorch for your system.

   - Clone & Install the dependencies including the package from avalentino for the s1isp decoder:
   
   ```sh
   python3 -m pip install --editable .
   ```

   

## Usage

Provide examples or steps to use the project. Include code snippets if necessary.

### * Decoding

```python
input_file = '/Data_large/marine/PythonProjects/AutoFocusNet/Data/SANPAOLO/S1A_S3_RAW__0SDH_20240524T213606_20240524T213631_054018_069139_241A.SAFE/s1a-s3-raw-s-hh-20240524t213606-20240524t213631-054018-069139.dat'
output_folder = '/Data_large/marine/PythonProjects/AutoFocusNet/Data/Decoded/SANPAOLO'
!python -m SARLens.processor.decode -i {input_file} -o {output_folder}
```

### * Focusing

```python
from SARLens.processor.focus import coarseRDA
from SARLens.utils.io import load, plot_with_cdf
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


radar_data = load(f'{your_data_path}/s1a-s3-raw-s-hh-20240524t213606-20240524t213631-054018-069139_pkt_0.pkl')
metadata = load(f'{your_data_path}/s1a-s3-raw-s-hh-20240524t213606-20240524t213631-054018-069139_pkt_0_metadata.pkl')
ephemeris = load(f'{your_data_path}/s1a-s3-raw-s-hh-20240524t213606-20240524t213631-054018-069139_ephemeris.pkl')

# init
radar_data = torch.from_numpy(radar_data).to(device)

RadarProcessor = coarseRDA(
    raw_data={'echo':radar_data[:,:], 'ephemeris':ephemeris, 'metadata':metadata}, 
    verbose=False, 
    backend='torch'
)

RadarProcessor.data_focus()
plot_with_cdf(RadarProcessor.radar_data, (24,24))
```

## Requirements

The following Python packages are required and will be installed automatically:

- `asf_search==7.1.2`
- `pandas==2.2.2`
- `rasterio==1.2.10`
- `torch==2.2.0`
- `s1isp`
- `matplotlib`
- `futures`

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to open an issue or contact us at [roberto.delprete@unina.it].

