import numpy as np
import argparse
try:
    import torch
except:
    print('Unable to import torch module')
import pickle
# import sentinel1decoder
import pandas as pd
from scipy.interpolate import interp1d
import math
from pathlib import Path 
import copy 
import gc
# check ram usage:
import psutil
import time

from os import environ
environ['OMP_NUM_THREADS'] = '8'

from SARLens.utils.io import dump

def timing_decorator(func):
    """
    A decorator to measure the execution time of a function and print the elapsed time.
    
    Parameters:
        func (callable): The function to measure.
        
    Returns:
        callable: The wrapped function with timing measurement.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

def printmemory():
    print(f'RAM memory usage: {psutil.virtual_memory().percent}%')
    return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_params(device=None, slant_range_vec=None, D=None, c=None, len_range_line=None, range_sample_freq=None, wavelength=None):
    params = {key: value for key, value in locals().items()}
    return params

class coarseRDA:

    def __init__(self, raw_data=None, verbose=False, backend='numpy'):
        ######### Settings Private Variables
        self._backend = backend
        self._verbose = verbose
        ######### Extraction of data
        self.radar_data = raw_data['echo']
        self.ephemeris = raw_data['ephemeris']
        self.metadata = raw_data['metadata']
        ######### Preliminary estimations
        self.len_range_line = self.radar_data.shape[1]
        self.len_az_line = self.radar_data.shape[0]
        if self._backend == 'torch':
            self.device = self.radar_data.device


    @timing_decorator
    def fft2D(self):
        # TODO: Test this function
        """
        Perform 2D FFT on a radar data array in range and azimuth dimensions.

        Args:
            raw_data (dict): Dict containing: 'echo':2D numpy array of radar data; 'metadata'; 'ephemeris';
            backend (str, optional): Backend to use for FFT. Defaults to 'numpy'.

        Returns:
            np.array: 2D numpy array of radar data after 2D FFT.
        """

        if self._backend == 'numpy':
            # FFT each range line
            self.radar_data = np.fft.fft(self.radar_data, axis=1)
            # FFT each azimuth line
            self.radar_data = np.fft.fftshift(np.fft.fft(self.radar_data, axis=0), axes=0)
            
        elif self._backend == 'torch':
            # Convert radar_data to a PyTorch tensor and move to device
            self.radar_data = torch.tensor(self.radar_data, dtype=torch.complex64, device=device)
            # FFT each range line
            self.radar_data = torch.fft.fft(self.radar_data, dim=1)
            # FFT each azimuth line
            self.radar_data = torch.fft.fftshift(torch.fft.fft(self.radar_data, dim=0), dim=0)
            
        elif self._backend == 'fftw':
            # Create FFTW plans
            a = pyfftw.empty_aligned(self.radar_data.shape, dtype='complex128')
            b = pyfftw.empty_aligned(self.radar_data.shape, dtype='complex128')
            fftw_plan_r = pyfftw.FFTW(a, b, axes=(1,), direction='FFTW_FORWARD', threads=4)
            fftw_plan_a = pyfftw.FFTW(b, a, axes=(0,), direction='FFTW_FORWARD', threads=4)

            # Copy data into the FFTW input array
            np.copyto(a, self.radar_data)

            # Perform FFT along the range dimension
            fftw_plan_r.execute()

            # Perform FFT along the azimuth dimension and apply FFT shift
            fftw_plan_a.execute()
            self.radar_data = fftshift(a, axes=0)
        else:
            raise ValueError('Backend not supported.')
        
        if self._verbose:
            print('- FFT performed successfully!')


    @timing_decorator
    def get_range_filter(self) -> np.ndarray:
        """
        Computes a range filter for radar data, specifically tailored to Sentinel-1 radar parameters.

        Notes:
            1. This function assumes that the Sentinel-1 specific constants are available through the 'sentinel1decoder.constants' module.
            2. The function makes use of the scipy `interp1d` function to interpolate spacecraft velocities.
            3. It assumes that the metadata DataFrame has specific columns like 'Range Decimation', 'PRI', 'Rank', and 'SWST'.

        """
        RGDEC = self.metadata["Range Decimation"].unique()[0]
        self.range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)

        # Nominal Replica Parameters
        TXPSF = self.metadata["Tx Pulse Start Frequency"].unique()[0]
        TXPRR = self.metadata["Tx Ramp Rate"].unique()[0]
        TXPL = self.metadata["Tx Pulse Length"].unique()[0]
        num_tx_vals = int(TXPL*self.range_sample_freq)
        tx_replica_time_vals = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals)
        phi1 = TXPSF + TXPRR*TXPL/2
        phi2 = TXPRR/2
        tx_replica = np.exp(2j * np.pi * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2))

        # Create range filter from replica pulse
        range_filter = np.zeros(self.len_range_line, dtype=complex)
        index_start = np.ceil((self.len_range_line-num_tx_vals)/2)-1
        index_end = num_tx_vals+np.ceil((self.len_range_line-num_tx_vals)/2)-2
        range_filter[int(index_start):int(index_end+1)] = tx_replica
        range_filter = np.conjugate(np.fft.fft(range_filter))
        return range_filter


    @timing_decorator
    def _compute_effective_velocities(self):
        # Tx pulse parameters
        self.c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
        self.PRI = self.metadata["PRI"].unique()[0]
        rank = self.metadata["Rank"].unique()[0]
        suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)
        range_start_time = self.metadata["SWST"].unique()[0] + suppressed_data_time
        

        # Sample rates along azimuth and range
        range_sample_period = 1/self.range_sample_freq
        self.az_sample_freq = 1 / self.PRI
        az_sample_period = self.PRI

        # Fast time vector - defines the time axis along the fast time direction
        sample_num_along_range_line = np.arange(0, self.len_range_line, 1)
        fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)

        # Slant range vector - defines R0, the range of closest approach, for each range cell
        self.slant_range_vec =((rank * self.PRI) + fast_time_vec) * self.c/2
        
        # Spacecraft velocity - numerical calculation of the effective spacecraft velocity
        ecef_vels = self.ephemeris.apply(lambda x: math.sqrt(x["X-axis velocity ECEF"]**2 + x["Y-axis velocity ECEF"]**2 +x["Z-axis velocity ECEF"]**2), axis=1)
        velocity_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(), fill_value="extrapolate")
        x_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(), self.ephemeris["X-axis position ECEF"].unique(), fill_value="extrapolate")
        y_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(), self.ephemeris["Y-axis position ECEF"].unique(), fill_value="extrapolate")
        z_interp = interp1d(self.ephemeris["POD Solution Data Timestamp"].unique(), self.ephemeris["Z-axis position ECEF"].unique(), fill_value="extrapolate")
        space_velocities = self.metadata.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

        x_positions = self.metadata.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
        y_positions = self.metadata.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
        z_positions = self.metadata.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

        position_array = np.transpose(np.vstack((x_positions, y_positions, z_positions)))

        a = sentinel1decoder.constants.WGS84_SEMI_MAJOR_AXIS_M
        b = sentinel1decoder.constants.WGS84_SEMI_MINOR_AXIS_M
        H = np.linalg.norm(position_array, axis=1)
        W = np.divide(space_velocities, H)
        lat = np.arctan(np.divide(position_array[:, 2], position_array[:, 0]))
        local_earth_rad = np.sqrt(
            np.divide(
                (np.square(a**2 * np.cos(lat)) + np.square(b**2 * np.sin(lat))),
                (np.square(a * np.cos(lat)) + np.square(b * np.sin(lat)))
            )
        )
        cos_beta = (np.divide(np.square(local_earth_rad) + np.square(H) - np.square(self.slant_range_vec[:, np.newaxis]) , 2 * local_earth_rad * H))
        ground_velocities = local_earth_rad * W * cos_beta

        self.effective_velocities = np.sqrt(space_velocities * ground_velocities)


    @timing_decorator
    def get_RCMC(self):
        """
        Calculate and return the RCMC filter for a given radar dataset.

        Args:
            metadata (pd.DataFrame): Pandas DataFrame containing metadata for the radar dataset.

        Returns:
            np.array: 1D numpy array representing the RCMC filter.
        """
        self._compute_effective_velocities()
        
        self.wavelength = sentinel1decoder.constants.TX_WAVELENGTH_M
        self.az_freq_vals = np.arange(-self.az_sample_freq/2, self.az_sample_freq/2, 1/(self.PRI*self.len_az_line))
        
        # Cosine of the instantanoeus squint angle
        self.D = np.sqrt(
            1 - np.divide(
                self.wavelength**2 * np.square(self.az_freq_vals),
                4 * np.square(self.effective_velocities)
            )
        ).T
        
        # Create RCMC filter
        range_freq_vals = np.linspace(-self.range_sample_freq/2, self.range_sample_freq/2, num=self.len_range_line)
        rcmc_shift = self.slant_range_vec[0] * (np.divide(1, self.D) - 1)
        rcmc_filter = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / self.c)
        return rcmc_filter
    
    
    @timing_decorator
    def ifft_rg(self):
        if self._backend == 'numpy':
            self.radar_data = np.fft.ifftshift(np.fft.ifft(self.radar_data, axis=1), axes=1)
    
    
    @timing_decorator
    def ifft_az(self):
        if self._backend == 'numpy':
            self.radar_data = np.fft.ifft(self.radar_data, axis=0)
    
    
    @timing_decorator 
    def get_azimuth_filter(self):
        az_filter = np.exp(4j * np.pi * self.slant_range_vec * self.D / self.wavelength)
        return az_filter
    
    
    @timing_decorator
    def data_focus(self):
        """
        Performs the focusing of the raw data.
        """
        self.fft2D()
        # RG compression
        self.radar_data = multiply(self.radar_data, 
                                        self.get_range_filter())

        # RCMC
        self.radar_data = multiply(self.radar_data, 
                                        self.get_RCMC())
        
        # IFFT Range #TODO: address this in a backend manner for supporting torch
        self.ifft_rg()
        
        # Az Compression
        self.radar_data = multiply(self.radar_data, 
                                    self.get_azimuth_filter())
        
        
        # IFFT #TODO: address this in a backend manner for supporting torch
        self.ifft_az()

    @timing_decorator
    def savefile(self, savepath):
        dump(self.radar_data, savepath)

def multiply(a, b):
    """
    Multiplies two arrays a and b using the specified backend.
    """
    return a * b
        # if not isinstance(a, np.ndarray):
        #     raise ValueError("Array 'a' must be a numpy ndarray")
        # if not isinstance(b, np.ndarray):
        #     raise ValueError("Array 'b' must be a numpy ndarray")
        # Ensure shapes are compatible for element-wise multiplication

        
        # elif backend == 'torch':
        #     if not isinstance(a, torch.Tensor):
        #         raise ValueError("Array 'a' must be a torch Tensor")
        #     if not isinstance(b, torch.Tensor):
        #         raise ValueError("Array 'b' must be a torch Tensor")
        #     # Ensure shapes are compatible for element-wise multiplication
        #     if a.shape != b.shape:
        #         raise ValueError("Shapes of arrays 'a' and 'b' must be the same")
        #     return torch.mul(a, b)
        # else:
        #     raise ValueError("Unsupported backend. Use 'numpy' or 'torch'.")




if __name__ == '__main__':
    pass
"""
    parser = argparse.ArgumentParser(description='SAR Processor')
    parser.add_argument('--data', type=str, default='radar_data.npy', help='path to the radar data')
    parser.add_argument('--meta', type=str, default='/path/to/ephemeris.pkl', help='Path to the ephemeris file')
    parser.add_argument('--ephemeris', type=str, default='radar_data.npy', help='path to the radar data')
    parser.add_argument('--output', type=str, default='outputdir', help='path to the focused radar data')
    parser.add_argument('--backend', type=str, default='numpy', help='backend used to process data')
    parser.add_argument('--num_chunks', type=int, default=15, help='Number of chunks to parse the SAR data')
    parser.add_argument('--idx_chunk', type=int, default=0, help='Index of the chunk to parse the SAR data')
    
    print('\n\n***   Starting SAR Processor   ***')
    args = parser.parse_args()
    # Load data:
    name = Path(args.data).stem
    idx = args.idx_chunk
    print(f'Processing chunk {idx+1}/{args.num_chunks}')
    printmemory()
    radar_data, meta, ephemeris = get_partition(data_path=args.data, ephem_path=args.ephemeris, meta_path=args.meta, num_chunks = args.num_chunks, idx_chunk=idx)

"""