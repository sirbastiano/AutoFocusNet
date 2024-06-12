import sentinel1decoder
import pandas as pd
import numpy as np
from pathlib import Path, os
import argparse
import pickle


def picklesavefile(path, datafile):
    with open(path, 'wb') as f:
        pickle.dump(datafile, f)

def split_radar_data(radar_data, L0_name, output_folder, num_chunks: int = 10):
    """
    Split radar data into chunks to avoid memory overload and then save the chunks as numpy arrays
    """
    try:
        n = len(radar_data)
        chunk_size = n // num_chunks
        for i in range(num_chunks):
            chunk = radar_data[i*chunk_size:(i+1)*chunk_size]
            outpath = os.path.join(output_folder, f'{L0_name}_chunk_{i+1}.pkl')
            picklesaver(outpath, chunk)
        return None
    except:
        print('Error splitting radar data. Retrying...')
        n = len(radar_data)
        chunk_size = n // num_chunks
        for i in range(num_chunks):
            chunk = radar_data[i*chunk_size:(i+1)*chunk_size]
            outpath = os.path.join(output_folder, f'{L0_name}_chunk_{i+1}.pkl')
            picklesaver(outpath, chunk)
        return None

def decoder(inputfile, burst_n):
    l0file = sentinel1decoder.Level0File(inputfile)
    ephemeris = l0file.ephemeris
    metadata = l0file.get_burst_metadata(burst_n)
    radar_data = l0file.get_burst_data(burst_n)
    return {'echo':radar_data, 'metadata':metadata, 'ephemeris':ephemeris}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--inputfile', type=str, help='Path to input .dat file', default=None)
    parser.add_argument('-o', '--output', type=str, help='Path to folder output files', default='/Users/robertodelprete/Desktop/AutoFocusNet/DATA/DECODED')
    parser.add_argument('-p','--packet', type=int, help='Packet number', default=8)
    args = parser.parse_args()

    inputfile = args.inputfile
    output_folder = args.output
    
    if inputfile is not None:
        print('Decoding Level 0 file...')
        l0file = sentinel1decoder.Level0File(inputfile)
        ephemeris = l0file.ephemeris
        ephemeris.to_pickle(os.path.join(output_folder, Path(inputfile).stem + '_ephemeris.pkl'))
        metadata = l0file.get_burst_metadata(args.packet)
        metadata.to_pickle(os.path.join(output_folder, Path(inputfile).stem + f'_pkt_{args.packet}_metadata.pkl'))
        radar_data = l0file.get_burst_data(args.packet)
        L0_name = Path(inputfile).stem
        picklesavefile(path=f'{output_folder}/{L0_name}.pkl', datafile=radar_data)