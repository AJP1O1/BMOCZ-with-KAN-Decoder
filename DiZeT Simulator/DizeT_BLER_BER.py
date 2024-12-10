from DiZeT import Encoder, Decoder, Channel
import file_helper_functions as F
import torch.nn as nn
import numpy as np
import torch
import time
import os

@torch.no_grad()
def BLER(
    k: int,
    EbN0_range: list = [x for x in range(0, 10)],
    batch: int = 512,
    errors_needed: int = 100,
    fading_channel: bool = False,
    disable_gpu: bool = False,
):

    """
    Simulates the BLER for a given BMOXZ structure over the provided EbN0 range. The channel can 
    be chosen as AWGN or flat fading + AWGN depending on the desired simulation. The variable
    errors_needed can be raised to make the BLER curves finer if desired. Changing the batch 
    size does not affect the BLER results, but may make the simulation time slower or faster
    depending on the current hardware.

    Parameters:
    -----------
    k : int
        Number of input bits, where the message space is 2^k.

    hd : int
        Number of hidden layers in the decoder.

    learnable_radii : bool
        Specify if encoder radius is learnable.

    EbN0_range: list, default = [x for x in range(0, 10)]
        EbN0 range to test.

    batch : int, default = 512
        Batch size for testing.

    errors_needed : int, default = 100
        Number of block errors needed for a given EbN0 before moving to the next EbN0

    fading_channel: bool, default = False
        Specify if channel is flat fading
        
    disable_gpu : bool, default = False
        Disable GPU for testing.
    """

    PATH = os.path.dirname(os.path.abspath(__file__))

    if disable_gpu:
        testing_device = "cpu"
    else:
        testing_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    enc = Encoder(k, device = testing_device)
    ch = Channel(k, EsNo = 0, fading_channel = fading_channel, device = testing_device)
    dec = Decoder(k, return_bit_vector = False, device = testing_device)
    
    model = nn.Sequential(enc, ch, dec).to(device = testing_device)
    model.eval()

    ohe = torch.eye(2**k, 2**k, dtype = torch.float32).to(device = testing_device)
    ohe = torch.repeat_interleave(ohe.unsqueeze(0), batch, dim = 0).flatten(0, 1)
    messages = (enc.messages).to(device = testing_device)
    messages = torch.repeat_interleave(messages.unsqueeze(0), batch, dim = 0).flatten(0, 1)

    bler = {}

    for dB in EbN0_range:
        model[1].update_snr(dB)
        blocks_total = 0
        block_errors = 0
        while block_errors < errors_needed:
            o = model(messages)
            mhat = o.cpu().numpy()
            m = torch.argmax(ohe, dim = -1).cpu().numpy()
            block_errors += np.sum(mhat != m)
            blocks_total += batch*2**k
        print(f"Complete for {dB} dB", end = "\r")
        bler[dB] = block_errors/blocks_total

    with open(PATH + f'\\ber_summary.txt', 'w') as file:
        file.write(f"{bler.keys()}\n{bler.values()}")

    print(bler)
    return bler

@torch.no_grad()
def BER(
    k: int,
    EbN0_range: list = [x for x in range(0, 10)],
    batch: int = 512,
    errors_needed: int = 100,
    fading_channel: bool = False,
    disable_gpu: bool = False,
):

    PATH = os.path.dirname(os.path.abspath(__file__))

    if disable_gpu:
        testing_device = "cpu"
    else:
        testing_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    enc = Encoder(k, device = testing_device)
    ch = Channel(k, EsNo = 0, fading_channel = fading_channel, device = testing_device)
    dec = Decoder(k, device = testing_device)
    
    model = nn.Sequential(enc, ch, dec).to(device = testing_device)
    model.eval()

    ohe = torch.eye(2**k, 2**k, dtype = torch.float32).to(device = testing_device)
    ohe = torch.repeat_interleave(ohe.unsqueeze(0), batch, dim = 0).flatten(0, 1)
    messages = (enc.messages).to(device = testing_device)
    messages = torch.repeat_interleave(messages.unsqueeze(0), batch, dim = 0).flatten(0, 1)

    ber = {}

    for dB in EbN0_range:
        model[1].update_snr(dB)
        bits_total = 0
        bit_errors = 0
        while bit_errors < errors_needed:
            o = model(messages)
            mhat = torch.flatten(o).cpu().numpy()
            m = torch.argmax(ohe, dim = -1)
            m = torch.flatten(messages[m,:]).cpu().numpy()
            bit_errors += np.sum(mhat != m)
            bits_total += (batch*2**k)*k
        print(f"Complete for {dB} dB", end = "\r")
        ber[dB] = bit_errors/bits_total

    with open(PATH + f'\\bler_summary.txt', 'w') as file:
        file.write(f"{ber.keys()}\n{ber.values()}")

    print(ber)
    return ber

def main():
    bits = 6
    disable_gpu = False

    BLER(
        k = bits,
        EbN0_range = [x for x in range(0, 41, 2)],
        errors_needed = 2_500,
        fading_channel = True,
        disable_gpu = disable_gpu,
    )
     
if __name__ == "__main__":
    main()