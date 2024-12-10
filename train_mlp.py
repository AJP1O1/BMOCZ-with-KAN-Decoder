from BMOXZ_MLP import Encoder, Decoder, Channel
import file_helper_functions as F
import torch.nn as nn
import numpy as np
import torch
import time
import os

def train(
    k: int,
    hd: int,
    initial_snr: float,
    final_snr: float,
    epochs: int = 10000,
    batch: int = 1024,
    learning_rate: float = 1e-3,
    learnable_radii: bool = False,
    load_model: bool = False,
    disable_gpu: bool = False,
):

    """
    Trains an MLP-based neural network to detect the transmitted message. The transmitted 
    message is modulated using BMOXZ, where the radius for the inner and outer circles are
    learned via gradient descent; furthermore, a shuffle vector is learned that determines
    where each bit position is located on each circle. The model is trained under an AWGN
    channel, where the SNR is reduced from initial_snr to final_snr throughout training.
    The state dictionary for each model is saved to the same folder as the current python 
    script; additionally, a summary of the training parameters is saved in a text file.

    Parameters:
    -----------
    k : int
        Number of input bits, where the message space is 2^k.

    hd : int
        Number of hidden layers in the decoder.

    initial_snr : float
        Initial SNR (dB) for training.

    final_snr : float
        Final SNR (dB) for training.

    epochs : int, default = 10000
        Number of training iterations.

    batch : int, default = 1024
        Batch size for training.

    learning_rate : float, default = 1e-3
        Learning rate for the Adam optimizer.

    learnable_radii : bool, default = False
        Enable training of encoder radius.

    load_model : bool, default = False
        Load existing model

    disable_gpu : bool, default = False
        Disable GPU for training.
    """

    PATH = os.path.dirname(os.path.abspath(__file__))

    if disable_gpu:
        training_device = "cpu"
    else:
        training_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    enc = Encoder(k, learnable_radius = learnable_radii, device = training_device)
    ch = Channel(k, EsNo = initial_snr, fading_channel = False, device = training_device)
    dec = Decoder(k, hd, device = training_device)

    if load_model:
        try:
            enc.load_state_dict(F.load_model())
            dec.load_state_dict(F.load_model())
        except:
            return

    model = nn.Sequential(enc, ch, dec).to(device = training_device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss = nn.CrossEntropyLoss()

    ohe = torch.eye(2**k, 2**k, dtype = torch.float32).to(device = training_device)
    ohe = torch.repeat_interleave(ohe.unsqueeze(0), batch, dim = 0).flatten(0, 1)
    messages = (enc.messages).to(device = training_device)
    messages = torch.repeat_interleave(messages.unsqueeze(0), batch, dim = 0).flatten(0, 1)

    previous_time = time.time()
    elapsed_time = 0
    
    for i in range(epochs):
        
        optimizer.zero_grad()
        o = model(messages)
    
        ls = loss(o, ohe)
        ls.backward()
        optimizer.step()

        current_snr = initial_snr - i/epochs*(initial_snr - final_snr)
        model[1].update_snr(current_snr)

        if i % 50 == 0 and i != 0:
            
            m = torch.argmax(ohe, dim = -1).detach().cpu().numpy()
            mhat = torch.argmax(o, dim = -1).detach().cpu().numpy()
            bler = np.sum(mhat != m)/(batch*2**k)

            current_time = time.time()
            elapsed_time += current_time - previous_time
            previous_time = current_time

            print(
                f"""
                ============ Training Info ============
                Training Progress (%): {np.round(i/epochs*100, 3)}
                Current iteration (epoch): {i}
                Current SNR (dB): {np.round(current_snr, 3)}
                Time elapsed (s): {np.round(elapsed_time, 3)}
                Loss: {ls}
                BLER: {np.round(bler, 8)}""",
                end = '\033[F\033[F\033[F\033[F\033[F\033[F\033[F\r',
                flush = True,
            )
         
    print("\n\n\n\n\n\n\n\n\nTraining Complete!") 

    torch.save(model[0].state_dict(), PATH + f"\\bmoxz_encoder.pth")
    torch.save(model[2].state_dict(), PATH + f"\\bmoxz_decoder.pth")

    with open(PATH + f'\\model_summary.txt', 'w') as file:
        file.write(
            f"""
            k = {k}
            hd = {hd}
            initial_snr = {initial_snr}
            final_snr = {final_snr}
            epochs = {epochs}
            batch = {batch}
            learning_rate = {learning_rate}
            learnable_radii = {learnable_radii}
            radii = {(enc.R.data if learnable_radii else enc.R)}
            suffle_vector = {enc.shuffle_vector.data}
            """
        )

@torch.no_grad()
def BLER(
    k: int,
    hd: int,
    learnable_radii: bool,
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

    enc = Encoder(k, learnable_radius = learnable_radii, device = testing_device)
    ch = Channel(k, EsNo = 0, fading_channel = fading_channel, device = testing_device)
    dec = Decoder(k, hd, device = testing_device)

    enc.load_state_dict(F.load_model())
    dec.load_state_dict(F.load_model())
    
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
            mhat = torch.argmax(o, dim = -1).cpu().numpy()
            m = torch.argmax(ohe, dim = -1).cpu().numpy()
            block_errors += np.sum(mhat != m)
            blocks_total += batch*2**k
        print(f"Complete for {dB} dB", end = "\r")
        bler[dB] = block_errors/blocks_total

    with open(PATH + f'\\bler_summary.txt', 'w') as file:
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

    enc.load_state_dict(F.load_model())
    dec.load_state_dict(F.load_model())
    
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
    hidden = 150
    ep = 50_000
    b = 128
    lr = 5e-3
    learnable_radii = True
    disable_gpu = False

    mode = 1 # 0 for training, 1 for BLER simulation, 2 BER simulation

    if mode == 0:
        train(
            k = bits,
            hd = hidden,
            initial_snr = 15.5,
            final_snr = 10.5,
            epochs = ep,
            batch = b,
            learning_rate = lr,
            learnable_radii = learnable_radii,
            disable_gpu = disable_gpu,
        )
        
    elif mode == 1:
        BLER(
            k = bits,
            hd = hidden,
            learnable_radii = learnable_radii,
            EbN0_range = [x for x in range(0, 13, 1)],
            errors_needed = 1_500,
            fading_channel = False,
            disable_gpu = disable_gpu,
        )
        
    elif mode == 2:
        BER(
            k = bits,
            hd = hidden,
            learnable_radii = learnable_radii,
            EbN0_range = [x for x in range(0, 15)],
            errors_needed = 1_500,
            fading_channel = False,
            disable_gpu = disable_gpu,
        )
        return
        
if __name__ == "__main__":
    main()
