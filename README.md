This repository contains the source code for an MLP and KAN-based decoder for BMOCZ. Existing models can be found for both KANs and MLPs when K = 4 bits or K = 6 bits. The train files contain ways to simulate BLER and BER; furthermore, there is a DiZeT simulator included so that the BLER curve can be simulated under flat-fading and AWGN channels.

Main Contributions:

Anthony Perre: 
(1) Build BLER simulator for DiZeT Decoder; furthermore, implemented DiZeT decoder in PyTorch for BLER comparisons. | (2) Implemented BMOCZ Encoder in Pytorch. | (3) Built main training loop & BLER simulation for MLP-based models. | (4) Trained MLP-based decoder for K = 4 and K = 6. | (5) Ran BLER simulations in flat-fading & AWGN channels for trained MLP models.

Jack Hyatt
(1) Trained KAN-based decoders for K = 4 and K = 6; specifically, altered BMOCZ file to support KAN instead of MLP. | (2) Ran BLER simulation for KAN models under flat-fading & AWGN channels. | (3) Helped to improve the performance by adjusting hyperparameters and optimizing the training loop.
