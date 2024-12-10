This repository contains the source code for an MLP and KAN-based decoder for BMOCZ. Existing models can be found for both KANs and MLPs when K = 4 bits or K = 6 bits. The train files contain ways to simulate BLER and BER; furthermore, there is a DiZeT simulator included so that the BLER curve can be simulated under flat-fading and AWGN channels.

Main Contributions:

Anthony Perre: 
(1) Build simulator for DiZeT Decoder
(2) Built training loop & BLER simulator for each MLP model
(3) Trained MLP-based decoder for K = 4 and K = 6
(4) Ran BLER simulation for MLP models

Jack Hyatt
(1) Trained & built KAN-based decoder for K = 4 and K = 6
(2) Ran BLER simulation for KAN models
(3) Helped tune the traininging loop algorithm to improve performance.
