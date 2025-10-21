# The-Tree-Mus-eers
Quantum Tic-Tac-Toe Classifier

A high-performance quantum machine learning project built with PennyLane and JAX to classify the final state of Tic-Tac-Toe boards (win, loss, or draw). This project is specifically optimized for the gpu4quantum challenge, aiming to maximize the use of GPUs for accelerating quantum circuit simulations.

Core Features

Advanced Quantum Circuit Design: Implements a hierarchical structure inspired by Quantum Convolutional Neural Networks (QCNNs) to effectively extract features from the game board.

High-Performance Backend: Utilizes the PennyLane-Lightning-GPU backend by default, leveraging NVIDIA's cuQuantum SDK for state-of-the-art simulation speed.

Deep JAX Integration:

Just-In-Time Compilation (@jax.jit): Compiles the entire computation graph, including loss and accuracy functions, into optimized GPU kernels.

Automatic Vectorization (jax.vmap): Achieves efficient data parallelism, allowing the entire batch of quantum circuit simulations to be processed on the GPU at once.

Advanced Training Strategies:

Handles imbalanced datasets through Oversampling and Class Weights.

Employs the Adam optimizer combined with a Learning Rate Warm-up and Cosine Decay schedule for more stable and efficient convergence.

Tech Stack

Quantum Computing Framework: PennyLane

High-Performance Backend: lightning.gpu / lightning.qubit

Numerical Computation & Acceleration: JAX

Optimizer: Optax

Core Library: NumPy

Installation and Setup

Clone the repository

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY


Create a virtual environment and install dependencies
It is recommended to use conda or venv to create a clean Python environment (Python 3.9+ is suggested).

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install requirements
The project dependencies are listed in requirements.txt.

pip install -r requirements.txt


Note: pennylane-lightning[gpu] requires a properly configured environment with NVIDIA drivers, CUDA Toolkit, and the cuQuantum SDK. If your environment does not meet these requirements, you can replace it with the CPU version by running pip install pennylane-lightning.

How to Run

Simply execute the main script to start data generation, model compilation, and training.

python quantum_tic_tac_toe_gpu_final_scheduled.py


Expected Output

After launching the script, you will see logs for environment detection, data preparation, and model compilation, followed by the training loop, which shows real-time progress for each batch and results for each epoch.

Successfully loaded lightning.gpu device, will use NVIDIA cuQuantum for acceleration.
Generating data...
Total samples: 6046
Class counts: {'draw': 4536, 'circle': 490, 'cross': 1020}
...
Training set size: 10887, Test set size: 1209
...
JIT compiling (first call will be slow)...
Compilation finished. Starting training.
Epoch 1/15 - T=150.21s - Loss=1.0524 - Test Acc=0.687
Epoch 2/15 - T=148.55s - Loss=0.8917 - Test Acc=0.751
...


License

This project is licensed under the MIT License.
