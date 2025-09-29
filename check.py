import torch
import qiskit
import qiskit_aer

# Check PyTorch and CUDA
cuda_available = torch.cuda.is_available()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {cuda_available}")
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Check Qiskit
print(f"\nQiskit version: {qiskit.__version__}")
#print(f"Qiskit Aer simulators: {[backend.name() for backend in qiskit_aer.Aer.backends()]}")
print(f"Qiskit Aer simulators: {[backend.name for backend in qiskit_aer.Aer.backends()]}")

# Check if Aer can see the GPU
aer_gpu_sim = qiskit_aer.AerSimulator(device='GPU')
print("\nSuccessfully initialized Qiskit Aer GPU simulator.")

# Thoát khỏi Python
#exit()
