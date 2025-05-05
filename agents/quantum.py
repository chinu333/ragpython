from azure.quantum import Workspace
from azure.quantum.cirq import AzureQuantumService
import cirq


workspace = Workspace ( 
  resource_id = "/subscriptions/dfbce8c6-7e40-4c00-b0c2-7542f1dd2814/resourceGroups/atlmtcquamtum/providers/Microsoft.Quantum/Workspaces/atlquantum", # Add your resource_id 
  location = "eastus"  # Add your workspace location (for example, "westus") 
)

service = AzureQuantumService(workspace)

print("This workspace's targets:")
for target in service.targets():
    print("-", target)

q0 = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.H(q0),               # Apply an H-gate to q0
    cirq.measure(q0)          # Measure q0
)
circuit
print("Circuit:", circuit)

# q0, q1 = cirq.LineQubit.range(2)
# circuit = cirq.Circuit(
#     cirq.H(q0),
#     cirq.H(q1),               # Apply an H-gate to q0
#     cirq.measure(q0, q1)          # Measure q0
# )
# circuit
# print("Circuit:\n", circuit)

# q0, q1 = cirq.LineQubit.range(2)
# circuit = cirq.Circuit(
#     cirq.X(q0)**0.5,             # Square root of X
#     cirq.CX(q0, q1),              # CNOT
#     cirq.measure(q0, q1, key='b') # Measure both qubits
# )
# print("Circuit:\n", circuit)

def quantum_simulator():

    # Using the IonQ simulator target, call "run" to submit the job. We'll
    # use 100 repetitions (simulated runs).
    job = service.targets("ionq.simulator").submit(circuit, name="cirq-ionq-simulator", repetitions=100)

    # Print the job ID.
    # print("Job id:", job.job_id())

    # Await job results.
    print("Awaiting job results...")
    result = job.results()
    job_id = job.job_id()
    print("Job Finished. IONQ SIMULATOR Result ::\n", result)
    print("Job Finished. IONQ JOB ID ::\n", job_id)

def submit_quantum_job(repetitions_count):

    # To view the probabilities computed for each Qubit state, you can print the result.

    result = service.run(
        program=circuit,
        repetitions=repetitions_count,
        target="ionq.simulator",
        timeout_seconds=500 # Set timeout to accommodate queue time on QPU
    )

    
    resultStr = str(result)
    resultStr = resultStr.split('=')[1]
    charzero = resultStr.count("0")
    charone = resultStr.count("1")

    print("Job Finished. IONQ Result ::\n", result)
    print("Character 0 Count ::::  ", charzero)
    print("Character 1 Count ::::  ", charone)
    # print("Measurement DataFrame:\n", measurement_dataframe)

    quantum_response = {
        "result": result,
        "zero": charzero,
        "one": charone,
        "circuit": str(circuit)
    }
    return quantum_response

# quantum_simulator()
# print(submit_quantum_job(80))