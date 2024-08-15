from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core import Experiment, ScriptRunConfig, Environment

ws = Workspace.from_config(path = './config.json')

compute_name = "gpu-cluster"
compute_min_nodes = 0
compute_max_nodes = 4
vm_size = "STANDARD_NC6"

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print(f"Found compute target: {compute_name}")
else:
    compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                              min_nodes=compute_min_nodes,
                                                           max_nodes=compute_max_nodes)
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)

compute_target.wait_for_completion(show_output=True)


experiment = Experiment(ws, "academic-risk-prediction")

# Define the environment (e.g., TensorFlow, Scikit-learn)
env = Environment.from_conda_specification(
    name="keras-env",
    file_path="environment.yml"
)

# Configure the training script to run on the compute target
src = ScriptRunConfig(source_directory="./", 
                      script="train.py",
                      compute_target=compute_target,
                      environment=env)

# Submit the experiment
run = experiment.submit(src)

# Monitor the run
run.wait_for_completion(show_output=True)
