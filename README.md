# 🤖 Automate PyTorch Model Training with Tekton and Paketo Buildpacks

This repository demonstrates building a CI/CD pipeline to fine-tune a GPT-2 model using Tekton pipelines and Paketo Buildpacks. It leverages Buildpacks for containerization—eliminating the need to manually write and maintain Dockerfiles—and uses Tekton to orchestrate the entire build and training workflow on Kubernetes.

## Why Tekton and Buildpacks?

- **[Tekton Pipelines](https://tekton.dev/)** is an open-source CI/CD system native to Kubernetes, enabling reproducible and scalable workflows.
- **[Paketo Buildpacks](https://paketo.io/)** automatically containerizes your applications by detecting runtimes and dependencies, simplifying container image creation without Dockerfiles.
- **[GPT-2](https://github.com/openai/gpt-2)** is a lightweight language model ideal for quick demonstration purposes.

## Repository Structure

- `training_process/`
  - `train.py` – GPT-2 training script.
  - `requirements.txt` – Dependencies automatically managed by Buildpacks.
  - `train.txt` – Small Q&A training dataset.
  - `serve.py` – Script to run the fine-tuned model.
- `untrained_model.py` – Script to demonstrate GPT-2 before training.
- Tekton and Kubernetes configurations:
  - `model-training-pipeline.yaml` – Tekton pipeline definition.
  - `source-pv-pvc.yaml` – PersistentVolume and PersistentVolumeClaim.
  - `kind-config.yaml` – Kind cluster setup for local testing.
  - `sa.yml` – ServiceAccount configuration for Docker registry authentication.

## Benefits

- **No Dockerfiles:** Automatic containerization and dependency management.
- **Consistency:** Reproducible builds and training across environments.
- **Efficiency:** Simple maintenance and easy updates.

By leveraging Tekton and Paketo Buildpacks, this pipeline streamlines ML model development and deployment, fostering rapid and reliable experimentation.

