# Gemma2 End-to-End Fine-Tuning Pipeline

This project demonstrates a complete, automated, and modular pipeline for fine-tuning and deploying fine tuned Gemma2 models. It incorporates data ingestion, validation, transformation, model training, evaluation, and deployment using cloud and local compute options, all managed with GitHub Actions and Docker. The pipeline also integrates logging, visualization, and continuous deployment on Streamlit Cloud.

This project is designed in a way that it can be **scaled up or down**, can use **different LLM models**, based on custom input **choose models by accuracy and cost**. **Custom logs** helps to identify any problem occured during pipeline. This project can be improved by adding technologies such as **GKE**, **Grafana**.

[Access the Deployed Endpoint for Testing](https://www.genome.gov/)

## Pipeline Diagram
![alt text](https://github.com/OnurAsimIlhan/end-to-end-finetuning-gemma2/blob/main/diagram.png)

## Table of Contents
- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
  - [Data Ingestion](#data-ingestion)
  - [Data Validation](#data-validation)
  - [Data Transformation](#data-transformation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Deployment](#model-deployment)
- [Deployment Options](#deployment-options)
- [Logging and Monitoring](#logging-and-monitoring)
- [Configuration](#configuration)

## Overview
This project enables seamless data processing, model tuning, and deployment using Gemma2 with the following features:
- Pushes data to MongoDB Atlas and ingests it using custom configurations.
- Manages pipeline stages (data ingestion, validation, transformation, model training, and evaluation) with custom configurations and versioned artifacts.
- Pushes models to Hugging Face Hub upon evaluation and creates a Streamlit Cloud endpoint for deployment.
- Uses GitHub Actions for continuous integration, with an option for Dockerized cloud or local execution.
- Supports detailed logging stored in Logstash and visualized in Kibana.

## Pipeline Stages

### Data Ingestion
Data ingestion initiates by pushing raw data to MongoDB Atlas, followed by processing with a custom configuration that outputs an artifact for the next stage.

### Data Validation
The validation stage ensures data integrity, applying rules set in the custom configuration and generating a validation artifact.

### Data Transformation
Data transformation prepares the data for training. Using a configurable setup, this stage produces a transformation artifact with cleaned, processed data.

### Model Training
Model training uses the transformed data and a specific configuration to fine-tune the model. A model artifact is created at this stage.

### Model Evaluation
In the evaluation step, the model undergoes testing based on criteria set in the configuration file. Depending on the results, the new model is pushed to Hugging Face Hub for deployment.

### Model Deployment
The evaluated model is deployed on Streamlit Cloud, creating an accessible endpoint for users. Streamlit deployment provides interactive visualizations and model usage.

## Deployment Options
The pipeline is fully Dockerized, allowing easy deployment on both cloud and local systems. GitHub Actions orchestrate each stage, triggering automated runs and validations for every code update.

## Logging and Monitoring
Logging is handled through Logstash, with logs stored and visualized in Kibana, enabling real-time monitoring of pipeline performance and debugging information.

## Configuration
Each stage operates based on a configurable YAML file, allowing flexibility in pipeline settings, such as database connections, transformation rules, and model parameters. 
