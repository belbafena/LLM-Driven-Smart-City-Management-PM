# LLM-Driven Decision Support System for Integrated Smart City Management

## Overview

This project presents an intelligent decision support system that integrates predictive modeling, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs) to support urban traffic management decisions. Traffic congestion prediction is used as a case study to demonstrate the proposed framework.

## Case Study

Traffic Congestion Prediction

## Technologies

- SAS Viya
- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- GPT-4o-mini

## Features

- Multi-source traffic and weather data integration
- Traffic congestion prediction using machine learning models
- RAG-based contextual knowledge retrieval
- LLM-generated explanations and recommendations
- Interactive decision support dashboard

## Repository Contents

- `data_cleaning.sas` – SAS preprocessing and feature engineering workflow
- `cleaned_dataset.csv` – Integrated and processed dataset
- `dashboard.py` – Streamlit dashboard interface
- `main.py` – Application entry point
- `rag_module.py` – RAG retrieval component
- `llm_module.py` – LLM explanation and recommendation component
- `rules.json`, `cases.json`, `strategies.json` – Knowledge base resources
