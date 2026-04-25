# Smart Irrigation System

## Setup
- create venv
- install requirements
- build processed dataset (local only)
- run API

## Run API
python api/app.py

## Endpoints
- GET /
- POST /predict

## Architecture
See docs/system_flow.md

## Dataset workflow (team standard)
- raw data location: `data/`
- processed output location: `data/processed/`
- build command: `python scripts/build_dataset.py --yes`
- validate command: `python -m scripts.validate_dataset`