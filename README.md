# AI-Powered Multi-Modal Insurance Claim Verification System
## Overview

This project implements a multi-modal damage claim verification system that analyzes image evidence, claim conversations, user history, and evidence requirements to determine whether a claim is supported by the submitted evidence.

The system processes claims related to three object categories:

* Car
* Laptop
* Package

For each claim, the solution identifies the visible issue, affected object part, severity level, evidence sufficiency, risk indicators, and overall claim status.

The final output is generated in the required CSV format.

---

## System Architecture

```
                Claims.csv
                     │
                     ▼
          Claim Extraction Module
                     │
                     ▼
      Image + Conversation + User History
                     │
                     ▼
     Vision-Language Model (via OpenRouter API)
                     │
                     ▼
      Evidence Verification Engine
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
 Risk Assessment             Severity Analysis
      │                             │
      └──────────────┬──────────────┘
                     ▼
        Structured Decision Generator
                     │
                     ▼
               dataset/output.csv
```
---

## Problem Statement

Modern insurance claim verification often involves multiple sources of information:

* User conversations
* Submitted images
* Historical user behavior
* Evidence requirements

The objective is to verify whether the submitted evidence actually supports the claim being made.

Images are treated as the primary source of truth, while conversations provide context and user history provides additional risk information.

---

## Solution Approach

The system follows a structured verification pipeline.

### 1. Claim Understanding

The claim conversation is analyzed to determine:

* Object being claimed
* Reported damage type
* Relevant object part
* Supporting contextual information

This step establishes what the system should verify in the submitted images.

---

### 2. Image Processing

Submitted images are loaded and analyzed using OpenRouter Vision Models capabilities.

The image analysis focuses on:

* Detecting visible damage
* Identifying object parts
* Estimating severity
* Determining whether evidence is sufficient

When multiple images are available, all images are considered together before making a decision.

---

### 3. Evidence Verification

The extracted claim information is compared against visual evidence.

The system determines whether:

* The claim is supported
* The claim is contradicted
* There is insufficient information

The evidence standard is evaluated according to the provided requirements.

---

### 4. Risk Assessment

Additional checks are performed for:

* Missing evidence
* Poor image quality
* Ambiguous claims
* Unsupported damage descriptions
* Verification uncertainty

Appropriate risk flags are generated whenever required.

---

### 5. Final Decision Generation

For every claim, the system generates:

* evidence_standard_met
* evidence_standard_met_reason
* risk_flags
* issue_type
* object_part
* claim_status
* claim_status_justification
* supporting_image_ids
* valid_image
* severity

The output strictly follows the required schema.

---

## Key Features

- Multi-modal claim verification
- Image-based damage assessment
- Evidence sufficiency evaluation
- Risk flag generation
- Severity estimation
- Structured claim justification
- Automated CSV output generation

---

## Project Structure

```text
├── code
│   ├── main.py
│   │
│   ├── verification
│   │   └── verify.py
│   │
│   └── evaluation
│       ├── evaluate.py
│       ├── eval_metrics.json
│       ├── eval_predictions.csv
│       └── evaluation_report.md
│
├── dataset
│   ├── claims.csv
│   ├── sample_claims.csv
│   ├── evidence_requirements.csv
│   ├── user_history.csv
│   ├── output.csv
│   └── images
│       ├── sample/
│       └── test/
│
├── .gitignore
├── README.md
└── requirements.txt
``` 
---

## Project Workflow

```text
Read CSV files
      ↓
Load claim
      ↓
Extract claim information
      ↓
Load evidence images
      ↓
Send prompt to OpenRouter Vision Model
      ↓
Parse structured response
      ↓
Validate evidence
      ↓
Generate prediction
      ↓
Write output.csv
      ↓
Run evaluation
```

---

### File Descriptions

#### main.py

Main entry point of the application.

Responsible for:

* Loading claims
* Running verification
* Generating predictions
* Writing output CSV

---

#### verification/verify.py

Contains the claim verification logic.

Responsible for:

* Image loading
* OpenRouter Vision Models integration
* Evidence assessment
* Decision generation

---

#### evaluation/evaluate.py

Evaluates predictions against the provided sample claims.

Generates:

* Evaluation report
* Accuracy metrics
* Prediction outputs

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

macOS / Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure OpenRouter Vision Models API

Create a `.env` file in the project root with a valid OpenRouter Vision Models API key from OpenRouter AI.

---

## Running Evaluation

To evaluate the verification pipeline using the provided sample dataset with reference outputs:

```bash
python code/main.py --evaluate
```

Generated files:

```text
code/evaluation/eval_predictions.csv
code/evaluation/eval_metrics.json
code/evaluation/evaluation_report.md
```

---

## Generating Predictions

To process claims and generate final predictions:

```bash
python code/main.py
```

The final results are written to:

```text
dataset/output.csv
```

---

## Output Fields

The generated CSV contains:

| Field                        | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| evidence_standard_met        | Whether evidence requirements are satisfied       |
| evidence_standard_met_reason | Reason for evidence decision                      |
| risk_flags                   | Any identified risks                              |
| issue_type                   | Detected damage type                              |
| object_part                  | Affected object component                         |
| claim_status                 | Supported / Contradicted / Not enough information |
| claim_status_justification   | Explanation of decision                           |
| supporting_image_ids         | Images used for decision                          |
| valid_image                  | Whether image evidence is usable                  |
| severity                     | Estimated severity level                          |

---

## Design Decisions

Several design choices were made while building the solution:

### Images as Primary Evidence

Visual evidence is treated as the primary source of truth.

Claims are not accepted solely based on conversation content.

---

### Conservative Decision Making

When evidence is unclear or insufficient, the system avoids unsupported conclusions and returns conservative outcomes.

---

### Multi-Image Support

Claims containing multiple images are analyzed collectively to improve decision quality.

---

### Separation of Components

The project separates:

* Verification logic
* Evaluation logic
* Execution pipeline

This makes debugging and testing easier.

---

## Technologies Used

### Programming
- Python

### AI & APIs
- OpenRouter API
- NVIDIA Vision Language Models
- Prompt Engineering

### Data Processing
- Pandas
- Pillow

### Development
- Git
- Python Dotenv

---

## Challenges Faced

Some challenges encountered during development included:

* Resolving image path issues
* Handling multiple image inputs
* Migrating model integration
* Managing API rate limits and quota restrictions
* Ensuring output schema consistency
* Generating structured justifications from image evidence

---

## Assumptions

The solution assumes:

* Images are the most reliable evidence source.
* User history provides risk context only.
* Missing evidence should not automatically support a claim.
* Multiple images belong to the same claim instance.
* Evidence-based decisions are preferred over speculative decisions.

---

## Key Learnings

During development this project provided experience with:

- Prompt engineering for multimodal LLMs
- Vision-language model integration
- Structured output generation
- AI pipeline design
- Evaluation-driven development
- API error handling
- Modular Python architecture

---

## Future Improvements

Potential future enhancements include:

* Local vision model fallback
* Confidence scoring
* Image authenticity verification
* Advanced fraud detection
* Ensemble model verification
* Improved severity estimation

---

## Author

Rakshat Tiwari