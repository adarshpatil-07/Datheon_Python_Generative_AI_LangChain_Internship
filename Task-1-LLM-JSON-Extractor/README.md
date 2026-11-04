# Structured LLM Prompt Pipeline & JSON Extractor

A Python-based invoice extraction system that uses Google's Gemini API to convert unstructured invoice text into structured, validated JSON.

Built as **Task 1** of the Datheon Python + Generative AI & LangChain Internship.

## Overview

This project demonstrates how Large Language Models (LLMs) can extract meaningful information from unstructured documents.

It processes invoice text using a parameterized prompt, generates structured JSON through Gemini, validates the response using Pydantic, and saves the result as a JSON file.

## Features

- Gemini API integration using the `google-genai` SDK.
- Parameterized prompt templates.
- Structured invoice information extraction.
- JSON output enforcement.
- Pydantic schema validation.
- Error handling for invalid input, missing files, API failures and validation errors.
- Unit tests for the extraction pipeline and validation logic.

## Technologies Used

- Python
- Google Gemini API
- Google GenAI SDK
- Pydantic
- python-dotenv
- Pytest

## Project Structure

```text
Task-1-LLM-JSON-Extractor/
│
├── prompts/
│   └── invoice_prompt.py
│
├── schemas/
│   └── invoice.py
│
├── sample_data/
│   └── invoice.txt
│
├── outputs/
│   └── invoice_output.json
│
├── tests/
│   ├── __init__.py
│   └── test_extractor.py
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

## Extracted Information

The application extracts:

- Invoice number
- Invoice date and due date
- Vendor and customer names
- Currency
- Line items
- Quantity and unit price
- Subtotal, tax and total amount

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/adarshpatil-07/Datheon_Python_Generative_AI_LangChain_Internship.git
cd Datheon_Python_Generative_AI_LangChain_Internship/Task-1-LLM-JSON-Extractor
```

**2. Create and activate a virtual environment**

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

**4. Configure environment variables**

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Replace the placeholder with your own Gemini API key. Never commit your `.env` file.

## Usage

Run the application:

```bash
python main.py
```

The extracted and validated invoice data will be saved to:

```text
outputs/invoice_output.json
```

## Running Tests

Execute the test suite:

```bash
python -m pytest -v
```

The test suite covers prompt generation, schema validation, JSON parsing, extraction pipeline behavior and error handling.

**Latest verification:** 20 tests passed.

## Sample Output

```json
{
    "invoice_number": "INV-2026-001",
    "invoice_date": "2026-11-04",
    "due_date": "2026-11-14",
    "vendor_name": "ABC Technologies Pvt Ltd",
    "customer_name": "XYZ Solutions",
    "currency": "INR",
    "items": [
        {
            "description": "Wireless Keyboard",
            "quantity": 2.0,
            "unit_price": "750",
            "total": "1500"
        },
        {
            "description": "Wireless Mouse",
            "quantity": 1.0,
            "unit_price": "500",
            "total": "500"
        }
    ],
    "subtotal": "2000",
    "tax": "360",
    "total_amount": "2360"
}
```

## Internship

**Program:** Datheon Python + Generative AI & LangChain Internship

**Task:** Structured LLM Prompt Pipeline & JSON Extractor

**Author:** Adarsh Patil

---

This project demonstrates structured information extraction using LLM APIs, prompt engineering and schema-based validation.