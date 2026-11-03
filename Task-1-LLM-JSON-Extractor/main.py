import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from schemas.invoice import Invoice
from prompts.invoice_prompt import build_invoice_prompt


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")


# Initialize Gemini client
client = genai.Client(api_key=API_KEY)


def extract_invoice(invoice_text: str) -> Invoice:
    """Extract structured invoice data using Gemini."""

    prompt = build_invoice_prompt(invoice_text)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Invoice.model_json_schema(),
            temperature=0
        )
    )

    # Validate Gemini's JSON response
    invoice = Invoice.model_validate_json(response.text)

    return invoice


def main():
    invoice_path = Path("sample_data/invoice.txt")

    invoice_text = invoice_path.read_text(encoding="utf-8")

    print("\nExtracting invoice information...\n")

    invoice = extract_invoice(invoice_text)

    print("Validated Invoice Data:\n")
    print(invoice.model_dump_json(indent=4))

    # Save JSON output
    output_path = Path("outputs/invoice_output.json")
    output_path.parent.mkdir(exist_ok=True)

    output_path.write_text(
        invoice.model_dump_json(indent=4),
        encoding="utf-8"
    )

    print(f"\nJSON saved to: {output_path}")


if __name__ == "__main__":
    main()
