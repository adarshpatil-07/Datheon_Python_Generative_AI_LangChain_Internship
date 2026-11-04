import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from schemas.invoice import Invoice
from prompts.invoice_prompt import build_invoice_prompt


# Resolve base directory for reliable relative path resolution
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def get_client() -> genai.Client:
    """Initialize and return a Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY") or API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")
    return genai.Client(api_key=api_key)


# Initialize default client if API key is present
client = genai.Client(api_key=API_KEY) if API_KEY else None


def extract_invoice(
    invoice_text: str,
    client: genai.Client | None = None,
    model: str | None = None
) -> Invoice:
    """Extract and validate structured invoice data using Gemini."""

    if not invoice_text or not isinstance(invoice_text, str) or not invoice_text.strip():
        raise ValueError("Invoice text cannot be empty.")

    target_client = client or get_client()
    target_model = model or MODEL

    prompt = build_invoice_prompt(invoice_text)

    response = target_client.models.generate_content(
        model=target_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Invoice.model_json_schema(),
            temperature=0
        )
    )

    if not response or not response.text or not response.text.strip():
        raise ValueError("Gemini returned an empty response.")

    # Validate Gemini's JSON response against the Pydantic schema
    invoice = Invoice.model_validate_json(response.text)

    return invoice


def main():
    invoice_path = BASE_DIR / "sample_data" / "invoice.txt"
    output_path = BASE_DIR / "outputs" / "invoice_output.json"

    try:
        # Read invoice text
        if not invoice_path.exists():
            raise FileNotFoundError(f"Invoice file not found at: {invoice_path}")

        invoice_text = invoice_path.read_text(encoding="utf-8")

        print("\nExtracting invoice information...\n")

        # Extract and validate
        invoice = extract_invoice(invoice_text)

        # Save validated JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            invoice.model_dump_json(indent=4),
            encoding="utf-8"
        )

        print("Invoice extracted and validated successfully!")
        print(f"\nJSON saved to: {output_path}")
        print("\nValidated Invoice Data:\n")
        print(invoice.model_dump_json(indent=4))

    except FileNotFoundError as error:
        print(f"Error: {error}")

    except ValidationError as error:
        print("\nError: Gemini returned data that failed schema validation.")
        print(error)

    except errors.APIError as error:
        print(f"\nGemini API Error: {error}")

    except ValueError as error:
        print(f"\nError: {error}")

    except OSError as error:
        print(f"\nFile system error: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
