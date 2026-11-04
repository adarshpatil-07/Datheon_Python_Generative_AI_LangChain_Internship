import json
import os
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project directory is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from google.genai import errors
from pydantic import ValidationError

from prompts.invoice_prompt import build_invoice_prompt, INVOICE_EXTRACTION_PROMPT
from schemas.invoice import Invoice, InvoiceItem
from main import extract_invoice, main, BASE_DIR


class TestInvoicePromptTemplate(unittest.TestCase):
    """Verifies requirement: Parameterized prompt templates."""

    def test_build_prompt_with_valid_text(self):
        sample_text = "Invoice #101\nTotal: 100 USD"
        prompt = build_invoice_prompt(sample_text)
        self.assertIn(sample_text, prompt)
        self.assertIn("INVOICE TEXT:", prompt)
        self.assertIn("INSTRUCTIONS:", prompt)

    def test_build_prompt_empty_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_invoice_prompt("")
        with self.assertRaises(ValueError):
            build_invoice_prompt("   ")

    def test_build_prompt_non_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_invoice_prompt(None)  # type: ignore
        with self.assertRaises(ValueError):
            build_invoice_prompt(12345)  # type: ignore


class TestInvoiceSchemaValidation(unittest.TestCase):
    """Verifies requirement: Pydantic schema validation & Strict JSON output."""

    def setUp(self):
        self.valid_data = {
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

    def test_valid_invoice_parsing(self):
        invoice = Invoice.model_validate(self.valid_data)
        self.assertEqual(invoice.invoice_number, "INV-2026-001")
        self.assertEqual(invoice.invoice_date, date(2026, 11, 4))
        self.assertEqual(invoice.due_date, date(2026, 11, 14))
        self.assertEqual(len(invoice.items), 2)
        self.assertEqual(invoice.total_amount, Decimal("2360"))

    def test_optional_due_date_allowed_none(self):
        data = dict(self.valid_data)
        data["due_date"] = None
        invoice = Invoice.model_validate(data)
        self.assertIsNone(invoice.due_date)

    def test_missing_required_field_raises_validation_error(self):
        data = dict(self.valid_data)
        del data["invoice_number"]
        with self.assertRaises(ValidationError):
            Invoice.model_validate(data)

    def test_empty_items_list_raises_validation_error(self):
        data = dict(self.valid_data)
        data["items"] = []
        with self.assertRaises(ValidationError):
            Invoice.model_validate(data)

    def test_negative_quantity_raises_validation_error(self):
        data = dict(self.valid_data)
        data["items"] = [
            {
                "description": "Item",
                "quantity": -1.0,
                "unit_price": "100",
                "total": "100"
            }
        ]
        with self.assertRaises(ValidationError):
            Invoice.model_validate(data)

    def test_zero_quantity_raises_validation_error(self):
        data = dict(self.valid_data)
        data["items"] = [
            {
                "description": "Item",
                "quantity": 0,
                "unit_price": "100",
                "total": "0"
            }
        ]
        with self.assertRaises(ValidationError):
            Invoice.model_validate(data)

    def test_negative_monetary_amount_raises_validation_error(self):
        data = dict(self.valid_data)
        data["total_amount"] = "-10"
        with self.assertRaises(ValidationError):
            Invoice.model_validate(data)

    def test_invalid_date_format_raises_validation_error(self):
        data = dict(self.valid_data)
        data["invoice_date"] = "not-a-date"
        with self.assertRaises(ValidationError):
            Invoice.model_validate(data)

    def test_strict_json_roundtrip(self):
        invoice = Invoice.model_validate(self.valid_data)
        json_output = invoice.model_dump_json(indent=2)
        # Parse back to verify valid strict JSON syntax
        parsed = json.loads(json_output)
        self.assertEqual(parsed["invoice_number"], "INV-2026-001")
        # Validate directly from JSON string via Pydantic
        reconstructed = Invoice.model_validate_json(json_output)
        self.assertEqual(invoice, reconstructed)


class TestInvoiceExtractionPipeline(unittest.TestCase):
    """Verifies requirement: Python integration with Gemini API & Structured extraction."""

    def setUp(self):
        self.valid_json_payload = json.dumps({
            "invoice_number": "INV-100",
            "invoice_date": "2026-05-10",
            "due_date": "2026-05-20",
            "vendor_name": "Acme Corp",
            "customer_name": "Beta LLC",
            "currency": "USD",
            "items": [
                {
                    "description": "Consulting Services",
                    "quantity": 10.0,
                    "unit_price": "150.00",
                    "total": "1500.00"
                }
            ],
            "subtotal": "1500.00",
            "tax": "0.00",
            "total_amount": "1500.00"
        })

    def test_extract_invoice_empty_text_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_invoice("")
        with self.assertRaises(ValueError):
            extract_invoice("   ")
        with self.assertRaises(ValueError):
            extract_invoice(None)  # type: ignore

    def test_extract_invoice_with_mocked_client(self):
        mock_response = MagicMock()
        mock_response.text = self.valid_json_payload

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        invoice = extract_invoice("Unstructured invoice text...", client=mock_client)

        self.assertIsInstance(invoice, Invoice)
        self.assertEqual(invoice.invoice_number, "INV-100")
        self.assertEqual(invoice.vendor_name, "Acme Corp")
        self.assertEqual(invoice.total_amount, Decimal("1500.00"))
        mock_client.models.generate_content.assert_called_once()

    def test_extract_invoice_empty_response_raises_value_error(self):
        mock_response = MagicMock()
        mock_response.text = ""

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            extract_invoice("Unstructured invoice text...", client=mock_client)
        self.assertIn("empty response", str(ctx.exception))

    def test_extract_invoice_invalid_json_raises_validation_error(self):
        mock_response = MagicMock()
        mock_response.text = '{"invoice_number": "INV-100"}'  # missing required fields

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with self.assertRaises(ValidationError):
            extract_invoice("Unstructured invoice text...", client=mock_client)

    def test_extract_invoice_propagates_gemini_api_error(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = errors.APIError(
            code=403,
            response_json={"error": {"message": "API key expired or invalid"}}
        )

        with self.assertRaises(errors.APIError):
            extract_invoice("Unstructured invoice text...", client=mock_client)


class TestExecutableScriptAndOutput(unittest.TestCase):
    """Verifies requirement: An executable script that produces validated JSON."""

    def test_sample_invoice_file_exists(self):
        sample_file = BASE_DIR / "sample_data" / "invoice.txt"
        self.assertTrue(sample_file.exists(), f"Sample invoice missing at {sample_file}")
        self.assertGreater(sample_file.stat().st_size, 0)

    def test_output_json_file_validity(self):
        output_file = BASE_DIR / "outputs" / "invoice_output.json"
        self.assertTrue(output_file.exists(), f"Output file missing at {output_file}")

        # Must parse as valid strict JSON
        raw_text = output_file.read_text(encoding="utf-8")
        parsed_json = json.loads(raw_text)
        self.assertIsInstance(parsed_json, dict)

        # Must validate strictly against Pydantic schema
        validated_invoice = Invoice.model_validate(parsed_json)
        self.assertEqual(validated_invoice.invoice_number, "INV-2026-001")
        self.assertEqual(validated_invoice.currency, "INR")
        self.assertEqual(len(validated_invoice.items), 2)
        self.assertEqual(validated_invoice.total_amount, Decimal("2360"))

    def test_main_handles_missing_file_gracefully(self):
        with patch("pathlib.Path.exists", return_value=False):
            with patch("builtins.print") as mock_print:
                main()
                printed = " ".join(call.args[0] for call in mock_print.call_args_list if call.args)
                self.assertIn("not found", printed.lower())


if __name__ == "__main__":
    unittest.main()
