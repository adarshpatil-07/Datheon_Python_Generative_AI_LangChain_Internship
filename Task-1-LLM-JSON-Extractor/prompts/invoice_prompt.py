INVOICE_EXTRACTION_PROMPT = """
You are an expert invoice information extraction system.

Your task is to extract structured information from the
invoice text provided below.

INSTRUCTIONS:

1. Extract information only from the supplied invoice text.
2. Never invent or assume missing information.
3. Use null for optional fields when information is unavailable.
4. Convert dates into YYYY-MM-DD format.
5. Identify the vendor and customer correctly.
6. Extract every invoice line item.
7. Calculate item totals only when the required values are available.
8. Extract subtotal, tax, and total amount accurately.
9. Use the appropriate three-letter currency code (e.g. INR, USD).
10. Return only data matching the requested JSON schema.

IMPORTANT:
- Do not include explanations outside the JSON.
- Do not fabricate invoice numbers, dates, prices, or names.
- If information is ambiguous, do not guess.

INVOICE TEXT:

{invoice_text}
"""


def build_invoice_prompt(invoice_text: str) -> str:
    """
    Build a parameterized prompt using the supplied invoice text.
    """
    return INVOICE_EXTRACTION_PROMPT.format(
        invoice_text=invoice_text
    )
