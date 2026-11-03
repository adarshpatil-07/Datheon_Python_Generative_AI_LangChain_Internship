from datetime import date
from decimal import Decimal
from pydantic import BaseModel, Field


class InvoiceItem(BaseModel):
    description: str = Field(description="Description of the purchased item")
    quantity: float = Field(gt=0, description="Quantity purchased")
    unit_price: Decimal = Field(ge=0, description="Price per unit")
    total: Decimal = Field(ge=0, description="Total price for this item")


class Invoice(BaseModel):
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: date = Field(description="Date of invoice in YYYY-MM-DD format")
    due_date: date | None = Field(default=None, description="Payment due date")
    vendor_name: str = Field(description="Name of the seller or vendor")
    customer_name: str = Field(description="Name of the buyer or customer")
    currency: str = Field(description="Three-letter currency code, e.g. INR")
    items: list[InvoiceItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    tax: Decimal = Field(ge=0)
    total_amount: Decimal = Field(ge=0)
