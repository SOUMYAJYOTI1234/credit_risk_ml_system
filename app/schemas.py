"""
schemas.py - Pydantic Request / Response Schemas for the Credit Risk API
"""

from pydantic import BaseModel, Field
from typing import Optional


class CreditApplicationRequest(BaseModel):
    """Input schema matching the cleaned dataset features.

    Field names follow the snake_case conventions used in data_loader.py.
    All 23 original features are expected; the API's feature-engineering
    step will add the 5 derived features automatically.
    """

    limit_bal: float = Field(..., description="Credit limit (NT dollars)")
    sex: int = Field(..., ge=1, le=2, description="1=Male, 2=Female")
    education: int = Field(..., ge=1, le=4, description="1=Grad, 2=University, 3=High school, 4=Other")
    marriage: int = Field(..., ge=1, le=3, description="1=Married, 2=Single, 3=Other")
    age: int = Field(..., ge=18, description="Age in years")

    # Repayment status for months 1–6  (-2 to 9)
    pay_1: int = Field(..., ge=-2, le=9)
    pay_2: int = Field(..., ge=-2, le=9)
    pay_3: int = Field(..., ge=-2, le=9)
    pay_4: int = Field(..., ge=-2, le=9)
    pay_5: int = Field(..., ge=-2, le=9)
    pay_6: int = Field(..., ge=-2, le=9)

    # Bill amounts for months 1–6
    bill_amt1: float = Field(...)
    bill_amt2: float = Field(...)
    bill_amt3: float = Field(...)
    bill_amt4: float = Field(...)
    bill_amt5: float = Field(...)
    bill_amt6: float = Field(...)

    # Payment amounts for months 1–6
    pay_amt1: float = Field(...)
    pay_amt2: float = Field(...)
    pay_amt3: float = Field(...)
    pay_amt4: float = Field(...)
    pay_amt5: float = Field(...)
    pay_amt6: float = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "limit_bal": 20000,
                    "sex": 2,
                    "education": 2,
                    "marriage": 1,
                    "age": 24,
                    "pay_1": 2,
                    "pay_2": 2,
                    "pay_3": -1,
                    "pay_4": -1,
                    "pay_5": -2,
                    "pay_6": -2,
                    "bill_amt1": 3913,
                    "bill_amt2": 3102,
                    "bill_amt3": 689,
                    "bill_amt4": 0,
                    "bill_amt5": 0,
                    "bill_amt6": 0,
                    "pay_amt1": 0,
                    "pay_amt2": 689,
                    "pay_amt3": 0,
                    "pay_amt4": 0,
                    "pay_amt5": 0,
                    "pay_amt6": 0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema returned by the /predict endpoint."""

    default_probability: float = Field(
        ..., ge=0, le=1,
        description="Probability that the applicant will default next month",
    )
    prediction: int = Field(
        ..., ge=0, le=1,
        description="Binary prediction (0 = no default, 1 = default)",
    )
    threshold: float = Field(
        ..., description="Decision threshold used for the binary prediction",
    )


class HealthResponse(BaseModel):
    """Response schema for the health / root endpoint."""

    status: str
    model_loaded: bool
    version: str
