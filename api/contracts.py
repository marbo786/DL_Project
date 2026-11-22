from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Region(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def bounds(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("region extends beyond the oriented image")
        return self


class Heatmap(BaseModel):
    width: int = Field(gt=0, le=128)
    height: int = Field(gt=0, le=128)
    values: list[float] = Field(max_length=128 * 128)

    @model_validator(mode="after")
    def bounds(self):
        if len(self.values) != self.width * self.height or any(
            not 0 <= value <= 1 for value in self.values
        ):
            raise ValueError("expected a bounded row-major heatmap")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    decision: Literal["no_strong_evidence", "suspicious", "inconclusive"]
    predicted_class: Literal["genuine", "manipulated"] | None = None
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    regions: list[Region] = Field(default_factory=list, max_length=32)
    heatmap: Heatmap | None = None
    warnings: list[str] = Field(default_factory=list, max_length=20)


class Analysis(Prediction):
    analysis_id: str
    model_version: str
    inference_ms: float
    image_width: int
    image_height: int
    rejected: bool