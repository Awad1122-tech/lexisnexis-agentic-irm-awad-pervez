from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PreviousAddress(BaseModel):
    address: str = Field(description="The previous address")
    residency_score: str = Field(description="Confidence level the subject lived here, e.g. High, Medium, Low")
    sources_count: int = Field(description="Number of data sources confirming this address")
    occupied_period: str = Field(description="Date range this address was occupied, e.g. '2018 - 2026'")
    forwarding_confirmed: Optional[bool] = Field(default=None, description="Whether a forwarding address link was confirmed to the next address, if applicable")


class Directorship(BaseModel):
    company_name: str = Field(description="Name of the company")
    company_number: str = Field(description="UK Companies House registration number")
    appointed_date: str = Field(description="Date the subject was appointed as director")
    status: Literal["Active", "Dissolved"] = Field(description="Current status of the company")


class PropertyRecord(BaseModel):
    address: str = Field(description="Address of the property")
    owner_confirmed: bool = Field(description="Whether ownership was confirmed via the property register")
    ownership_status: str = Field(description="Detail of ownership, e.g. 'Confirmed current owner since 2019' or 'No register entry - rental property'")
    is_subject_residence: bool = Field(description="True if the subject actually lives here, False if owned but not resided in (e.g. buy-to-let)")



class TraceIQFindings(BaseModel):
    # Identity Tracing (LexID Linking)
    lex_id: Optional[str] = Field(default=None, description="Unique LexID identifier, None if not yet assigned")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/spellings associated with this identity")

    # Current Address Profile
    current_address: str = Field(description="The subject's current address on file")
    residency_score: Literal["Not Calculated", "Very Low", "Low", "Medium", "High", "Very High"] = Field(description="Confidence level the subject genuinely lives at the current address")
    last_seen: Optional[str] = Field(default=None, description="Most recent date confirming this address")
    sources_count: int = Field(description="Number of data sources backing the current address")

    # Property Details
    current_property: Optional[PropertyRecord] = Field(default=None, description="Property record for the subject's current residence")
    additional_properties: List[PropertyRecord] = Field(default_factory=list, description="Any additional owned properties not resided in, e.g. buy-to-let")

    # Occupants
    linked_occupants_count: int = Field(description="Number of other people linked to the current address")
    linked_occupants_names: List[str] = Field(default_factory=list, description="Names of linked occupants, if known")

    # Previous Address History
    previous_addresses: List[PreviousAddress] = Field(default_factory=list, description="List of previous addresses on file")

    # Financial and Corporate Records
    mortality_status: Literal["No Match", "Potential Match", "Confirmed Match"] = Field(description="Result of mortality/death register screening")
    ccj_count: int = Field(description="Number of County Court Judgments found")
    has_bankruptcy_iva_dro: bool = Field(description="Whether any bankruptcy, IVA, or Debt Relief Order was found")
    directorships: List[Directorship] = Field(default_factory=list, description="Company directorships associated with this identity")

    # Linked Associated Individuals
    associated_individuals_note: Optional[str] = Field(default=None, description="Summary note on associated individuals and whether connections are established or unexplained")