"""Paths, source URLs and raw-file layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
MANIFEST = RAW / "manifest.json"

NHTSA_FFDD = "https://static.nhtsa.gov/odi/ffdd"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    columns: tuple[str, ...]


# Field order from https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt (last updated 2026-04-30).
COMPLAINT_COLUMNS = (
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE",
    "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES",
    "OCCURENCES", "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE",
    "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL",
    "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
)

# Field order from https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt (last updated May 2025).
RECALL_COLUMNS = (
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO", "COMPNAME", "MFGNAME",
    "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE",
    "DATEA", "RPNO", "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
    "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO", "DO_NOT_DRIVE",
    "PARK_OUTSIDE",
)

# Columns that identify a person or their location precisely; dropped before anything is
# written to data/clean. Free-text narratives (CDESCR) are scrubbed separately in Phase 2.
COMPLAINT_PII_COLUMNS = (
    "CITY", "VIN", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_ZIP", "VEHICLE_OPERATOR",
)

SOURCES = (
    Source("complaints", f"{NHTSA_FFDD}/cmpl/FLAT_CMPL.zip", COMPLAINT_COLUMNS),
    # NHTSA no longer posts a single FLAT_RCL.zip; recalls are split at 2010.
    Source("recalls_pre_2010", f"{NHTSA_FFDD}/rcl/FLAT_RCL_PRE_2010.zip", RECALL_COLUMNS),
    Source("recalls_post_2010", f"{NHTSA_FFDD}/rcl/FLAT_RCL_POST_2010.zip", RECALL_COLUMNS),
)
