"""Mask personal details in complaint narratives -> data/clean/narratives.parquet.

NHTSA keeps complainants' names and contact details out of its public file and redacts names in
the text as [XXX], but the redaction misses things. Profiling the October 2026 files found e-mail
addresses, phone numbers, full VINs, links to personal photo albums, case and claim numbers that
tie a complaint to a person, and names after titles ("MR. ..."). Each is replaced with a typed
placeholder such as [PHONE]. NHTSA reviewer codes (*AK, *JB), on 45% of narratives, are dropped.

Names without a title or "my name is" (a signature such as "JOHN SMITH 555-0100" loses the phone
number but keeps the name) cannot be found by pattern, so the text is safer than NHTSA's file but
not guaranteed to be name-free.

Output: one row per complaint row with text
  complaint_id   joins to complaints.parquet
  narrative_id   md5 of the scrubbed text; identical texts share it (embed.py keys on it)
  description    the scrubbed narrative
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
from duckdb.sqltypes import VARCHAR

from pipeline.config import CLEAN

_I = re.IGNORECASE

_URL = re.compile(r"(?:https?://|\bwww\.)[^\s<>\"]*[^\s<>\".,;:!?)\]'*]", _I)
_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)*\.[A-Z]{2,}", _I)
# Reviewer initials such as "*AK" or ".*DT*AK", but not emphasis like "did *NOT* deploy".
_STAFF_CODE = re.compile(r"(?<![A-Za-z0-9])(?:\*{1,3}(?!NO\b)[A-Z]{2,3})+(?=\s|$|[.,;:])")
# Full VIN: check digit (or X) in position 9 and digits in the last three positions.
_VIN = re.compile(r"\b[A-HJ-NPR-Z0-9]{8}[0-9X][A-HJ-NPR-Z0-9]{5}[0-9]{3}\b", _I)
# Partial VINs after the word VIN ("VIN 1HGCM8263******", "LAST 8 OF VIN 3A004352").
_VIN_AFTER = re.compile(
    r"(\bVIN\b(?:\s*(?:NUMBER|NUM|NO\.?|#))?\s*[:#.-]?\s*)"
    r"(?=[A-Z0-9*]*\d)[A-HJ-NPR-Z0-9*]{6,17}(?![A-Z0-9*])", _I)
# A separator is required after a bare area code: "AFFECTED: 234567 2019 HONDA" is no phone.
_PHONE = re.compile(
    r"(?:\+?1[-. ]?)?(?:\(\s?[2-9]\d{2}\s?\)\s?|\b[2-9]\d{2}[-. ])\d{3}[-. ]\d{4}\b")
_PHONE_AFTER = re.compile(
    r"(\b(?:PHONE|CELL|TEL|TELEPHONE|FAX|MOBILE)\b(?:\s*(?:NUMBER|NUM|NO\.?|#))?(?:\s*IS)?"
    r"\s*[:#-]?\s*)(\+?\(?\d(?:[\d()./-]|\s(?=[\d(]))*\d)", _I)
# Manufacturer case numbers, insurance claims, police reports: they link a complaint to a person.
_CASE_ID = re.compile(
    r"(\b(?:CLAIM|CASE|POLICY|REFERENCE|REF|CONFIRMATION|TICKET|ACCOUNT|ACCT|FILE|"
    r"SERVICE REQUEST|REPAIR ORDER|RO|(?:POLICE|ACCIDENT|INCIDENT) REPORT)"
    r"\s*(?:NUMBER|NUM|NO\.?|#|ID|:)\s*[:#]?\s*)"
    r"(?=[A-Z0-9-]{5})(?=[A-Z-]*\d)[A-Z0-9][A-Z0-9-]*", _I)
_PLATE = re.compile(
    r"(\b(?:LICENSE\s*(?:PLATE|TAG)|(?:PLATE|TAG)\s*(?:NUMBER|NO\.?|#))\s*(?:NUMBER|NO\.?|#)?"
    r"\s*[:#]?\s*)(?=[A-Z-]*\d)[A-Z0-9-]{2,9}\b", _I)
# 9+ digit runs are phone, case, account or claim numbers, except recall/bulletin/part numbers.
_LONG_NUMBER = re.compile(r"\b\d{9,}\b")
_KEEP_NUMBER_AFTER = re.compile(
    r"\b(?:RECALL|CAMPAIGN|BULLETIN|TSB|PART)S?\s*(?:NUMBERS?|NOS?\.?|#|ID)?\s*[:#.]?\s*$", _I)
# "MS" and "DR" need a period: bare, they usually mean Mississippi or drive. After a number,
# "4 DR. LT" means doors.
_TITLE = re.compile(r"(?<![\w*])(?<!\d\s)(?:mrs?\.?|ms\.|dr\.)\s+", _I)
_MY_NAME_IS = re.compile(r"my name is\s+", _I)  # no \b: also catches typos like "GMy name is"
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*\.?")
# Words that follow a title or "my name is" without being part of the name.
_NOT_NAMES = frozenset("""
    MR MRS MS DR MISS
    A AN THE AND OR BUT NOR SO AT OF ON IN TO INTO FROM WITH FOR BY AS IF THAN THEN THERE HERE
    IS WAS WERE ARE BE BEEN BEING AM HAS HAD HAVE DO DID DOES DONE WILL WOULD CAN COULD SHOULD
    MAY MIGHT MUST SHALL I WE HE SHE THEY IT ME HIM HER US THEM MY HIS OUR THEIR YOUR YOU WHO
    WHOM WHICH WHAT THAT THIS THESE THOSE WHEN WHILE WHERE AFTER BEFORE SINCE UNTIL UPON DURING
    ALSO NOT NO YES ONE TWO THREE TOLD SAID SAYS STATED STATES ADVISED INFORMED ASKED CALLED
    REPLIED EXPLAINED AGREED REFUSED OFFERED WENT CAME GOT GAVE TOOK MADE SENT WROTE CONFIRMED
    CONSUMER CONTACT DEALER DEALERSHIP SERVICE MANAGER ADVISOR TECH TECHNICIAN REP OWNER
    CUSTOMER VEHICLE CAR TRUCK TIRE TIRES RECALL ENGINE TRANSMISSION CONTROL APPT APPOINTMENT
    OFFICE VISIT SCC GOODWRENCH APPTS APPOINTMENTS CEO FORD HUSBAND WIFE PLEASE BEGAN RESPONSE
    BACK COMPANY LETTER REGARDING ABOUT MOTORS FOUR ONCE HIMSELF HERSELF AUTO REPAIR TODAY VERY
    CALLS ANOTHER OPPOSES DIDN'T KNOW AGAIN PROVIDE SYSTEM PHONE BOSS EXECUTIVE TIMES VICE OUT
    VIA OVER UP EMAIL MET SAW SEE SEEN SAY TELL TELLS GET GETS GO GOES GONE KEPT LEFT LET PUT RAN
    SET TRIED FIXED USED JUST STILL ALREADY NEVER ALWAYS ONLY EVEN SOON LATER NOW AROUND ACROSS
    AGAINST ALONG AMONG BEHIND BETWEEN DOWN NEAR OFF THROUGH UNDER WITHOUT ALL ANY EACH EVERY
    SOME MANY MUCH MORE MOST OTHER SUCH SAME
""".split())
_ADDRESS = re.compile(
    r"(\b(?:I|WE)\s+LIVE\s+(?:AT|ON)\s+|\bMY\s+(?:HOME\s+)?ADDRESS\s+IS\s+|\bADDRESS\s*:\s*|"
    r"\bRESIDE\s+AT\s+)\d[\w-]*(?:\s+(?!(?:AND|BUT|THE|WHEN|WHERE|WHICH|WITH|THAT|SO|I|WE|MY)\b)"
    r"[A-Z0-9.'#-]+){1,5}", _I)
_ZIP = re.compile(r"(\bZIP(?:\s*CODE)?\s*[:#]?\s*)\d{5}(?:-\d{4})?\b", _I)
_SPACES = re.compile(r"\s+")


def _phone_after(m: re.Match) -> str:
    digits = sum(c.isdigit() for c in m.group(2))
    return m.group(1) + "[PHONE]" if digits >= 7 else m.group(0)


def _long_number(m: re.Match) -> str:
    keep = _KEEP_NUMBER_AFTER.search(m.string, max(0, m.start() - 40), m.start())
    return m.group(0) if keep else "[NUMBER]"


def _is_name_word(word: str, position: int, capitalized: bool) -> bool:
    core = word.rstrip(".").upper()
    if not word or (capitalized and not word[0].isupper()) or core in _NOT_NAMES:
        return False
    # In all-caps text a following verb or adverb looks like a surname ("MR. SMITH CONTACTED").
    return position == 0 or not (len(core) >= 6 and core.endswith(("ED", "ING", "LY")))


def _mask_names(text: str, trigger: re.Pattern, capitalized: bool) -> str:
    """Replace the name after each trigger with [NAME]: one or two words, or three when the
    middle one is an initial ("JOHN W. SMITH"). Only triggers are matched, so "MR. AND MRS.
    SMITH" still reaches SMITH."""
    out, pos = [], 0
    for m in trigger.finditer(text):
        if m.start() < pos:
            continue
        i = end = m.end()
        for position in range(3):
            w = _WORD.match(text, i)
            word = w.group() if w else ""
            if not _is_name_word(word, position, capitalized):
                break
            if len(word) > 2 and word.endswith("."):  # "SMITH." ends a sentence: keep the period
                end = w.end() - 1
                break
            end = w.end()
            gap = _SPACES.match(text, end)
            if not gap or (position == 1 and len(word.rstrip(".")) > 1):
                break
            i = gap.end()
        if end > m.end():
            out += [text[pos:m.end()], "[NAME]"]
            pos = end
    out.append(text[pos:])
    return "".join(out)


def scrub(text: str) -> str:
    # Non-breaking and doubled spaces would hide "555\xa0123-4567" and "MY  NAME IS" from the rules.
    text = _SPACES.sub(" ", text)
    low = text.lower()  # cheap keyword checks skip rules that cannot match
    if "://" in text or "www." in low:
        text = _URL.sub("[URL]", text)
    if "@" in text:
        text = _EMAIL.sub("[EMAIL]", text)
    if "*" in text:
        text = _STAFF_CODE.sub("", text)
    text = _VIN.sub("[VIN]", text)
    if "vin" in low:
        text = _VIN_AFTER.sub(r"\1[VIN]", text)
    if any(k in low for k in ("phone", "cell", "tel", "fax", "mobile")):
        text = _PHONE_AFTER.sub(_phone_after, text)
    text = _PHONE.sub("[PHONE]", text)
    text = _CASE_ID.sub(r"\1[ID]", text)
    if "plate" in low or "tag" in low:
        text = _PLATE.sub(r"\1[PLATE]", text)
    text = _LONG_NUMBER.sub(_long_number, text)
    if "mr" in low or "ms." in low or "dr." in low:
        text = _mask_names(text, _TITLE, capitalized=True)
    if "my name is" in low:
        text = _mask_names(text, _MY_NAME_IS, capitalized=False)
    if any(k in low for k in ("live at", "live on", "address", "reside at")):
        text = _ADDRESS.sub(r"\1[ADDRESS]", text)
    if "zip" in low:
        text = _ZIP.sub(r"\1[ZIP]", text)
    return _SPACES.sub(" ", text).strip()


def run(clean: Path = CLEAN) -> None:
    src, out = clean / "complaints.parquet", clean / "narratives.parquet"
    con = duckdb.connect()
    con.create_function("scrub", scrub, [VARCHAR], VARCHAR)
    # Scrub each distinct text once (one complaint repeats its text per component), joining back
    # on an md5 of the raw text so the join never holds two copies of every narrative.
    # MATERIALIZED: otherwise DuckDB copies scrub() into the pushed-down filter and calls it twice.
    con.execute(f"""
        COPY (
            WITH scrubbed AS MATERIALIZED (
                SELECT md5(description_raw) AS raw_md5,
                       nullif(scrub(description_raw), '') AS description
                FROM (SELECT DISTINCT description_raw FROM '{src.as_posix()}'
                      WHERE description_raw IS NOT NULL)
            )
            SELECT c.complaint_id, md5(s.description) AS narrative_id, s.description
            FROM (SELECT complaint_id, md5(description_raw) AS raw_md5 FROM '{src.as_posix()}'
                  WHERE description_raw IS NOT NULL) c
            JOIN scrubbed s USING (raw_md5)
            WHERE s.description IS NOT NULL
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    n, texts = con.execute(
        f"SELECT count(*), count(DISTINCT narrative_id) FROM '{out.as_posix()}'").fetchone()
    print(f"[scrub] narratives: {n:,} rows, {texts:,} distinct texts")
