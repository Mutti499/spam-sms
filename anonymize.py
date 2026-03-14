"""
Generalized SMS PII Anonymizer
- Detects names from context patterns (Sayın X, deliveries, transfers, etc.)
- Detects emails, phone numbers, masked IDs, IMEI, IPs via regex
- Assigns consistent [PERSON_1], [PERSON_2], etc. tags per unique identity
- Uses word-boundary-aware replacement to avoid false positives
"""

import json
import re
from collections import defaultdict

with open("sms.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ════════════════════════════════════════════════════════════
# PHASE 1: Discover PII from the dataset
# ════════════════════════════════════════════════════════════

# Turkish letter set for word boundaries
TR_LETTERS = r'a-zA-ZçÇğĞıİöÖşŞüÜ'
WB_BEFORE = f'(?<![{TR_LETTERS}@])'  # not preceded by letter
WB_AFTER = f'(?![{TR_LETTERS}@])'     # not followed by letter

# ── Generic words that are NOT person names ──
NOT_NAMES = {
    # Turkish customer/member/student terms
    "musterimiz", "müsterimiz", "musterimiz bekleyen",
    "misafirimiz", "yolcumuz", "ogrencimiz", "öğrencimiz",
    "abonemiz", "görevli", "istanbullumuz", "hastamız",
    "adayımız", "genç", "akademi", "öğrenciler", "velimiz",
    "üyelerimiz", "üyemiz",
    # Company/brand names
    "turknet", "turknetli", "turkcell", "sirketimizce",
    "compecli", "compec", "gigafiber", "giga", "turk",
    "macfit", "amazon hub counter personeli",
    # Generic words
    "senin", "hesabınız", "degerli", "sevgili",
    "genç arkadaşım", "senin için bir",
    # Facility names (not people)
    "turgut özakman", "bülent ecevit",
    "kuzey kargo dolabı",
    # Roles
    "veli ve sevgili", "akademi öğrencimiz",
}

# Turkish suffixes for name matching (cım, m, nın, aa, iii, etc.)
TR_SUFFIX = r'(?:[cçCÇ][iıİIıi][mMğĞ]|[mnMN][iıİI][nN]?|[aAeEğĞ]{1,5}|[iıİI]{1,5}|[mM])?'


def looks_like_person_name(name):
    """Filter: does this look like a real person name?"""
    name = name.strip()
    if name.lower() in NOT_NAMES:
        return False
    if len(name) < 3:
        return False
    # Skip if starts with B0/B1 (message codes)
    if re.match(r'^B\d', name):
        return False
    # Must have at least one proper-case word (capitalized or all-caps)
    words = name.split()
    if not any(w[0].isupper() or w.isupper() for w in words if w and w[0].isalpha()):
        return False
    # Skip if it's a single generic Turkish word
    if len(words) == 1 and name.lower() in {
        "musterimiz", "müsterimiz", "misafirimiz", "yolcumuz",
        "ogrencimiz", "öğrencimiz", "abonemiz", "görevli",
        "hastamız", "adayımız", "genç", "akademi", "velimiz",
        "üyelerimiz", "üyemiz", "buse",
    }:
        return False
    # Skip service/location names
    if any(x in name.lower() for x in [
        "kargo", "servis", "personeli", "hub counter",
        "arkadaşım", "için bir", "hesab", "üyemiz",
        "ogrencimiz", "öğrencimiz", "sirketimiz",
        "firmamiz", "yolcumuz", "mr ", "hastamız"
    ]):
        return False
    return True


def extract_person_names(text):
    """Extract person names from contextual patterns in SMS text."""
    names = set()

    patterns = [
        # "Sayın/Sevgili/Merhaba FirstName LastName"
        (re.compile(
            r'(?:Say[ıi]n|Sevgili|Merhaba|SAYIN)\s+'
            r'([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+){0,2})',
            re.UNICODE
        ), True),
        # "Sn. FirstName LastName,"
        (re.compile(
            r'Sn\.?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\*]+(?:\s+[A-ZÇĞİÖŞÜa-zçğıöşü\*]+){0,2})\s*[,.\n]',
            re.UNICODE
        ), True),
        # "SAYIN X PAROLANIZ" / "SAYIN X,"
        (re.compile(
            r'SAYIN\s+([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,2})\s*[,.\n]',
            re.UNICODE
        ), True),
        # Delivery: "gönderiniz FirstName LastName(Kendisi)"
        (re.compile(
            r'(?:gönderiniz|gonderiniz)\s+'
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){0,2})\s*\(',
            re.UNICODE
        ), True),
        # "X(Kendisi/Komşu/Baba/Anne/Yakını)" — only capitalized proper names
        (re.compile(
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,2})\s*\((?:Kendisi|Komşu|Kapı|Anne|Baba|Yakın[ıi])',
            re.UNICODE
        ), True),
        # Transfer: "hesaptan X adına Y"
        (re.compile(
            r'(?:hesaptan)\s+'
            r'([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\s+(?:adina|adına)',
            re.UNICODE
        ), True),
        # "X tarafından FAST ile"
        (re.compile(
            r'([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+)+)\s+(?:tarafindan|tarafından)\s+FAST',
            re.UNICODE
        ), True),
        # Masked names: "M**** A****", "MU*** AT***"
        (re.compile(
            r'([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]*\*{2,}(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]*\*{2,})+)',
            re.UNICODE
        ), False),
        # Masked: "S..... A....."
        (re.compile(
            r'([A-ZÇĞİÖŞÜ]\.{3,}(?:\s+[A-ZÇĞİÖŞÜ]\.{3,})+)',
            re.UNICODE
        ), False),
    ]

    for pattern, needs_filter in patterns:
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if needs_filter and not looks_like_person_name(name):
                continue
            if not needs_filter or looks_like_person_name(name):
                names.add(name)

    return names


# ── Collect all names ──
all_names = defaultdict(int)
for msg in data:
    for name in extract_person_names(msg["text"]):
        all_names[name] += 1

# ── Cluster into identities ──
# Two names are the same person if they share a significant token (>=4 chars)
# AND the shared token is not a common word
COMMON_TOKENS = {"nolu", "adli", "kisi", "sayin", "teslim", "siparis", "amazon"}


def normalize(name):
    n = name.lower().strip()
    n = re.sub(r'[\*\.]+', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def significant_tokens(name):
    norm = normalize(name)
    return {t for t in norm.split() if len(t) >= 4 and t not in COMMON_TOKENS}


identity_map = {}  # name -> person_id
person_id_counter = 0
clusters = []  # list of (person_id, set of significant tokens)

sorted_names = sorted(all_names.items(), key=lambda x: (-x[1], -len(x[0])))

for name, count in sorted_names:
    tokens = significant_tokens(name)
    if not tokens:
        # Short/masked name - create new identity
        person_id_counter += 1
        identity_map[name] = f"[PERSON_{person_id_counter}]"
        clusters.append((f"[PERSON_{person_id_counter}]", tokens))
        continue

    # Try to match with existing cluster
    matched = None
    for pid, cluster_tokens in clusters:
        overlap = tokens & cluster_tokens
        if overlap:
            matched = pid
            cluster_tokens.update(tokens)  # expand cluster
            break

    if matched is None:
        person_id_counter += 1
        matched = f"[PERSON_{person_id_counter}]"
        clusters.append((matched, tokens))

    identity_map[name] = matched

# ── Collect emails ──
email_re = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
SKIP_EMAIL_DOMAINS = {"kvkk", "destek@", "helloworld@", "info@", "iletisim@"}
all_emails = set()
for msg in data:
    for m in email_re.finditer(msg["text"]):
        email = m.group()
        if not any(x in email.lower() for x in SKIP_EMAIL_DOMAINS):
            all_emails.add(email)

email_map = {}
for i, email in enumerate(sorted(all_emails), 1):
    email_map[email] = f"[EMAIL_{i}]"

# ── Print discoveries ──
print("=" * 60)
print(f"DISCOVERED {len(identity_map)} NAME VARIANTS → {person_id_counter} IDENTITIES")
print("=" * 60)
for name, pid in sorted(identity_map.items(), key=lambda x: (x[1], -all_names[x[0]])):
    print(f"  {pid:14s}  ←  \"{name}\" ({all_names[name]}x)")

print(f"\n{'=' * 60}")
print(f"DISCOVERED {len(email_map)} PERSONAL EMAILS")
print("=" * 60)
for email, tag in email_map.items():
    print(f"  {tag}  ←  {email}")


# ════════════════════════════════════════════════════════════
# PHASE 2: Anonymize
# ════════════════════════════════════════════════════════════

# Sort replacements longest-first
sorted_name_repls = sorted(identity_map.items(), key=lambda x: len(x[0]), reverse=True)
sorted_email_repls = sorted(email_map.items(), key=lambda x: len(x[0]), reverse=True)

# Collect first-name tokens for suffix-aware matching
firstname_set = {}
for name, pid in identity_map.items():
    for part in name.split():
        clean = re.sub(r'[\*\.]+', '', part)
        if len(clean) >= 4 and clean[0].isupper():
            if clean not in firstname_set:
                firstname_set[clean] = pid
# Sort longest first
sorted_firstnames = sorted(firstname_set.items(), key=lambda x: len(x[0]), reverse=True)


def anonymize_text(text):
    # 1. Replace full name strings (word-boundary aware for all)
    for name, tag in sorted_name_repls:
        if '*' in name or '.' in name:
            # Masked names: exact replace is fine (unlikely false positive)
            text = text.replace(name, tag)
        else:
            # Use word-boundary-aware replacement
            pattern = (
                WB_BEFORE
                + re.escape(name)
                + WB_AFTER
            )
            text = re.sub(pattern, tag, text)

    # 2. First names with Turkish suffix awareness (case-insensitive)
    for firstname, tag in sorted_firstnames:
        pattern = (
            WB_BEFORE
            + re.escape(firstname)
            + TR_SUFFIX
            + WB_AFTER
        )
        text = re.sub(pattern, tag, text, flags=re.IGNORECASE)

    # 3. Emails
    for email, tag in sorted_email_repls:
        text = text.replace(email, tag)

    # 4. Masked email patterns (e.g., "mu**2@ho**.com")
    text = re.sub(
        r'[a-zA-Z0-9]+\*{2,}[a-zA-Z0-9]*@[a-zA-Z0-9]+\*{2,}[a-zA-Z0-9]*\.[a-zA-Z]+',
        '[EMAIL_MASKED]',
        text
    )

    # 5. Turkish phone numbers
    text = re.sub(
        r'(?<!\d)(?:\+?90)?[- ]?(5\d{2})[- ]?(\d{3})[- ]?(\d{2})[- ]?(\d{2})(?!\d)',
        '[PHONE]',
        text
    )

    # 6. Masked phones/IDs (digits + asterisks, 10+ chars)
    text = re.sub(
        r'(?<![a-zA-Z\d])[\d\*]{10,15}(?!\d)',
        lambda m: '[PHONE_MASKED]' if '*' in m.group() else m.group(),
        text
    )

    # 7. IP addresses (private/static, not URLs)
    text = re.sub(
        r'(?<!\d)(?:10|172|192|212|213|78|85)\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)',
        '[IP_ADDRESS]',
        text
    )

    # 8. Prescription numbers
    text = re.sub(
        r'reçete numaranız:\s*[A-Z0-9]{6,8}/[A-Z0-9]{6,8}',
        'reçete numaranız: [PRESCRIPTION]',
        text
    )

    # 9. Card last-4 in banking context
    text = re.sub(
        r'(?i)(SON 4 HANES[İI]\s+)\d{4}',
        r'\1[CARD_LAST4]',
        text
    )
    text = re.sub(
        r'(?i)(\d{4})(\s+(?:ile biten|OLAN)\s+(?:kart|KART))',
        r'[CARD_LAST4]\2',
        text
    )

    return text


# ── Process ──
count = 0
for msg in data:
    original = msg["text"]
    msg["text"] = anonymize_text(original)
    if msg["text"] != original:
        count += 1

print(f"\n{'=' * 60}")
print(f"RESULT: Anonymized {count} messages out of {len(data)} total")
print("=" * 60)

with open("sms_anonymized.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved to sms_anonymized.json")
