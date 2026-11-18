"""URL + text feature extraction for SafeNet."""
import math
import re
from urllib.parse import urlparse

URL_REGEX = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)
IP_REGEX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "support", "zip", "country", "kim", "loan"}
URL_KEYWORDS = ["login", "verify", "secure", "account", "update", "bank", "confirm", "signin",
                "password", "wallet", "free", "bonus", "gift", "billing", "suspend", "unlock", "paypal", "invoice"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly"}

URL_FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "num_dots", "num_hyphens", "num_digits", "num_special",
    "has_at", "has_ip", "is_https", "num_subdomains", "kw_count", "bad_tld", "entropy",
    "has_punycode", "double_slash_path", "is_shortener", "digit_ratio", "port_in_url",
]


def extract_urls(text: str):
    return [u.rstrip(".,);]") for u in URL_REGEX.findall(text or "")]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _parse(url: str):
    u = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u
    p = urlparse(u)
    host = (p.hostname or "").lower()
    return u, p, host


def url_features(url: str) -> list:
    u, p, host = _parse(url)
    low = u.lower()
    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    try:
        port = p.port
    except ValueError:
        port = None
    return [
        len(u), len(host), len(p.path), u.count("."), u.count("-"),
        sum(c.isdigit() for c in u), sum(not c.isalnum() for c in u),
        int("@" in u), int(bool(IP_REGEX.match(host))), int(p.scheme == "https"),
        max(len(parts) - 2, 0), sum(k in low for k in URL_KEYWORDS), int(tld in SUSPICIOUS_TLDS),
        _entropy(host), int("xn--" in host), int("//" in p.path), int(host in SHORTENERS),
        sum(c.isdigit() for c in u) / max(len(u), 1), int(port is not None),
    ]


def url_red_flags(url: str) -> list:
    """Human-readable reasons, used for explainability."""
    f = dict(zip(URL_FEATURE_NAMES, url_features(url)))
    flags = []
    if f["has_ip"]:
        flags.append("Link uses a raw IP address instead of a domain name")
    if f["has_at"]:
        flags.append("Link contains '@', which can hide the real destination")
    if f["num_subdomains"] >= 3:
        flags.append("Link has many subdomains (common in spoofed domains)")
    if f["bad_tld"]:
        flags.append("Domain uses a TLD frequently abused in scams")
    if f["kw_count"] >= 2:
        flags.append("Link contains several sensitive keywords (login/verify/secure...)")
    if f["has_punycode"]:
        flags.append("Domain uses punycode (possible look-alike characters)")
    if f["is_shortener"]:
        flags.append("Link is shortened, so the destination is hidden")
    if not f["is_https"]:
        flags.append("Link does not use HTTPS")
    if f["num_hyphens"] >= 3:
        flags.append("Link contains many hyphens")
    if f["url_len"] > 75:
        flags.append("Link is unusually long")
    return flags


TEXT_CUES = {
    "Urgency": r"\b(urgent|immediately|within \d+ ?(hours?|hrs?|minutes?)|asap|last chance|expires?|final notice|act now|today only)\b",
    "Credential request": r"\b(password|otp|pin|cvv|verify your (account|identity)|confirm your (account|details)|login|sign in|ssn|card number)\b",
    "Money / reward lure": r"\b(winner|won|prize|lottery|reward|cash ?back|gift card|free|bonus|claim|refund|inheritance|investment|crypto|double your)\b",
    "Threat / fear": r"\b(suspended|blocked|locked|terminated|unauthorized|legal action|arrest|penalty|compromised|deactivated)\b",
    "Authority impersonation": r"\b(bank|paypal|amazon|microsoft|apple|irs|income tax|hr department|it support|customs|courier|delivery)\b",
}


def text_cues(text: str) -> dict:
    t = (text or "").lower()
    out = {}
    for name, rx in TEXT_CUES.items():
        m = [x.group(0) for x in re.finditer(rx, t)]
        if m:
            out[name] = m
    return out
