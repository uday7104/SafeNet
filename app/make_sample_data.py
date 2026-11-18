"""Generates SMALL SYNTHETIC seed datasets so the prototype runs out of the box.
Replace data/text.csv and data/urls.csv with real datasets for meaningful results."""
import random
from pathlib import Path

import pandas as pd

random.seed(42)
DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

legit_t = [
    "Hi {n}, are we still meeting for lunch at {t}?", "Your order #{o} has been shipped and will arrive on Friday.",
    "Reminder: team standup tomorrow at {t}. Agenda attached.", "Thanks for your payment. Your receipt for invoice {o} is attached.",
    "Can you review my pull request when you get a chance?", "Happy birthday {n}! Hope you have a great day.",
    "The quarterly report is ready. Let me know if you have questions.", "Your appointment is confirmed for Monday at {t}.",
    "Class notes for today's lecture are uploaded to the portal.", "Mom: dinner is at {t}, don't be late!",
    "Your OTP for the transaction is {o}. Do not share it with anyone. - Bank", "Please find the meeting minutes attached.",
    "Check out this article on machine learning: https://en.wikipedia.org/wiki/Machine_learning",
    "Here is the project repo: https://github.com/example/project, please take a look.",
]
phish_t = [
    "URGENT: Your {b} account has been suspended. Verify your password immediately at {u}",
    "Dear customer, unusual login detected. Confirm your account details within 24 hours: {u}",
    "Congratulations {n}! You have won a $1000 gift card. Claim your prize now: {u}",
    "Final notice: your bank account is locked. Click {u} and enter your card number and PIN to unlock.",
    "You have a pending refund of $450. Login to {u} to claim it today only.",
    "Your package could not be delivered. Pay the customs fee at {u} or it will be returned.",
    "IT Support: your mailbox is full. Sign in at {u} to avoid deactivation within 2 hours.",
    "Earn double your money with our crypto investment. Limited time bonus, register: {u}",
    "Security alert: unauthorized access to your {b} account. Update your password now {u}",
    "Hello, I am a prince and need your help transferring funds. Send your bank details and get 20% reward.",
    "Your tax refund is ready. Verify your identity and SSN at {u} immediately.",
    "WINNER!! Free cash prize waiting. Reply with your OTP to claim your reward now.",
]
names = ["Aman", "Riya", "Sam", "Priya", "John", "Neha"]
brands = ["PayPal", "Amazon", "Microsoft", "HDFC Bank", "Apple", "Netflix"]
times = ["1pm", "9am", "4:30pm", "6pm"]


def fill(tpl, u=""):
    return tpl.format(n=random.choice(names), t=random.choice(times), o=random.randint(1000, 99999),
                      b=random.choice(brands), u=u)


def phish_url():
    brand = random.choice(["paypal", "amazon", "microsoft", "hdfcbank", "apple", "netflix"])
    kind = random.randint(0, 4)
    tld = random.choice(["xyz", "top", "tk", "click", "ml", "com", "net"])
    kw = random.choice(["login", "verify", "secure", "update", "account", "confirm"])
    if kind == 0:
        return f"http://{brand}-{kw}-{random.choice(['secure','support','id'])}.{tld}/{kw}/index.php?id={random.randint(1000,99999)}"
    if kind == 1:
        return f"http://{random.randint(11,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}/{brand}/{kw}.html"
    if kind == 2:
        return f"https://{kw}.{brand}.com.{random.choice(['account','secure'])}.{random.choice(['verify','portal'])}.{tld}/signin"
    if kind == 3:
        return f"http://www.{brand}.com@{random.randint(11,223)}.{random.randint(0,255)}.1.{random.randint(1,254)}/{kw}"
    return f"http://{brand}{random.randint(1,99)}{kw}.{tld}/wp-content/{kw}/confirm.php"


def legit_url():
    dom = random.choice(["google.com", "wikipedia.org", "github.com", "amazon.com", "microsoft.com", "bbc.co.uk",
                         "stackoverflow.com", "paruluniversity.ac.in", "python.org", "nytimes.com", "apple.com"])
    path = random.choice(["", "/", "/about", "/docs/guide", "/search?q=python", "/products/item/" + str(random.randint(1, 999)),
                          "/wiki/Machine_learning", "/news/2024/tech", "/help/contact"])
    sub = random.choice(["", "www.", "docs.", "mail."])
    return f"https://{sub}{dom}{path}"


rows = []
for _ in range(400):
    rows.append((fill(random.choice(legit_t)), 0))
    u = phish_url() if random.random() < 0.7 else "bit.ly/3xYz"
    rows.append((fill(random.choice(phish_t), u=u), 1))
pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=1).to_csv(DATA / "text.csv", index=False)

urows = [(legit_url(), 0) for _ in range(600)] + [(phish_url(), 1) for _ in range(600)]
pd.DataFrame(urows, columns=["url", "label"]).sample(frac=1, random_state=1).to_csv(DATA / "urls.csv", index=False)
print("Sample data written to", DATA)
