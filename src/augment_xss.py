import pandas as pd
import random

xss_payloads = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>",
    "\"><script>alert(1)</script>",
    "<iframe src=javascript:alert(1)>",
]

def make_xss_samples(n=20):
    rows = []
    for _ in range(n):
        payload = random.choice(xss_payloads)
        url = f"http://localhost:8080/tienda1/publico/buscar.jsp?search={payload}"
        rows.append(["get", url, "", 1])
    return rows

df = pd.read_csv("data/csic_cleaned.csv")
extra = pd.DataFrame(make_xss_samples(), columns=["Method","URL","content","classification"])
df2 = pd.concat([df, extra], ignore_index=True)
df2.to_csv("data/csic_cleaned.csv", index=False)

print("Added synthetic XSS samples!")
