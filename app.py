import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (path is relative to this file so it works on Vercel too)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "medical_cost_predictor.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Model input order: age, sex, bmi, children, smoker, region
# Categorical values were label-encoded alphabetically during training:
SEX_MAP = {"female": 0, "male": 1}
SMOKER_MAP = {"no": 0, "yes": 1}
REGION_MAP = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
REGION_NAME = {
    "northeast": "North-East",
    "northwest": "North-West",
    "southeast": "South-East",
    "southwest": "South-West",
}

# The model predicts yearly charges in US dollars -> convert to Indian rupees.
# Update this rate whenever you want (1 USD = ? INR).
# Set it to 1.0 if you want to show raw model output without conversion.
USD_TO_INR = 88.0


def indian_format(n: float) -> str:
    """Format a number with Indian digit grouping, e.g. 4,65,56,100."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    last3, rest = s[-3:], s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last3])


def money(x: float) -> str:
    return "\u20b9" + indian_format(x)


def money_words(x: float) -> str:
    """Easy-to-read form: '₹3.88 Crore' / '₹45.20 Lakh'."""
    if x >= 1e7:
        return f"\u20b9{x / 1e7:.2f} Crore"
    if x >= 1e5:
        return f"\u20b9{x / 1e5:.2f} Lakh"
    return money(x)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "healthy weight"
    if bmi < 30:
        return "overweight"
    return "obese"


def predict_usd(age, sex, bmi, children, smoker, region) -> float:
    x = np.array([[age, SEX_MAP[sex], bmi, children, SMOKER_MAP[smoker], REGION_MAP[region]]], dtype=float)
    return float(model.predict(x)[0])


# ---------------------------------------------------------------------------
# Frontend (single page, light + dark theme)
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medical Cost Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f3f8f8;
    --bg-accent: radial-gradient(1200px 500px at 8% -10%, #d6f3ee 0%, transparent 60%),
                 radial-gradient(900px 500px at 100% 0%, #dbe8ff 0%, transparent 55%);
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --input-bg: #f8fafc;
    --primary: #0d9488;
    --primary-2: #0ea5e9;
    --primary-soft: #e6f7f5;
    --accent: #0f766e;
    --warn: #b45309;
    --warn-soft: #fffbeb;
    --danger: #dc2626;
    --danger-soft: #fef2f2;
    --shadow: 0 10px 30px rgba(15, 23, 42, .08);
  }
  :root[data-theme="dark"] {
    --bg: #0a1416;
    --bg-accent: radial-gradient(1200px 500px at 8% -10%, #0f2f2d 0%, transparent 60%),
                 radial-gradient(900px 500px at 100% 0%, #10264a 0%, transparent 55%);
    --card: #101d20;
    --text: #e6f1f2;
    --muted: #8fa9ad;
    --border: #1e3236;
    --input-bg: #0b1719;
    --primary: #2dd4bf;
    --primary-2: #38bdf8;
    --primary-soft: #11302e;
    --accent: #5eead4;
    --warn: #fbbf24;
    --warn-soft: #2b2310;
    --danger: #f87171;
    --danger-soft: #2a1618;
    --shadow: 0 10px 30px rgba(0, 0, 0, .5);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); background-image: var(--bg-accent); background-attachment: fixed;
    color: var(--text); min-height: 100vh; line-height: 1.5;
    transition: background .25s, color .25s;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 26px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #fff;
  }
  .brand h1 { font-size: 1.15rem; margin: 0; letter-spacing: -.01em; }
  .brand p { margin: 0; font-size: .8rem; color: var(--muted); }
  .theme-btn {
    border: 1px solid var(--border); background: var(--card); color: var(--text);
    width: 42px; height: 42px; border-radius: 12px; cursor: pointer; display: grid; place-items: center;
    box-shadow: var(--shadow); transition: transform .15s;
  }
  .theme-btn:hover { transform: translateY(-1px); }
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; align-items: start; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 24px; box-shadow: var(--shadow); transition: background .25s, border-color .25s;
  }
  .card h2 { margin: 0 0 4px; font-size: 1.05rem; }
  .card .sub { margin: 0 0 20px; color: var(--muted); font-size: .88rem; }
  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 520px) { .fields { grid-template-columns: 1fr; } }
  .field.full { grid-column: 1 / -1; }
  label.lbl { display: block; font-size: .82rem; font-weight: 600; margin-bottom: 6px; }
  label.lbl small { color: var(--muted); font-weight: 500; }
  input[type=number], select {
    width: 100%; padding: 11px 13px; font: inherit; color: var(--text);
    background: var(--input-bg); border: 1px solid var(--border); border-radius: 10px; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  input[type=number]:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-soft); }
  .seg { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 4px; background: var(--input-bg); border: 1px solid var(--border); border-radius: 12px; }
  .seg input { position: absolute; opacity: 0; pointer-events: none; }
  .seg label { text-align: center; padding: 8px 6px; border-radius: 9px; font-size: .88rem; font-weight: 600; cursor: pointer; color: var(--muted); transition: all .15s; }
  .seg input:checked + label { background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #fff; box-shadow: 0 4px 12px rgba(13,148,136,.3); }
  .seg input:focus-visible + label { outline: 2px solid var(--primary); outline-offset: 2px; }
  .bmi-chip { display: inline-block; margin-top: 6px; font-size: .75rem; font-weight: 600; padding: 2px 10px; border-radius: 999px; background: var(--primary-soft); color: var(--accent); }
  button.primary {
    margin-top: 22px; width: 100%; padding: 13px 16px; font: inherit; font-weight: 600; color: #fff;
    background: linear-gradient(135deg, var(--primary), var(--primary-2)); border: 0; border-radius: 12px; cursor: pointer;
    transition: transform .15s, opacity .15s; display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  button.primary:hover { transform: translateY(-1px); }
  button.primary:disabled { opacity: .65; cursor: wait; transform: none; }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; display: none; }
  button.loading .spinner { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .placeholder { text-align: center; padding: 34px 10px; color: var(--muted); }
  .placeholder svg { opacity: .55; margin-bottom: 10px; }
  .result { display: none; animation: fade .35s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .eyebrow { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
  .price { font-size: 2.4rem; font-weight: 700; letter-spacing: -.03em; color: var(--accent); margin: 4px 0 2px; }
  .range { color: var(--muted); font-size: .9rem; }
  .summary { margin: 18px 0 12px; padding: 14px 16px; border-radius: 12px; background: var(--primary-soft); font-size: .93rem; }
  .insight { margin: 0 0 14px; padding: 12px 14px; border-radius: 12px; background: var(--warn-soft); color: var(--warn); font-size: .88rem; display: none; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
  .stat b { display: block; font-size: 1rem; }
  .stat span { font-size: .75rem; color: var(--muted); }
  .error { display: none; margin-top: 16px; padding: 12px 14px; border-radius: 10px; background: var(--danger-soft); color: var(--danger); font-size: .88rem; }
  .note { margin-top: 16px; font-size: .75rem; color: var(--muted); }
  footer { text-align: center; color: var(--muted); font-size: .78rem; margin-top: 30px; }
</style>
<script>
  // Apply saved theme (or system preference) before paint to avoid flashing
  (function () {
    var saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) {}
    var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  })();
</script>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/></svg>
      </div>
      <div>
        <h1>Medical Cost Predictor</h1>
        <p>Estimate your yearly medical insurance cost</p>
      </div>
    </div>
    <button class="theme-btn" id="themeBtn" aria-label="Toggle theme" title="Toggle theme"></button>
  </header>

  <div class="grid">
    <form class="card" id="form" novalidate>
      <h2>Your details</h2>
      <p class="sub">Enter a few personal details to get an estimated yearly medical cost.</p>
      <div class="fields">
        <div class="field">
          <label class="lbl" for="age">Age <small>(years)</small></label>
          <input type="number" id="age" name="age" min="18" max="100" step="1" value="30" required>
        </div>
        <div class="field">
          <label class="lbl" for="bmi">BMI <small>(body mass index)</small></label>
          <input type="number" id="bmi" name="bmi" min="10" max="60" step="0.1" value="25" required>
          <span class="bmi-chip" id="bmiChip">Healthy weight</span>
        </div>
        <div class="field">
          <span class="lbl" style="display:block;font-size:.82rem;font-weight:600;margin-bottom:6px">Sex</span>
          <div class="seg">
            <input type="radio" name="sex" id="sexM" value="male" checked><label for="sexM">Male</label>
            <input type="radio" name="sex" id="sexF" value="female"><label for="sexF">Female</label>
          </div>
        </div>
        <div class="field">
          <span class="lbl" style="display:block;font-size:.82rem;font-weight:600;margin-bottom:6px">Do you smoke?</span>
          <div class="seg">
            <input type="radio" name="smoker" id="smN" value="no" checked><label for="smN">No</label>
            <input type="radio" name="smoker" id="smY" value="yes"><label for="smY">Yes</label>
          </div>
        </div>
        <div class="field">
          <label class="lbl" for="children">Children <small>(dependents)</small></label>
          <input type="number" id="children" name="children" min="0" max="10" step="1" value="0" required>
        </div>
        <div class="field">
          <label class="lbl" for="region">Region</label>
          <select id="region" name="region">
            <option value="northeast">North-East</option>
            <option value="northwest">North-West</option>
            <option value="southeast">South-East</option>
            <option value="southwest" selected>South-West</option>
          </select>
        </div>
      </div>
      <button class="primary" id="submitBtn" type="submit">
        <span class="spinner"></span><span id="btnText">Estimate cost</span>
      </button>
      <div class="error" id="error"></div>
    </form>

    <div class="card">
      <h2>Estimated cost</h2>
      <p class="sub">Your estimate will appear here.</p>

      <div class="placeholder" id="placeholder">
        <svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/></svg>
        <div>Fill in your details and press <b>Estimate cost</b>.</div>
      </div>

      <div class="result" id="result">
        <div class="eyebrow">Estimated yearly medical cost</div>
        <div class="price" id="price"></div>
        <div class="range" id="range"></div>
        <div class="summary" id="summary"></div>
        <div class="insight" id="insight"></div>
        <div class="stats">
          <div class="stat"><b id="monthly"></b><span>Approx. per month</span></div>
          <div class="stat"><b id="bmiCat"></b><span>BMI category</span></div>
        </div>
        <div class="note">This is an automated estimate based on historical insurance data. It is not medical or financial advice, and real costs depend on your insurer, plan and health.</div>
      </div>
    </div>
  </div>

  <footer>Built with Flask &amp; scikit-learn</footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const sun = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const moon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  function paintThemeIcon() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    $("themeBtn").innerHTML = dark ? sun : moon;
  }
  paintThemeIcon();
  $("themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    paintThemeIcon();
  });

  function bmiLabel(v) {
    if (!isFinite(v) || v <= 0) return "";
    if (v < 18.5) return "Underweight";
    if (v < 25) return "Healthy weight";
    if (v < 30) return "Overweight";
    return "Obese";
  }
  $("bmi").addEventListener("input", (e) => { $("bmiChip").textContent = bmiLabel(parseFloat(e.target.value)); });

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("error"); err.style.display = "none";
    const btn = $("submitBtn");
    btn.disabled = true; btn.classList.add("loading"); $("btnText").textContent = "Estimating...";

    const payload = {};
    new FormData($("form")).forEach((v, k) => payload[k] = v);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Something went wrong.");

      $("price").textContent = data.price;
      $("range").textContent = "Typical range: " + data.low + " – " + data.high + " per year";
      $("summary").textContent = data.summary;
      $("monthly").textContent = data.monthly;
      $("bmiCat").textContent = data.bmi_category;
      const ins = $("insight");
      if (data.insight) { ins.textContent = data.insight; ins.style.display = "block"; } else { ins.style.display = "none"; }
      $("placeholder").style.display = "none";
      const r = $("result"); r.style.display = "block";
      r.style.animation = "none"; r.offsetHeight; r.style.animation = "";
      if (window.innerWidth <= 820) r.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (ex) {
      err.textContent = ex.message; err.style.display = "block";
    } finally {
      btn.disabled = false; btn.classList.remove("loading"); $("btnText").textContent = "Estimate cost";
    }
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return HTML


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()

    def number(key, label, lo, hi):
        try:
            val = float(data.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"Please enter a valid number for '{label}'.")
        if not (lo <= val <= hi):
            raise ValueError(f"'{label}' must be between {lo} and {hi}.")
        return val

    try:
        age = number("age", "Age", 18, 100)
        bmi = number("bmi", "BMI", 10, 60)
        children = number("children", "Children", 0, 10)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400

    sex = str(data.get("sex", "")).lower()
    smoker = str(data.get("smoker", "")).lower()
    region = str(data.get("region", "")).lower()
    if sex not in SEX_MAP or smoker not in SMOKER_MAP or region not in REGION_MAP:
        return jsonify(ok=False, error="Please choose valid values for sex, smoker and region."), 400

    usd = predict_usd(age, sex, bmi, children, smoker, region)
    cost = usd * USD_TO_INR

    if cost <= 0:
        return jsonify(
            ok=False,
            error="These inputs are outside the range the model was trained on, so it "
                  "could not produce a realistic estimate. Please try more typical values.",
        ), 422

    low, high = cost * 0.80, cost * 1.20  # rough +/-20% band
    cat = bmi_category(bmi)
    is_smoker = smoker == "yes"

    # How much does smoking change this person's estimate?
    other = "no" if is_smoker else "yes"
    other_cost = predict_usd(age, sex, bmi, children, other, region) * USD_TO_INR
    diff = abs(cost - other_cost)
    if diff > 0:
        if is_smoker:
            insight = (
                f"Smoking adds roughly {money_words(diff)} per year to your estimate. "
                f"A non-smoker with the same profile would be around {money_words(max(other_cost, 0))}."
            )
        else:
            insight = (
                f"Being a non-smoker keeps your estimate low. A smoker with the same profile "
                f"would be around {money_words(other_cost)} per year, about {money_words(diff)} more."
            )
    else:
        insight = ""

    kids = int(children)
    article = "an" if int(age) in (8, 11, 18) or 80 <= int(age) <= 89 else "a"
    summary = (
        f"For {article} {int(age)}-year-old {sex} ({'smoker' if is_smoker else 'non-smoker'}) "
        f"with a BMI of {bmi:g} ({cat}), {kids} child{'ren' if kids != 1 else ''}, "
        f"living in the {REGION_NAME[region]} region, the estimated medical insurance cost is "
        f"about {money_words(cost)} per year, which is roughly {money_words(cost / 12)} per month."
    )
    if age > 64:
        summary += " Note: the model was trained on ages 18-64, so this is less reliable for your age."

    return jsonify(
        ok=True,
        price=money(cost),
        low=money_words(low),
        high=money_words(high),
        monthly=money(cost / 12),
        bmi_category=cat.capitalize(),
        insight=insight,
        summary=summary,
    )


if __name__ == "__main__":
    app.run(debug=True)
