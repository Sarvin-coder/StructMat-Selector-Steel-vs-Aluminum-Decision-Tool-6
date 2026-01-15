import streamlit as st
import pandas as pd

# =========================================================
# Steel vs Aluminium Structural Material Selection Tool
# Interactive + Charts + Final Scoring System
# =========================================================

st.set_page_config(page_title="Steel vs Aluminium Tool", page_icon="⚙️", layout="centered")

# ---------- Material property data (default, editable) ----------
steel_default = {
    "name": "Steel",
    "tensile_strength": 400.0,      # MPa
    "yield_strength": 250.0,        # MPa
    "E": 200.0,                     # GPa
    "density": 7850.0,              # kg/m^3
    "cost'sc": 3.0,                 # RM/kg
    "corrosion_rating": 2           # /5
}

al_default = {
    "name": "Aluminium",
    "tensile_strength": 310.0,      # MPa
    "yield_strength": 275.0,        # MPa
    "E": 69.0,                      # GPa
    "density": 2700.0,              # kg/m^3
    "cost'sc": 12.0,                # RM/kg
    "corrosion_rating": 5           # /5
}

# ---------- UI Header ----------
st.title("⚙️ Steel vs Aluminium Structural Material Selection Tool")
st.caption("Interactive Streamlit App | User Inputs + Engineering Calculations + Charts + Scoring")

# ---------- Sidebar: Editable material properties ----------
st.sidebar.header("🔧 Material Properties (Editable)")

with st.sidebar.expander("Steel", expanded=True):
    steel_ts = st.number_input("Steel Tensile Strength (MPa)", min_value=0.0, value=steel_default["tensile_strength"])
    steel_ys = st.number_input("Steel Yield Strength (MPa)", min_value=0.0, value=steel_default["yield_strength"])
    steel_E  = st.number_input("Steel Young’s Modulus (GPa)", min_value=0.0, value=steel_default["E"])
    steel_rho = st.number_input("Steel Density (kg/m³)", min_value=0.0, value=steel_default["density"])
    steel_cost = st.number_input("Steel Cost (RM/kg)", min_value=0.0, value=steel_default["cost'sc"])
    steel_corr = st.slider("Steel Corrosion Rating (1–5)", 1, 5, steel_default["corrosion_rating"])

with st.sidebar.expander("Aluminium", expanded=True):
    al_ts = st.number_input("Aluminium Tensile Strength (MPa)", min_value=0.0, value=al_default["tensile_strength"])
    al_ys = st.number_input("Aluminium Yield Strength (MPa)", min_value=0.0, value=al_default["yield_strength"])
    al_E  = st.number_input("Aluminium Young’s Modulus (GPa)", min_value=0.0, value=al_default["E"])
    al_rho = st.number_input("Aluminium Density (kg/m³)", min_value=0.0, value=al_default["density"])
    al_cost = st.number_input("Aluminium Cost (RM/kg)", min_value=0.0, value=al_default["cost'sc"])
    al_corr = st.slider("Aluminium Corrosion Rating (1–5)", 1, 5, al_default["corrosion_rating"])

steel = {
    "name": "Steel",
    "tensile_strength": float(steel_ts),
    "yield_strength": float(steel_ys),
    "E": float(steel_E),
    "density": float(steel_rho),
    "cost": float(steel_cost),
    "corrosion_rating": int(steel_corr)
}
aluminum = {
    "name": "Aluminium",
    "tensile_strength": float(al_ts),
    "yield_strength": float(al_ys),
    "E": float(al_E),
    "density": float(al_rho),
    "cost": float(al_cost),
    "corrosion_rating": int(al_corr)
}

# ---------- Helper: table ----------
def material_table():
    df = pd.DataFrame({
        "Property": [
            "Tensile Strength (MPa)", "Yield Strength (MPa)", "Young’s Modulus (GPa)",
            "Density (kg/m³)", "Cost (RM/kg)", "Corrosion Rating (/5)"
        ],
        "Steel": [
            steel["tensile_strength"], steel["yield_strength"], steel["E"],
            steel["density"], steel["cost"], steel["corrosion_rating"]
        ],
        "Aluminium": [
            aluminum["tensile_strength"], aluminum["yield_strength"], aluminum["E"],
            aluminum["density"], aluminum["cost"], aluminum["corrosion_rating"]
        ]
    })
    return df

with st.expander("📊 View Material Property Table"):
    st.dataframe(material_table(), hide_index=True)

st.markdown("---")

# =========================================================
# Engineering formulas (Beam: Simply Supported + UDL)
# =========================================================
def beam_udl_results(L_m, w_kN_m, b_m, h_m, mat):
    # Convert UDL to N/m
    w = w_kN_m * 1000.0

    # Step 1: Mmax = wL^2 / 8
    Mmax = (w * (L_m ** 2)) / 8.0  # N·m

    # Step 2: I = bh^3 / 12
    I = (b_m * (h_m ** 3)) / 12.0  # m^4

    # Step 3: sigma = M c / I, c = h/2
    c = h_m / 2.0
    sigma_Pa = (Mmax * c) / I
    sigma_MPa = sigma_Pa / 1e6

    # Step 4: FOS = sigma_y / sigma
    sigma_y = mat["yield_strength"]  # MPa
    FOS = sigma_y / sigma_MPa if sigma_MPa > 0 else float("inf")

    # Step 5: deflection = 5 w L^4 / (384 E I)
    E = mat["E"] * 1e9
    delta_m = (5.0 * w * (L_m ** 4)) / (384.0 * E * I)
    delta_mm = delta_m * 1000.0

    # Step 6: deflection limit = L/360
    delta_allow_mm = (L_m / 360.0) * 1000.0
    pass_deflection = delta_mm <= delta_allow_mm

    # Step 7: Volume
    V = b_m * h_m * L_m

    # Step 8: Mass
    mass = mat["density"] * V

    # Step 9: Cost
    cost_total = mass * mat["cost"]

    return {
        "Mmax (N·m)": Mmax,
        "I (m⁴)": I,
        "Stress (MPa)": sigma_MPa,
        "FOS": FOS,
        "Deflection (mm)": delta_mm,
        "Allowable (mm)": delta_allow_mm,
        "Serviceability PASS": pass_deflection,
        "Volume (m³)": V,
        "Mass (kg)": mass,
        "Cost (RM)": cost_total
    }

# =========================================================
# Scoring system (user-adjustable weights)
# =========================================================
def normalize_benefit(x, xmin, xmax):
    if xmax - xmin == 0:
        return 1.0
    return (x - xmin) / (xmax - xmin)

def normalize_cost(x, xmin, xmax):
    # lower is better
    if xmax - xmin == 0:
        return 1.0
    return (xmax - x) / (xmax - xmin)

def compute_scores(steel_metrics, alu_metrics, w_strength, w_weight, w_cost, w_corr):
    # Use Yield Strength for "strength"
    s_strength = steel["yield_strength"]
    a_strength = aluminum["yield_strength"]

    # Weight proxy = mass from beam case if available; else density
    s_mass = steel_metrics.get("Mass (kg)", steel["density"])
    a_mass = alu_metrics.get("Mass (kg)", aluminum["density"])

    # Cost proxy = total cost from beam case if available; else cost/kg
    s_cost = steel_metrics.get("Cost (RM)", steel["cost"])
    a_cost = alu_metrics.get("Cost (RM)", aluminum["cost"])

    # Corrosion rating
    s_corr = steel["corrosion_rating"]
    a_corr = aluminum["corrosion_rating"]

    # Normalize each criterion between the two materials
    # Strength (benefit)
    str_min, str_max = min(s_strength, a_strength), max(s_strength, a_strength)
    steel_str = normalize_benefit(s_strength, str_min, str_max)
    alu_str = normalize_benefit(a_strength, str_min, str_max)

    # Weight/Mass (cost-like; lower is better)
    mass_min, mass_max = min(s_mass, a_mass), max(s_mass, a_mass)
    steel_wt = normalize_cost(s_mass, mass_min, mass_max)
    alu_wt = normalize_cost(a_mass, mass_min, mass_max)

    # Cost (lower is better)
    cost_min, cost_max = min(s_cost, a_cost), max(s_cost, a_cost)
    steel_c = normalize_cost(s_cost, cost_min, cost_max)
    alu_c = normalize_cost(a_cost, cost_min, cost_max)

    # Corrosion (benefit)
    corr_min, corr_max = min(s_corr, a_corr), max(s_corr, a_corr)
    steel_cor = normalize_benefit(s_corr, corr_min, corr_max)
    alu_cor = normalize_benefit(a_corr, corr_min, corr_max)

    # Weighted score (0–100)
    total_w = w_strength + w_weight + w_cost + w_corr
    if total_w == 0:
        total_w = 1.0

    steel_score = (steel_str * w_strength + steel_wt * w_weight + steel_c * w_cost + steel_cor * w_corr) / total_w
    alu_score = (alu_str * w_strength + alu_wt * w_weight + alu_c * w_cost + alu_cor * w_corr) / total_w

    return steel_score * 100.0, alu_score * 100.0

# =========================================================
# Apps 1–5 (updated to be interactive + charts + scoring)
# =========================================================
def app1_beam_udl():
    st.subheader("Application 1: Beam Design (Simply Supported Beam + UDL)")
    st.write("Enter beam & load values → the app computes stress, FOS, deflection, mass, cost for **both** materials.")

    c1, c2 = st.columns(2)
    with c1:
        L = st.number_input("Beam span L (m)", value=6.0, min_value=0.1)
        w = st.number_input("UDL w (kN/m)", value=12.0, min_value=0.0)
    with c2:
        b = st.number_input("Width b (m)", value=0.10, min_value=0.001)
        h = st.number_input("Height h (m)", value=0.20, min_value=0.001)

    st.markdown("#### Set decision weights (Scoring System)")
    w_strength = st.slider("Strength importance", 0, 10, 5)
    w_weight   = st.slider("Weight importance", 0, 10, 5)
    w_cost     = st.slider("Cost importance", 0, 10, 5)
    w_corr     = st.slider("Corrosion importance", 0, 10, 2)

    if st.button("Run Beam Design Calculator ✅"):
        s_res = beam_udl_results(L, w, b, h, steel)
        a_res = beam_udl_results(L, w, b, h, aluminum)

        # Table output
        df = pd.DataFrame({
            "Metric": list(s_res.keys()),
            "Steel": list(s_res.values()),
            "Aluminium": list(a_res.values())
        })
        st.dataframe(df, hide_index=True)

        # Key metrics cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Steel Deflection (mm)", f"{s_res['Deflection (mm)']:.2f}")
        k2.metric("Aluminium Deflection (mm)", f"{a_res['Deflection (mm)']:.2f}")
        k3.metric("Steel Cost (RM)", f"{s_res['Cost (RM)']:.0f}")
        k4.metric("Aluminium Cost (RM)", f"{a_res['Cost (RM)']:.0f}")

        # Charts (auto Streamlit colors)
        st.markdown("### 📊 Comparison Charts")
        chart_df = pd.DataFrame({
            "Steel": {
                "Stress (MPa)": s_res["Stress (MPa)"],
                "FOS": s_res["FOS"],
                "Deflection (mm)": s_res["Deflection (mm)"],
                "Mass (kg)": s_res["Mass (kg)"],
                "Cost (RM)": s_res["Cost (RM)"]
            },
            "Aluminium": {
                "Stress (MPa)": a_res["Stress (MPa)"],
                "FOS": a_res["FOS"],
                "Deflection (mm)": a_res["Deflection (mm)"],
                "Mass (kg)": a_res["Mass (kg)"],
                "Cost (RM)": a_res["Cost (RM)"]
            }
        })
        st.bar_chart(chart_df)

        # Serviceability decision
        st.markdown("### ✅ Serviceability Check (Deflection Limit L/360)")
        if s_res["Serviceability PASS"] and not a_res["Serviceability PASS"]:
            st.success("✅ Aluminium fails deflection, so **STEEL** is recommended for this beam case.")
        elif a_res["Serviceability PASS"] and not s_res["Serviceability PASS"]:
            st.success("✅ Steel fails deflection, so **ALUMINIUM** is recommended for this beam case.")
        else:
            st.info("Both materials have the same pass/fail deflection status. We use scoring to decide overall best choice.")

        # Scoring decision
        st.markdown("### 🧮 Final Scoring Decision (User-Weighted)")
        steel_score, alu_score = compute_scores(s_res, a_res, w_strength, w_weight, w_cost, w_corr)
        s_col, a_col = st.columns(2)
        s_col.metric("Steel Score", f"{steel_score:.1f} / 100")
        a_col.metric("Aluminium Score", f"{alu_score:.1f} / 100")

        if steel_score > alu_score:
            st.success("🏆 Overall Winner: **STEEL** (based on your chosen weights)")
        elif alu_score > steel_score:
            st.success("🏆 Overall Winner: **ALUMINIUM** (based on your chosen weights)")
        else:
            st.warning("Both materials have the same score based on your weights.")


def app2_weight():
    st.subheader("Application 2: Weight-Based Selection (User Input)")
    st.write("Enter geometry → the app calculates mass for steel and aluminium and recommends the lighter one.")

    L = st.number_input("Length L (m)", value=6.0, min_value=0.1, key="wL")
    b = st.number_input("Width b (m)", value=0.10, min_value=0.001, key="wb")
    h = st.number_input("Height h (m)", value=0.20, min_value=0.001, key="wh")

    if st.button("Run Weight Comparison ✅"):
        V = b * h * L
        ms = steel["density"] * V
        ma = aluminum["density"] * V

        st.metric("Steel Mass (kg)", f"{ms:.2f}")
        st.metric("Aluminium Mass (kg)", f"{ma:.2f}")

        st.bar_chart(pd.DataFrame({"Mass (kg)": {"Steel": ms, "Aluminium": ma}}))

        if ma < ms:
            st.success("✅ Recommendation: **ALUMINIUM** (lighter → better when weight reduction is needed).")
        else:
            st.success("✅ Recommendation: **STEEL** (lighter in this case).")


def app3_cost():
    st.subheader("Application 3: Cost-Based Selection (User Input)")
    st.write("Enter geometry → app estimates cost using mass × cost/kg and recommends the cheaper one.")

    L = st.number_input("Length L (m)", value=6.0, min_value=0.1, key="cL")
    b = st.number_input("Width b (m)", value=0.10, min_value=0.001, key="cb")
    h = st.number_input("Height h (m)", value=0.20, min_value=0.001, key="ch")

    if st.button("Run Cost Comparison ✅"):
        V = b * h * L
        steel_cost_total = (steel["density"] * V) * steel["cost"]
        alu_cost_total = (aluminum["density"] * V) * aluminum["cost"]

        st.metric("Steel Cost (RM)", f"{steel_cost_total:.2f}")
        st.metric("Aluminium Cost (RM)", f"{alu_cost_total:.2f}")

        st.bar_chart(pd.DataFrame({"Cost (RM)": {"Steel": steel_cost_total, "Aluminium": alu_cost_total}}))

        if steel_cost_total < alu_cost_total:
            st.success("✅ Recommendation: **STEEL** (more cost-effective).")
        else:
            st.success("✅ Recommendation: **ALUMINIUM** (more cost-effective).")


def app4_corrosion():
    st.subheader("Application 4: Corrosion Resistance Selection (Interactive)")
    st.write("Select environment → app recommends the better material for corrosion exposure.")

    env = st.selectbox("Environment type:", ["Indoor (dry)", "Outdoor (normal)", "Coastal / highly corrosive"])

    if st.button("Run Corrosion Recommendation ✅"):
        st.bar_chart(pd.DataFrame({
            "Corrosion Rating (/5)": {"Steel": steel["corrosion_rating"], "Aluminium": aluminum["corrosion_rating"]}
        }))

        if env == "Coastal / highly corrosive":
            st.success("✅ Recommendation: **ALUMINIUM** (higher corrosion resistance for harsh environments).")
        else:
            st.success("✅ Recommendation: **STEEL** (cost-effective; corrosion can be managed with coating/paint).")


def app5_element():
    st.subheader("Application 5: Structural Element Recommendation (With Priority)")
    st.write("Select the element and your priority → the app recommends steel or aluminium with a reason.")

    element = st.selectbox("Structural element:", ["Beam", "Column", "Slab", "Truss", "Frame"])
    priority = st.selectbox("Main priority:", ["High Strength/Stiffness", "Low Cost", "Low Weight", "High Corrosion Resistance"])

    if st.button("Run Element Recommendation ✅"):
        if priority == "Low Weight":
            st.success("✅ Recommendation: **ALUMINIUM** (lighter material is preferred).")
        elif priority == "High Corrosion Resistance":
            st.success("✅ Recommendation: **ALUMINIUM** (better corrosion resistance).")
        elif priority == "Low Cost":
            st.success("✅ Recommendation: **STEEL** (cheaper for most structural works).")
        else:
            # Strength/Stiffness
            if element in ["Beam", "Column", "Slab"]:
                st.success("✅ Recommendation: **STEEL** (higher stiffness → better deflection control).")
            else:
                st.success("✅ Recommendation: **ALUMINIUM** (lightweight advantage often useful in trusses/frames).")

# =========================================================
# Cleaner UI: Tabs + Module selector
# =========================================================
tabs = st.tabs(["🏠 App Modules", "🧮 Scoring Only (No Beam)", "ℹ️ How to Explain"])

with tabs[0]:
    module = st.selectbox(
        "Select an Application Module:",
        [
            "Application 1: Beam Design (UDL Calculator + Charts + Scoring)",
            "Application 2: Weight-Based Selection",
            "Application 3: Cost-Based Selection",
            "Application 4: Corrosion Resistance Selection",
            "Application 5: Structural Element Recommendation"
        ]
    )
    st.markdown("---")

    if module.startswith("Application 1"):
        app1_beam_udl()
    elif module.startswith("Application 2"):
        app2_weight()
    elif module.startswith("Application 3"):
        app3_cost()
    elif module.startswith("Application 4"):
        app4_corrosion()
    else:
        app5_element()

with tabs[1]:
    st.subheader("Scoring System Only (General Decision)")
    st.write("This is a general scoring tool using material properties (yield strength, density, cost/kg, corrosion rating).")

    w_strength = st.slider("Strength importance", 0, 10, 5, key="s_strength")
    w_weight   = st.slider("Weight importance", 0, 10, 5, key="s_weight")
    w_cost     = st.slider("Cost importance", 0, 10, 5, key="s_cost")
    w_corr     = st.slider("Corrosion importance", 0, 10, 2, key="s_corr")

    # Here we use density + cost/kg (no geometry)
    steel_metrics = {"Mass (kg)": steel["density"], "Cost (RM)": steel["cost"]}
    alu_metrics = {"Mass (kg)": aluminum["density"], "Cost (RM)": aluminum["cost"]}

    steel_score, alu_score = compute_scores(steel_metrics, alu_metrics, w_strength, w_weight, w_cost, w_corr)
    c1, c2 = st.columns(2)
    c1.metric("Steel Score", f"{steel_score:.1f} / 100")
    c2.metric("Aluminium Score", f"{alu_score:.1f} / 100")

    st.bar_chart(pd.DataFrame({"Score": {"Steel": steel_score, "Aluminium": alu_score}}))

    if steel_score > alu_score:
        st.success("🏆 Winner: **STEEL** (based on your chosen weights)")
    elif alu_score > steel_score:
        st.success("🏆 Winner: **ALUMINIUM** (based on your chosen weights)")
    else:
        st.warning("Both materials have equal score based on your chosen weights.")

with tabs[2]:
    st.subheader("How to Explain During Presentation (Short Script)")
    st.write(
        "“This is our Streamlit web app. Users select an application module, enter design inputs, and the app "
        "calculates engineering outputs instantly. For example, in the beam calculator, it computes bending moment, "
        "stress, factor of safety, deflection, deflection limit, mass, and cost for both steel and aluminium. "
        "Then it visualizes comparisons using charts and provides a final recommendation. We also included a scoring "
        "system where users can set priorities like strength, weight, cost, and corrosion to get a weighted final decision.”"
    )

st.markdown("---")
st.caption("KKCE1112 | Streamlit + Python Engineering Tool | Interactive Inputs + Charts + Weighted Scoring")
