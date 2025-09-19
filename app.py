from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# ------------------ Setup ------------------
load_dotenv()
gemini_api_key = os.getenv("API_KEY")
genai.configure(api_key=gemini_api_key)

app = Flask(__name__)
app.secret_key = "supersecret"  # session ke liye

SPECIALIST_CONTEXT = """Act as scenario simulator with provided data.
Your job is to recommend sustainable, circular, and metallurgical improvements. Just give answer in 2 lines, be direct.
"""

# Load ML model, scaler, encoders
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

# ------------------ Helpers ------------------
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

def create_base64_plot(fig):
    """Convert Matplotlib fig to base64 string for embedding in HTML"""
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close(fig)
    return plot_url

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat_gemini", methods=["POST"])
def chat_gemini():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "Please enter a valid question."})

    try:
        # Load saved inputs + outputs from session
        inputs = session.get("inputs", {})
        output = session.get("output", {})
        last_context = session.get("last_context", "")

        if not inputs or not output:
            return jsonify({"answer": "Please run a prediction first to provide context."})

        # Prompt
        full_prompt = f"""
        {SPECIALIST_CONTEXT}

        Inputs:
            Material = {inputs.get("Material")}, Source% = {inputs.get("Source")}, Quantity = {inputs.get("Quantity")} tons, 
            Ore Grade = {inputs.get("Ore_Grade")}%, Energy = {inputs.get("Energy")}%, Transport Mode = {inputs.get("TransportMode")}, Distance = {inputs.get("Distance")} km,
            Smelting Efficiency = {inputs.get("SmeltingEff")}% 

        Outputs (ML Predictions):
        {output}

        Latest ML Prediction Context:
        {last_context}

        User Question:
        {query}
        """

        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(full_prompt)

        # Save this Q/A in session as context for continuity
        session["last_context"] = f"Q: {query}\nA: {response.text}"

        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Inputs
        Material = request.form["Material"]
        Source = float(request.form["Source_%"])
        Quantity = float(request.form["Quantity_tons"])
        Ore_Grade = float(request.form["Ore_Grade_%"])
        Energy = float(request.form["Energy_%"])
        TransportMode = request.form["Transport_Mode"]
        Distance = float(request.form["Distance_km"])
        SmeltingEff = float(request.form["Smelting_Efficiency_%"])

        # Encode categorical
        Material_enc = safe_encode(encoders["Metal"], Material)
        Transport_enc = safe_encode(encoders["Transport_Mode"], TransportMode)

        # Features
        features = np.array([[Material_enc, Source, Quantity, Ore_Grade, Energy,
                              Transport_enc, Distance, SmeltingEff]])
        features_scaled = scaler.transform(features)

        # ML Prediction
        preds = model.predict(features_scaled)[0]
        output = {
            "Carbon_Footprint_kgCO2": round(preds[0], 2),
            "Water_Use_m3": round(preds[1], 2),
            "Energy_Intensity_MJ": round(preds[2], 2),
            "Land_Disturbance_m2": round(preds[3], 2),
            "Reuse_%": round(preds[4], 2),
            "Recycle_%": round(preds[5], 2),
            "Global_Warming_Potential": round(preds[6], 2),
            "End_of_Life_Score": round(preds[7], 2)
        }

        # Save inputs + outputs in session for chat
        session["inputs"] = {
            "Material": Material, "Source": Source, "Quantity": Quantity,
            "Ore_Grade": Ore_Grade, "Energy": Energy, "TransportMode": TransportMode,
            "Distance": Distance, "SmeltingEff": SmeltingEff
        }
        session["output"] = output

        # ------------------ Charts ------------------
        charts = {}

        # 1. Bar Chart
        fig, ax = plt.subplots()
        sns.barplot(
            x=["Carbon", "Water", "Energy", "Land"],
            y=[
                output["Carbon_Footprint_kgCO2"],
                output["Water_Use_m3"],
                output["Energy_Intensity_MJ"],
                output["Land_Disturbance_m2"]
            ],
            ax=ax
        )
        ax.set_title("Impact Distribution")
        charts["bar"] = create_base64_plot(fig)

        # 2. Pie Chart (Reuse vs Recycle vs End of Life)
        fig, ax = plt.subplots()
        values = [output["Reuse_%"], output["Recycle_%"], output["End_of_Life_Score"]]
        labels = ["Reuse %", "Recycle %", "End of Life %"]
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title("Circular Economy Breakdown")
        charts["pie"] = create_base64_plot(fig)

        # 3. Radar Chart (Spider Plot)
        categories = ["Carbon", "Water", "Energy", "Land", "Reuse", "Recycle"]
        values = [
            output["Carbon_Footprint_kgCO2"] / 10,  # scaled for visualization
            output["Water_Use_m3"],
            output["Energy_Intensity_MJ"] / 20,
            output["Land_Disturbance_m2"] / 5,
            output["Reuse_%"],
            output["Recycle_%"]
        ]
        values += values[:1]  # close loop
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Radar Chart of Impacts")
        charts["radar"] = create_base64_plot(fig)

        # 4. Stacked Bar Chart
        fig, ax = plt.subplots()
        reuse = output["Reuse_%"]
        recycle = output["Recycle_%"]
        waste = 100 - reuse - recycle
        ax.bar("Lifecycle", reuse, label="Reuse %")
        ax.bar("Lifecycle", recycle, bottom=reuse, label="Recycle %")
        ax.bar("Lifecycle", waste, bottom=reuse + recycle, label="Waste %")
        ax.legend()
        ax.set_title("Reuse vs Recycle vs Waste")
        charts["stacked"] = create_base64_plot(fig)

        # ------------------ Gemini Recommendations ------------------
        try:
            context = f"""
            Inputs:
            Material = {Material}, Source% = {Source}, Quantity = {Quantity} tons, 
            Ore Grade = {Ore_Grade}%, Energy = {Energy}%, Transport Mode = {TransportMode}, Distance = {Distance} km,
            Smelting Efficiency = {SmeltingEff}%

            Outputs (ML Predictions):
            {output}

            Task: Give exactly 2 practical recommendations to improve circularity across full value chain.
            """
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            rec_response = model_gemini.generate_content(context)
            recommendations = rec_response.text.strip()
        except Exception as e:
            recommendations = f"Could not fetch recommendations: {str(e)}"

        return render_template("index.html", prediction=output, recommendations=recommendations, charts=charts)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
