import os
from flask import Flask, request, jsonify, render_template

from agri_agent import (
    run_agent_from_structured_inputs,
    run_agent_from_text,
    CROP_KNOWLEDGE_BASE,
)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/agent")
    def api_agent():
        data = request.get_json(force=True, silent=True) or {}
        mode = (data.get("mode") or "structured").strip().lower()
        if mode == "text":
            user_text = data.get("user_text") or ""
            final_state = run_agent_from_text(user_text)
        else:
            crop = data.get("crop") or ""
            soil_type = data.get("soil_type") or ""
            soil_ph = data.get("soil_ph")
            soil_moisture = data.get("soil_moisture")
            n = data.get("n")
            p = data.get("p")
            k = data.get("k")
            final_state = run_agent_from_structured_inputs(
                crop, soil_type, soil_ph, soil_moisture, n, p, k
            )
        return jsonify({
            "crop_name": final_state.get("crop_name"),
            "optimized_values": final_state.get("optimized_values"),
            "remedies_text": final_state.get("remedies_text"),
        })

    @app.post("/submit")
    def submit_form():
        crop = request.form.get("crop", "")
        soil_type = request.form.get("soil_type", "")
        soil_ph = request.form.get("soil_ph", "")
        soil_moisture = request.form.get("soil_moisture", "")
        n = request.form.get("n", "")
        p = request.form.get("p", "")
        k = request.form.get("k", "")

        final_state = run_agent_from_structured_inputs(
            crop, soil_type, soil_ph, soil_moisture, n, p, k
        )
        remedies_text = (final_state.get("remedies_text") or "").replace("**", "")
        optimized = final_state.get("optimized_values") or {}

        # Try to pull KB tips for detected/entered crop
        crop_key = (final_state.get("crop_name") or crop or "").lower().strip()
        kb = CROP_KNOWLEDGE_BASE.get(crop_key, {})
        context = {
            "crop": crop_key or "unknown",
            "remedies_text": remedies_text,
            "optimized": optimized,
            "fertilizer_tips": kb.get("fertilizer_tips", "N/A"),
            "pest_control": kb.get("pest_control", "N/A"),
            "plan": kb.get("plan", "N/A"),
            "gov_schemes": kb.get("gov_schemes", "N/A"),
        }
        return render_template("result.html", **context)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)


