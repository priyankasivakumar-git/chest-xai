import { useState } from "react";

export default function MedicalAIDashboard() {

  // =========================
  // STATES
  // =========================
  const [selectedImage, setSelectedImage] = useState(null);

  const [preview, setPreview] = useState("");

  const [prediction, setPrediction] = useState("Waiting");

  const [confidence, setConfidence] = useState(0);

  const [scores, setScores] = useState({});

  const [heatmap, setHeatmap] = useState("");

  const [overlay, setOverlay] = useState("");

  const [shapValues, setShapValues] = useState({});

  const [loading, setLoading] = useState(false);

  // =========================
  // CLINICAL DATA STATES
  // =========================
  const [age, setAge] = useState(45);

  const [gender, setGender] = useState(1);

  const [fever, setFever] = useState(1);

  const [cough, setCough] = useState(0);

  // =========================
  // IMAGE UPLOAD
  // =========================
  const handleImageUpload = (e) => {

    const file = e.target.files[0];

    if (!file) return;

    setSelectedImage(file);

    setPreview(URL.createObjectURL(file));
  };

  // =========================
  // ANALYZE
  // =========================
  const handleAnalyze = async () => {

    if (!selectedImage) {
      alert("Please upload X-ray image");
      return;
    }

    setLoading(true);

    const formData = new FormData();

    // =========================
    // IMAGE
    // =========================
    formData.append("image", selectedImage);

    // =========================
    // CLINICAL DATA
    // =========================
    formData.append("age", age);

    formData.append("gender", gender);

    formData.append("fever", fever);

    formData.append("cough", cough);

    try {

      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      console.log(data);

      // =========================
      // PREDICTION
      // =========================
      setPrediction(data.prediction);

      setConfidence(data.confidence);

      setScores(data.scores);

      // =========================
      // GRAD CAM
      // =========================
      setHeatmap(
        `data:image/png;base64,${data.heatmap}`
      );

      setOverlay(
        `data:image/png;base64,${data.overlay}`
      );

      // =========================
      // SHAP
      // =========================
      setShapValues(data.shap);

    } catch (error) {

      console.error(error);

      alert("Prediction failed");

    }

    setLoading(false);
  };

  // =========================
  // AI INTERPRETATION
  // =========================
  const getInterpretation = () => {

    if (prediction === "COVID") {
      return "Radiological analysis indicates abnormal pulmonary opacity patterns with strong activation in infected lung regions suggesting COVID-related infection.";
    }

    if (prediction === "PNEUMONIA") {
      return "Dense pulmonary regions indicate possible pneumonia infection with inflammatory lung involvement.";
    }

    if (prediction === "Lung_Opacity") {
      return "Opacity regions detected indicating abnormal lung density and potential pulmonary abnormalities.";
    }

    if (prediction === "TUBERCULOSIS") {
      return "Upper lung activation patterns suggest possible tuberculosis-related infection.";
    }

    if (prediction === "NORMAL") {
      return "No significant pulmonary abnormalities detected. Chest X-ray appears normal.";
    }

    return "Waiting for AI analysis...";
  };

  return (

    <div className="min-h-screen bg-slate-100 p-6 font-sans">

      {/* ========================= */}
      {/* HEADER */}
      {/* ========================= */}
      <div className="bg-white rounded-3xl shadow-md p-6 mb-6 border border-slate-200">

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">

          <div>

            <h1 className="text-4xl font-bold text-slate-800">
              Pulmonary Disease Analysis System
            </h1>

          </div>

          <div className="flex gap-3">

            {/* UPLOAD */}
            <label className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-xl shadow-sm transition-all cursor-pointer">

              Upload X-ray

              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />

            </label>

            {/* ANALYZE */}
            <button
              onClick={handleAnalyze}
              className="bg-slate-800 hover:bg-slate-900 text-white px-5 py-2 rounded-xl shadow-sm transition-all"
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>

          </div>
        </div>

        {/* ========================= */}
        {/* CLINICAL INPUTS */}
        {/* ========================= */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">

          {/* AGE */}
          <div>

            <label className="text-sm font-medium text-slate-600">
              Age
            </label>

            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="w-full mt-2 p-3 rounded-xl border border-slate-200 outline-none"
            />

          </div>

          {/* GENDER */}
          <div>

            <label className="text-sm font-medium text-slate-600">
              Gender
            </label>

            <select
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              className="w-full mt-2 p-3 rounded-xl border border-slate-200 outline-none"
            >
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>

          </div>

          {/* FEVER */}
          <div>

            <label className="text-sm font-medium text-slate-600">
              Fever
            </label>

            <select
              value={fever}
              onChange={(e) => setFever(e.target.value)}
              className="w-full mt-2 p-3 rounded-xl border border-slate-200 outline-none"
            >
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>

          </div>

          {/* COUGH */}
          <div>

            <label className="text-sm font-medium text-slate-600">
              Cough
            </label>

            <select
              value={cough}
              onChange={(e) => setCough(e.target.value)}
              className="w-full mt-2 p-3 rounded-xl border border-slate-200 outline-none"
            >
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>

          </div>

        </div>
      </div>

      {/* ========================= */}
      {/* MAIN GRID */}
      {/* ========================= */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ========================= */}
        {/* LEFT SIDE */}
        {/* ========================= */}
        <div className="lg:col-span-2 space-y-6">

          {/* XRAY VISUALIZATION */}
          <div className="bg-white rounded-3xl shadow-md border border-slate-200 p-6">

            <div className="flex items-center justify-between mb-5">

              <h2 className="text-2xl font-semibold text-slate-800">
                X-ray Visualization
              </h2>

              <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full">
                Grad-CAM Enabled
              </span>

            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">

              {/* ORIGINAL */}
              <div>

                <h3 className="text-sm font-medium text-slate-600 mb-2">
                  Original X-ray
                </h3>

                <div className="bg-slate-100 rounded-2xl overflow-hidden border border-slate-200 h-72 flex items-center justify-center">

                  {preview ? (

                    <img
                      src={preview}
                      alt="Xray"
                      className="w-full h-full object-cover"
                    />

                  ) : (

                    <p className="text-slate-400">
                      Upload X-ray
                    </p>

                  )}

                </div>
              </div>

              {/* HEATMAP */}
              <div>

                <h3 className="text-sm font-medium text-slate-600 mb-2">
                  Heatmap
                </h3>

                <div className="bg-slate-100 rounded-2xl overflow-hidden border border-slate-200 h-72 flex items-center justify-center">

                  {heatmap ? (

                    <img
                      src={heatmap}
                      alt="Heatmap"
                      className="w-full h-full object-cover"
                    />

                  ) : (

                    <p className="text-slate-400">
                      Grad-CAM Heatmap
                    </p>

                  )}

                </div>
              </div>

              {/* OVERLAY */}
              <div>

                <h3 className="text-sm font-medium text-slate-600 mb-2">
                  Affected Region Localization
                </h3>

                <div className="bg-slate-100 rounded-2xl overflow-hidden border border-slate-200 h-72 flex items-center justify-center relative">

                  {overlay ? (

                    <img
                      src={overlay}
                      alt="Overlay"
                      className="w-full h-full object-cover"
                    />

                  ) : (

                    <p className="text-slate-400">
                      AI Overlay Result
                    </p>

                  )}

                </div>
              </div>
            </div>
          </div>

          {/* SHAP */}
          <div className="bg-white rounded-3xl shadow-md border border-slate-200 p-6">

            <div className="flex items-center justify-between mb-5">

              <h2 className="text-2xl font-semibold text-slate-800">
                Clinical Feature Importance (SHAP)
              </h2>

              <span className="text-sm bg-orange-100 text-orange-700 px-3 py-1 rounded-full">
                Explainable AI
              </span>

            </div>

            <div className="space-y-5">

              {Object.entries(shapValues).map(([feature, value]) => (

                <div key={feature}>

                  <div className="flex justify-between mb-2 text-sm">

                    <span className="font-medium text-slate-700">
                      {feature}
                    </span>

                    <span className="text-slate-500">
                      {value}%
                    </span>

                  </div>

                  <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">

                    <div
                      className="bg-orange-500 h-full rounded-full"
                      style={{ width: `${value}%` }}
                    ></div>

                  </div>

                </div>

              ))}
            </div>
          </div>
        </div>

        {/* RIGHT SIDE */}
        <div className="space-y-6">

          {/* RESULT */}
          <div className="bg-white rounded-3xl shadow-md border border-slate-200 p-6">

            <div className="flex items-center justify-between mb-5">

              <h2 className="text-2xl font-semibold text-slate-800">
                Prediction Result
              </h2>

            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 mb-5">

              <p className="text-slate-500 text-sm">
                Predicted Disease
              </p>

              <h1 className="text-4xl font-bold text-red-600 mt-2">
                {prediction}
              </h1>

              <p className="text-slate-500 mt-2">
                Confidence Score:
                <span className="font-semibold ml-1">
                  {confidence}%
                </span>
              </p>

            </div>

            {/* SCORES */}
            <div className="space-y-4">

              {Object.entries(scores).map(([label, score]) => (

                <div key={label}>

                  <div className="flex justify-between mb-1 text-sm">

                    <span className="font-medium text-slate-700">
                      {label}
                    </span>

                    <span className="text-slate-500">
                      {score.toFixed(1)}%
                    </span>

                  </div>

                  <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">

                    <div
                      className="bg-blue-600 h-full rounded-full"
                      style={{ width: `${score}%` }}
                    ></div>

                  </div>

                </div>

              ))}
            </div>
          </div>

          {/* INTERPRETATION */}
          <div className="bg-white rounded-3xl shadow-md border border-slate-200 p-6">

            <h2 className="text-2xl font-semibold text-slate-800 mb-4">
              AI Interpretation
            </h2>

            <div className="bg-blue-50 border border-blue-100 rounded-2xl p-5 text-slate-700 leading-relaxed">

              {getInterpretation()}

            </div>
          </div>
        </div>
      </div>
    </div>
  );
}