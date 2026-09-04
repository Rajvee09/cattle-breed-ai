import { useState } from "react";
import "./App.css";

function App() {
  const [species, setSpecies] = useState("Auto");
  const [region, setRegion] = useState("");
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // ============================================================
  // IMAGE UPLOAD
  // ============================================================

  const handleImageUpload = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      setImage(URL.createObjectURL(selectedFile));
      setResult(null);
      setError("");
    }
  };

  // ============================================================
  // ANALYZE IMAGE
  // ============================================================

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please upload an animal image first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();

    // Send image to Flask
    formData.append("image", file);

    // Optional fields
    formData.append("species", species);
    formData.append("region", region);

    try {
      // IMPORTANT:
      // Do NOT put markdown/link syntax here.
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      console.log("BACKEND RESPONSE:", data);

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "Prediction failed."
        );
      }

      // ========================================================
      // GET PREDICTIONS FROM FLASK
      // ========================================================

      const predictions = data.predictions;

      if (!predictions || predictions.length === 0) {
        throw new Error(
          "Model returned no predictions."
        );
      }

      // Highest-confidence prediction
      const primary = predictions[0];

      // ========================================================
      // MAP BACKEND DATA TO FRONTEND
      // ========================================================

      setResult({
        primaryBreed: primary.breed,

        confidence: primary.confidence,

        // Actual breed profile information
        origin:
          primary.origin || "Information not available",

        type:
          primary.type || "Information not available",

        use:
          primary.use || "Information not available",

        traits:
          primary.features || [],

        // Display actual predicted category
        species:
          primary.type || "Indian Bovine",

        // Remaining model predictions
        alternatives: predictions
          .slice(1)
          .map((prediction) => ({
            breed: prediction.breed,
            confidence: prediction.confidence,
          })),
      });

    } catch (err) {
      console.error("Prediction failed:", err);

      setError(
        err.message ||
        "Unable to connect to the AI prediction server."
      );

    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app">

      {/* ================= NAVBAR ================= */}

      <header className="navbar">

        <div className="logo">
          <span>Bovine Breed Identifier</span>
        </div>

        <div className="tagline">
        </div>

      </header>


      {/* ================= MAIN CONTENT ================= */}

      <main className="main-content">

        {/* ================= HERO ================= */}

        <section className="hero">

          <p className="eyebrow">
          </p>

          <h1>
            Identify Indian
            <br />
            <span>Cattle & Buffalo Breeds</span>
          </h1>

          <p className="description">
          </p>

        </section>


        {/* ================= CLASSIFICATION CARD ================= */}

        <section className="card">

          <h2>Breed Classification</h2>

          <p className="section-description">
          </p>


          {/* ================= IMAGE UPLOAD ================= */}

          <div className="field">

            <label>
              Animal Photo
            </label>

            <label className="upload-box">

              {image ? (

                <img
                  src={image}
                  alt="Uploaded animal"
                  className="preview"
                />

              ) : (

                <>

                  <div className="upload-icon">
                  </div>

                  <strong>
                    Upload animal photo
                  </strong>

                  <span>
                    Click here
                  </span>

                </>

              )}

              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
              />

            </label>

          </div>


          {/* ================= SPECIES ================= */}

          <div className="field">

            <label>
              Species / Category{" "}
              <span className="optional-tag">
                (Optional)
              </span>
            </label>

            <div className="animal-options">

              {/* AUTO */}

              <button
                type="button"
                className={`animal-option ${
                  species === "Auto"
                    ? "selected"
                    : ""
                }`}
                onClick={() => setSpecies("Auto")}
              >

                <span>✨</span>

                <div>

                  <strong>
                    Auto-Detect
                  </strong>

                  <small>
                    Let AI detect
                  </small>

                </div>

              </button>


              {/* CATTLE */}

              <button
                type="button"
                className={`animal-option ${
                  species === "Cattle"
                    ? "selected"
                    : ""
                }`}
                onClick={() => setSpecies("Cattle")}
              >

                <span>🐄</span>

                <div>

                  <strong>
                    Cattle
                  </strong>

                </div>

              </button>


              {/* BUFFALO */}

              <button
                type="button"
                className={`animal-option ${
                  species === "Buffalo"
                    ? "selected"
                    : ""
                }`}
                onClick={() => setSpecies("Buffalo")}
              >

                <span>🐃</span>

                <div>

                  <strong>
                    Buffalo
                  </strong>

                </div>

              </button>

            </div>

          </div>


          {/* ================= REGION ================= */}

          <div className="field">

            <label>
              Region{" "}
              <span className="optional-tag">
                (Optional)
              </span>
            </label>

            <select
              value={region}
              onChange={(e) =>
                setRegion(e.target.value)
              }
            >

              <option value="">
                Auto / Not specified
              </option>

              <option value="North India">
                North India
              </option>

              <option value="South India">
                South India
              </option>

              <option value="West India">
                West India
              </option>

              <option value="East India">
                East India
              </option>

              <option value="Central India">
                Central India
              </option>

            </select>

          </div>


          {/* ================= ANALYZE BUTTON ================= */}

          <button
            className="analyze-button"
            disabled={!file || loading}
            onClick={handleAnalyze}
          >

            {loading
              ? "⏳ Analyzing Image..."
              : "🔍 Identify Breed"}

          </button>


          {/* ================= ERROR ================= */}

          {error && (

            <div className="error-message">

              <strong>
                Prediction Error
              </strong>

              <p>
                {error}
              </p>

              <small>
                Make sure the Flask server is running
                on port 5000.
              </small>

            </div>

          )}


          {/* ================= RESULTS ================= */}

          {result && (

            <div className="results-container">

              <hr className="divider" />

              <h3>
                Identification Result
              </h3>


              {/* ================= PRIMARY RESULT ================= */}

              <div className="primary-result">

                <div className="breed-header">

                  <h4>
                    {result.primaryBreed}
                  </h4>

                  <span className="confidence-badge">

                    {Number(
                      result.confidence
                    ).toFixed(2)}

                    % Match

                  </span>

                </div>


                {/* CATEGORY */}

                <p>

                  <strong>
                    Category:
                  </strong>{" "}

                  {result.species}

                </p>


                {/* ORIGIN */}

                <p>

                  <strong>
                    Origin:
                  </strong>{" "}

                  {result.origin}

                </p>


                {/* TYPE */}

                <p>

                  <strong>
                    Type:
                  </strong>{" "}

                  {result.type}

                </p>


                {/* USE */}

                <p>

                  <strong>
                    Primary Use:
                  </strong>{" "}

                  {result.use}

                </p>


                {/* FEATURES */}

                {result.traits && (

                  <div className="traits-list">

                    <strong>
                      Key Physical Features:
                    </strong>

                    <ul>

                      {Array.isArray(
                        result.traits
                      ) ? (

                        result.traits.map(
                          (trait, idx) => (

                            <li key={idx}>
                              {trait}
                            </li>

                          )
                        )

                      ) : (

                        <li>
                          {result.traits}
                        </li>

                      )}

                    </ul>

                  </div>

                )}

              </div>


              {/* ================= ALTERNATIVES ================= */}

              {result.alternatives &&
                result.alternatives.length > 0 && (

                  <div className="alternatives-section">

                    <h5>
                      Other Close Predictions
                    </h5>

                    <ul>

                      {result.alternatives.map(
                        (alt, idx) => (

                          <li key={idx}>

                            <span>
                              {alt.breed}
                            </span>

                            <span>

                              {Number(
                                alt.confidence
                              ).toFixed(2)}

                              %

                            </span>

                          </li>

                        )
                      )}

                    </ul>

                  </div>

                )}

            </div>

          )}


          {/* ================= PRIVACY ================= */}

          <p className="privacy-note">
          </p>

        </section>

      </main>

    </div>
  );
}

export default App;