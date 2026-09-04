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

  const handleImageUpload = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      setImage(URL.createObjectURL(selectedFile));
      setResult(null);
      setError("");
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();

    // Send the actual image to Flask
    formData.append("image", file);

    // These are optional and currently don't affect
    // the ConvNeXt model prediction.
    formData.append("species", species);
    formData.append("region", region);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      console.log("BACKEND RESPONSE:", data);

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Prediction failed");
      }

      /*
       * Flask returns:
       *
       * {
       *   success: true,
       *   predictions: [...]
       * }
       *
       * The first prediction is the model's
       * highest-confidence prediction.
       */

      const predictions = data.predictions;

      if (!predictions || predictions.length === 0) {
        throw new Error("Model returned no predictions.");
      }

      const primary = predictions[0];

      /*
       * Convert the backend response into the format
       * already used by the UI.
       */
      setResult({
        primaryBreed: primary.breed,
        confidence: primary.confidence,

        // These are displayed as general information.
        // They are NOT fake model predictions.
        species:
          species === "Auto"
            ? "Indian Bovine"
            : species,

        origin:
          region !== ""
            ? region
            : "Not specified",

        // The ConvNeXt model currently returns breed
        // predictions, not physical trait descriptions.
        traits: [],

        alternatives: predictions.slice(1).map((prediction) => ({
          breed: prediction.breed,
          confidence: prediction.confidence,
        })),
      });

    } catch (error) {
      console.error("Prediction failed:", error);

      setError(
        error.message ||
        "Unable to connect to the AI prediction server."
      );
    } finally {
      setLoading(false);
    }
  };

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

              <button
                type="button"
                className={`animal-option ${
                  species === "Auto" ? "selected" : ""
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


              <button
                type="button"
                className={`animal-option ${
                  species === "Cattle" ? "selected" : ""
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


              <button
                type="button"
                className={`animal-option ${
                  species === "Buffalo" ? "selected" : ""
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
                Make sure the Flask server is running on
                port 5000.
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


              {/* PRIMARY PREDICTION */}

              <div className="primary-result">

                <div className="breed-header">

                  <h4>
                    {result.primaryBreed}
                  </h4>

                  <span className="confidence-badge">
                    {Number(result.confidence).toFixed(2)}% Match
                  </span>

                </div>


                <p>
                  <strong>
                    Category:
                  </strong>{" "}
                  {result.species}
                </p>


                <p>
                  <strong>
                    Region:
                  </strong>{" "}
                  {result.origin}
                </p>


                {/* PHYSICAL FEATURES */}

                {result.traits &&
                  result.traits.length > 0 && (

                    <div className="traits-list">

                      <strong>
                        Key Physical Features:
                      </strong>

                      <ul>

                        {result.traits.map(
                          (trait, idx) => (
                            <li key={idx}>
                              {trait}
                            </li>
                          )
                        )}

                      </ul>

                    </div>

                  )}

              </div>


              {/* ================= ALTERNATIVE PREDICTIONS ================= */}

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
            Your image is processed by the local AI
            prediction server for classification.
          </p>

        </section>

      </main>

    </div>
  );
}

export default App;