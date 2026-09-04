import { useState } from "react";
import "./App.css";

function App() {
  const [species, setSpecies] = useState("Auto");
  const [region, setRegion] = useState("");
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleImageUpload = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImage(URL.createObjectURL(selectedFile));
      setResult(null); // Reset previous result on new upload
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("species", species);
    formData.append("region", region);

    try {
      // Dummy delay for UI testing
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Mocked Response Data
      setResult({
        primaryBreed: "Gir",
        confidence: 94.2,
        species: "Cattle (Bos indicus)",
        origin: "Gujarat, India",
        traits: ["Distinct convex forehead", "Pendulous ears", "Red/Speckled coat"],
        alternatives: [
          { breed: "Sahiwal", confidence: 4.1 },
          { breed: "Rathi", confidence: 1.2 },
        ],
      });
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo">
          🐄 <span>BovineAI</span>
        </div>
        <div className="tagline">AI-Powered Indian Breed Identification</div>
      </header>

      <main className="main-content">
        <section className="hero">
          <p className="eyebrow">AI + LIVESTOCK TECHNOLOGY</p>
          <h1>
            Identify Indian
            <br />
            <span>Cattle & Buffalo Breeds</span>
          </h1>
          <p className="description">
            Upload a photo and get an instant AI-powered breed prediction with confidence scores and trait analysis.
          </p>
        </section>

        <section className="card">
          <h2>Breed Classification</h2>
          <p className="section-description">
            Upload an image below. Provide optional context to improve prediction accuracy.
          </p>

          {/* Image Upload */}
          <div className="field">
            <label>Animal Photo</label>
            <label className="upload-box">
              {image ? (
                <img src={image} alt="Uploaded animal" className="preview" />
              ) : (
                <>
                  <div className="upload-icon">📷</div>
                  <strong>Upload animal photo</strong>
                  <span>Click here to select a JPG, PNG, or WEBP image</span>
                </>
              )}
              <input type="file" accept="image/*" onChange={handleImageUpload} />
            </label>
          </div>

          {/* Species Selection */}
          <div className="field">
            <label>
              Species / Category <span className="optional-tag">(Optional)</span>
            </label>
            <div className="animal-options">
              <button
                type="button"
                className={`animal-option ${species === "Auto" ? "selected" : ""}`}
                onClick={() => setSpecies("Auto")}
              >
                <span>✨</span>
                <div>
                  <strong>Auto-Detect</strong>
                  <small>Let AI detect</small>
                </div>
              </button>

              <button
                type="button"
                className={`animal-option ${species === "Cattle" ? "selected" : ""}`}
                onClick={() => setSpecies("Cattle")}
              >
                <span>🐄</span>
                <div>
                  <strong>Cattle</strong>
                </div>
              </button>

              <button
                type="button"
                className={`animal-option ${species === "Buffalo" ? "selected" : ""}`}
                onClick={() => setSpecies("Buffalo")}
              >
                <span>🐃</span>
                <div>
                  <strong>Buffalo</strong>
                </div>
              </button>
            </div>
          </div>

          {/* Region Dropdown */}
          <div className="field">
            <label htmlFor="region">
              State / Region <span className="optional-tag">(Optional)</span>
            </label>
            <select
              id="region"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
            >
              <option value="">Select Indian State / Region</option>
              <option>Andhra Pradesh</option>
              <option>Gujarat</option>
              <option>Haryana</option>
              <option>Madhya Pradesh</option>
              <option>Maharashtra</option>
              <option>Punjab</option>
              <option>Rajasthan</option>
              <option>Tamil Nadu</option>
              <option>Uttar Pradesh</option>
              <option>Other / Unknown</option>
            </select>
            <small className="hint">
              Helps narrow down geographically localized native breeds (e.g., Gir, Murrah, Kangayam).
            </small>
          </div>

          <button
            className="analyze-button"
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {loading ? "⏳ Analyzing Image..." : "🔍 Identify Breed"}
          </button>

          {/* Results Display Section */}
          {result && (
            <div className="results-container">
              <hr className="divider" />
              <h3>Identification Result</h3>

              <div className="primary-result">
                <div className="breed-header">
                  <h4>{result.primaryBreed}</h4>
                  <span className="confidence-badge">{result.confidence}% Match</span>
                </div>
                <p><strong>Category:</strong> {result.species}</p>
                <p><strong>Native Region:</strong> {result.origin}</p>

                <div className="traits-list">
                  <strong>Key Physical Features:</strong>
                  <ul>
                    {result.traits.map((trait, idx) => (
                      <li key={idx}>{trait}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {result.alternatives && result.alternatives.length > 0 && (
                <div className="alternatives-section">
                  <h5>Other Close Predictions</h5>
                  <ul>
                    {result.alternatives.map((alt, idx) => (
                      <li key={idx}>
                        <span>{alt.breed}</span>
                        <span>{alt.confidence}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <p className="privacy-note">
            Your image is processed securely and used only for classification.
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;