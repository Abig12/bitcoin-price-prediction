document
  .getElementById("prediction-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    // Gather form data
    const formData = {};
    const inputs = document.querySelectorAll("#prediction-form input");
    inputs.forEach((input) => {
      formData[input.name] = parseFloat(input.value);
    });

    // Show loading state
    const submitButton = this.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;
    submitButton.innerHTML =
      '<i class="fas fa-spinner fa-spin mr-2"></i>Predicting...';
    submitButton.disabled = true;

    // Make prediction request
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        // Create result HTML
        const resultDiv = document.getElementById("prediction-result");
        const predictionColor =
          data.prediction === "Buy" ? "#2ecc71" : "#e74c3c";

        resultDiv.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h3 class="card-title" style="color: ${predictionColor};">
                        <i class="fas ${
                          data.prediction === "Buy"
                            ? "fa-arrow-up"
                            : "fa-arrow-down"
                        } mr-2"></i>
                        Prediction: ${data.prediction}
                    </h3>
                    <p class="mb-2">Confidence: ${(
                      data.confidence * 100
                    ).toFixed(2)}%</p>
                    <p class="mb-0">Model Accuracy: ${(
                      data.model_accuracy * 100
                    ).toFixed(2)}%</p>
                </div>
            </div>
        `;
      })
      .catch((error) => {
        console.error("Error:", error);
        document.getElementById("prediction-result").innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle mr-2"></i>
                An error occurred while making the prediction. Please try again.
            </div>
        `;
      })
      .finally(() => {
        // Restore button state
        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;
      });
  });
