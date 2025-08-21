
# Threat Detection for Web Server Logs using Autoencoders

This project implements a machine learning-based anomaly detection system for web server logs. The goal is to detect malicious or abnormal requests in web traffic, such as injection attempts, unusual payloads, or suspicious patterns.

The system uses an autoencoder trained exclusively on legitimate traffic (class 0 requests in the original datasets) to learn the normal behavior of web requests. Any request that deviates significantly from this learned pattern is flagged as a potential threat.



## Key Concepts

**Autoencoder**

   * A bottleneck Autoencoder was chosen for this application - trained for 300 epochs.
   * Learns a compressed latent representation (12D -> 6D) and reconstructs the input data from it.
   * Requests with high reconstruction loss do not fit the learned pattern of the latent space and might be potential threats.

**Feature Engineering**

   * Web requests are represented using carefully designed numeric features:

     * Number of single quotes, double quotes, dashes, braces, spaces, semicolons, angle brackets.
     * Count of special characters outside safe URL characters.
     * Lengths of path and body strings.
     * Count of badwords from a curated list or your own custom list.
   * Both **raw and URL-decoded request strings** are analyzed to account for URL decoding and missing any potential obfuscated special characters.

**Thresholding and Threat Levels**

   * After training, a threshold is computed based on the max of reconstruction errors on legitimate traffic.
   * Requests with reconstruction errors above this threshold are flagged as threats.
   * Threat levels are categorized into `low`, `medium`, `high`, and `critical` based on error magnitude.

---
## How it Works

**Training Phase**

   * Load `training_data.csv` containing only legitimate requests.
   * Extract features for each request.
   * Normalize features using `StandardScaler`.
   * Train the `SimpleAutoencoder` to reconstruct legitimate request features.
   * Compute reconstruction errors and calculate a threshold for anomaly detection.
   * Save the model, scaler, feature columns, and threshold for future use.

**Prediction Phase**

   * Load new web server logs.
   * Extract and normalize features using the saved scaler.
   * Evaluate features using trained AE to get reconstruction error for each request.
   * Classify each request as threat or non-threat based on the threshold.
   * Assign a threat level based on the error magnitude.
## Installation

Clone the repo onto your machine and then 
```bash
pip install -r requirements.txt
```

Then simply run 
```bash
python threat_detector.py
```

Training and testing datasets are included in the repo, with the original datasets that these were derived from for exploration purposes.
