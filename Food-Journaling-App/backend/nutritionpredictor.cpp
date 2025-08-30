#include "nutritionpredictor.h"
#include <QDebug>
#include <algorithm>

NutritionPredictor::NutritionPredictor(const QString& modelPath, const QString& vocabPath, const QString& scalerPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "NutritionApp"),
    session(nullptr),
    sessionOptions(),
    scalerLoaded_(false)
{
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // Initialize units and regex
    initializeUnits();

    try {
        session = Ort::Session(env, modelPath.toStdString().c_str(), sessionOptions);
        preprocessor = std::make_unique<TextPreprocessor>(vocabPath);
        scalerLoaded_ = loadScalerParams(scalerPath);

        qDebug() << "ONNX model loaded from" << modelPath;
        qDebug() << "Scaler loaded:" << (scalerLoaded_ ? "Yes" : "No");

    } catch (const Ort::Exception& e) {
        qWarning() << "Failed to load ONNX model:" << e.what();
    }
}

void NutritionPredictor::initializeUnits() {
    // Weight units (grams conversion factors)
    weightUnits_["kg"] = 1000.0;
    weightUnits_["kilogram"] = 1000.0;
    weightUnits_["kilograms"] = 1000.0;
    weightUnits_["g"] = 1.0;
    weightUnits_["gram"] = 1.0;
    weightUnits_["grams"] = 1.0;
    weightUnits_["mg"] = 0.001;
    weightUnits_["milligram"] = 0.001;
    weightUnits_["milligrams"] = 0.001;
    weightUnits_["oz"] = 28.349523125;
    weightUnits_["ounce"] = 28.349523125;
    weightUnits_["ounces"] = 28.349523125;
    weightUnits_["lb"] = 453.59237;
    weightUnits_["lbs"] = 453.59237;
    weightUnits_["pound"] = 453.59237;
    weightUnits_["pounds"] = 453.59237;

    // Volume units (ignored to avoid wrong conversions)
    volumeUnits_ << "cup" << "cups" << "tbsp" << "tablespoon" << "tablespoons"
                 << "tsp" << "teaspoon" << "teaspoons";

    // Regex pattern: number (int/decimal) + optional space + unit
    qtyRegex_.setPattern(R"((\d+(?:\.\d+)?)\s*([a-zA-Z]+))");
}

QPair<int, QString> NutritionPredictor::extractQuantityGrams(const QString& text) const {
    QString s = text.toLower().trimmed();
    QList<double> gramsCandidates;
    QList<QPair<int, int>> spansToRemove; // start, length pairs

    QRegularExpressionMatchIterator it = qtyRegex_.globalMatch(s);
    while (it.hasNext()) {
        QRegularExpressionMatch match = it.next();
        QString numStr = match.captured(1);
        QString unit = match.captured(2);

        // Skip if not a weight unit or if it's a volume unit
        if (!weightUnits_.contains(unit) || volumeUnits_.contains(unit)) {
            continue;
        }

        bool ok;
        double val = numStr.toDouble(&ok);
        if (!ok) continue;

        double grams = val * weightUnits_[unit];
        gramsCandidates.append(grams);
        spansToRemove.append({match.capturedStart(), match.capturedLength()});
    }

    int finalGrams = 100; // default
    QList<QPair<int, int>> actualSpansToRemove;

    if (!gramsCandidates.isEmpty()) {
        // Choose the largest plausible weight mention
        double gramsFloat = *std::max_element(gramsCandidates.begin(), gramsCandidates.end());

        // Clamp to safe range; else fallback
        if (gramsFloat >= 10 && gramsFloat <= 2000 &&
            !std::isnan(gramsFloat) && !std::isinf(gramsFloat)) {
            finalGrams = static_cast<int>(std::round(gramsFloat));
            actualSpansToRemove = spansToRemove;
        }
    }

    // Remove matched quantity tokens from text
    QString cleanText = s;
    if (!actualSpansToRemove.isEmpty()) {
        // Sort by start position in reverse order to maintain indices
        std::sort(actualSpansToRemove.begin(), actualSpansToRemove.end(),
                  [](const QPair<int, int>& a, const QPair<int, int>& b) {
                      return a.first > b.first;
                  });

        for (const auto& span : actualSpansToRemove) {
            cleanText.remove(span.first, span.second);
        }
    }

    // Collapse whitespace
    cleanText = cleanText.simplified();

    return {finalGrams, cleanText};
}

bool NutritionPredictor::loadScalerParams(const QString& scalerPath) {
    QFile file(scalerPath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open scaler params file:" << scalerPath;
        return false;
    }

    QByteArray data = file.readAll();
    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(data, &err);

    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        qWarning() << "Invalid scaler JSON format:" << err.errorString();
        return false;
    }

    QJsonObject obj = doc.object();

    // Load center values (medians)
    QJsonArray centerArray = obj["center_"].toArray();
    scalerCenter_.clear();
    scalerCenter_.reserve(centerArray.size());
    for (const auto& val : centerArray) {
        scalerCenter_.push_back(val.toDouble());
    }

    // Load scale values (IQR-based)
    QJsonArray scaleArray = obj["scale_"].toArray();
    scalerScale_.clear();
    scalerScale_.reserve(scaleArray.size());
    for (const auto& val : scaleArray) {
        scalerScale_.push_back(val.toDouble());
    }

    qDebug() << "Loaded scaler params - Center size:" << scalerCenter_.size()
             << "Scale size:" << scalerScale_.size();
    qDebug() << "Centers:" << QString("[%1, %2, %3, %4]")
                                  .arg(scalerCenter_[0], 0, 'f', 2)
                                  .arg(scalerCenter_[1], 0, 'f', 2)
                                  .arg(scalerCenter_[2], 0, 'f', 2)
                                  .arg(scalerCenter_[3], 0, 'f', 2);
    qDebug() << "Scales:" << QString("[%1, %2, %3, %4]")
                                 .arg(scalerScale_[0], 0, 'f', 2)
                                 .arg(scalerScale_[1], 0, 'f', 2)
                                 .arg(scalerScale_[2], 0, 'f', 2)
                                 .arg(scalerScale_[3], 0, 'f', 2);

    return scalerCenter_.size() == 4 && scalerScale_.size() == 4;
}

std::vector<double> NutritionPredictor::inverseTransform(const std::vector<float>& scaledValues) const {
    std::vector<double> result;
    result.reserve(scaledValues.size());

    for (size_t i = 0; i < scaledValues.size() && i < scalerCenter_.size(); ++i) {
        // RobustScaler inverse transform: X = X_scaled * scale + center
        double original = static_cast<double>(scaledValues[i]) * scalerScale_[i] + scalerCenter_[i];
        result.push_back(original);
    }

    return result;
}

void NutritionPredictor::applyClipping(std::vector<double>& values) const {
    if (values.size() >= 4) {
        values[0] = std::max(0.0, std::min(1200.0, values[0])); // calories
        values[1] = std::max(0.0, std::min(120.0, values[1]));  // protein
        values[2] = std::max(0.0, std::min(120.0, values[2]));  // fat
        values[3] = std::max(0.0, std::min(180.0, values[3]));  // carbs
    }
}

NutritionResult NutritionPredictor::predict(const QString& inputText) {
    NutritionResult result{0, 0, 0, 0};

    if (!scalerLoaded_) {
        qWarning() << "Scaler not loaded, cannot make predictions";
        return result;
    }

    // Parse quantity and clean text (same as Python)
    QPair<int, QString> parsed = extractQuantityGrams(inputText);
    int grams = parsed.first;
    QString cleanText = parsed.second;

    qDebug() << "Input:" << inputText;
    qDebug() << "Parsed: grams =" << grams << ", clean text ='" << cleanText << "'";

    // Convert input into features using cleaned text
    std::vector<float> features = preprocessor->transform(cleanText);
    size_t inputTensorSize = features.size();
    std::array<int64_t, 2> inputShape{1, static_cast<int64_t>(features.size())};

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input / output names
    Ort::AllocatedStringPtr inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
    const char* inputName = inputNameAllocated.get();
    const char* outputName = outputNameAllocated.get();

    // Create tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        features.data(),
        inputTensorSize,
        inputShape.data(),
        inputShape.size());

    // Run inference
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        &inputName, &inputTensor, 1,   // inputs
        &outputName, 1);               // outputs

    // Extract raw scaled results
    float* floatArray = outputTensors.front().GetTensorMutableData<float>();
    std::vector<float> scaledOutput = {floatArray[0], floatArray[1], floatArray[2], floatArray[3]};

    // Apply inverse transform to get per-100g nutrition values
    std::vector<double> per100g = inverseTransform(scaledOutput);

    // Apply clipping to plausible ranges
    applyClipping(per100g);

    qDebug() << "Prediction (per-100g):" << QString("%1 kcal | %2 g P | %3 g F | %4 g C")
                                                .arg(per100g[0], 0, 'f', 1)
                                                .arg(per100g[1], 0, 'f', 1)
                                                .arg(per100g[2], 0, 'f', 1)
                                                .arg(per100g[3], 0, 'f', 1);

    // Scale to actual serving size (same as Python: final = per100g * (grams / 100.0))
    double scale = static_cast<double>(grams) / 100.0;
    result.calories = per100g[0] * scale;
    result.protein  = per100g[1] * scale;
    result.fat      = per100g[2] * scale;
    result.carbs    = per100g[3] * scale;

    qDebug() << QString("Scaled to %1 g:").arg(grams) << QString("%1 kcal | %2 g P | %3 g F | %4 g C")
                                                             .arg(result.calories, 0, 'f', 1)
                                                             .arg(result.protein, 0, 'f', 1)
                                                             .arg(result.fat, 0, 'f', 1)
                                                             .arg(result.carbs, 0, 'f', 1);

    return result;
}
