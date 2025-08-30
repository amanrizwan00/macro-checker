#ifndef NUTRITIONPREDICTOR_H
#define NUTRITIONPREDICTOR_H

#pragma once
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QFile>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "textpreprocessor.h"

struct NutritionResult {
    double calories;
    double protein;
    double fat;
    double carbs;
};

class NutritionPredictor {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<TextPreprocessor> preprocessor;

    // Scaler parameters
    std::vector<double> scalerCenter_;
    std::vector<double> scalerScale_;
    bool scalerLoaded_;

    // Quantity parsing
    QMap<QString, double> weightUnits_;
    QSet<QString> volumeUnits_;
    QRegularExpression qtyRegex_;

    void initializeUnits();
    QPair<int, QString> extractQuantityGrams(const QString& text) const;

public:
    NutritionPredictor(const QString& modelPath, const QString& vocabPath, const QString& scalerPath);
    NutritionResult predict(const QString& inputText);

private:
    bool loadScalerParams(const QString& scalerPath);
    std::vector<double> inverseTransform(const std::vector<float>& scaledValues) const;
    void applyClipping(std::vector<double>& values) const;
};


#endif // NUTRITIONPREDICTOR_H
