#include "appcontroller.h"

#include <QDebug>
#include <QCoreApplication>
#include <QDir>
#include <QStandardPaths>

AppController::AppController(QObject* parent) : QObject(parent) {
    QString modelPath = copyAssetToAppData("assets:/nutrition_model.onnx");
    QString vocabPath = copyAssetToAppData("assets:/vectorizer_vocab.json");
    QString scalerPath = copyAssetToAppData("assets:/output_scaler_params.json");

    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QString modelPathtest = copyAssetToAppData("assets:/nutrition_model.onnx");

    predictor = std::make_unique<NutritionPredictor>(modelPath, vocabPath, scalerPath);
}

void AppController::predict(const QString& foodText) {
    if (!predictor) {
        qWarning() << "Predictor not initialized!";
        return;
    }

    m_lastInput = foodText;
    m_result = predictor->predict(foodText);

    emit predictionChanged();
}



QString AppController::copyAssetToAppData(const QString &assetPath) {
    QFile asset(assetPath);
    if (!asset.exists()) {
        qWarning() << "Asset not found:" << assetPath;
        return {};
    }
    QString appData = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(appData);
    QString dest = appData + "/" + QFileInfo(assetPath).fileName();

    if (QFile::exists(dest))
        QFile::remove(dest);

    if (!asset.copy(dest)) {
        qWarning() << "Failed to copy asset to" << dest;
        return {};
    }
    return dest;
}
