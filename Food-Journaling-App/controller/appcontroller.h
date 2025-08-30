#ifndef APPCONTROLLER_H
#define APPCONTROLLER_H

#pragma once
#include <QObject>
#include <memory>
#include "backend/nutritionpredictor.h"

class AppController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString lastInput READ lastInput NOTIFY predictionChanged)
    Q_PROPERTY(double calories READ calories NOTIFY predictionChanged)
    Q_PROPERTY(double protein READ protein NOTIFY predictionChanged)
    Q_PROPERTY(double fat READ fat NOTIFY predictionChanged)
    Q_PROPERTY(double carbs READ carbs NOTIFY predictionChanged)

public:
    explicit AppController(QObject* parent = nullptr);

    Q_INVOKABLE void predict(const QString& foodText);

    QString lastInput() const { return m_lastInput; }
    double calories() const { return m_result.calories; }
    double protein() const { return m_result.protein; }
    double fat() const { return m_result.fat; }
    double carbs() const { return m_result.carbs; }

signals:
    void predictionChanged();

private:
    QString m_lastInput;
    NutritionResult m_result;
    std::unique_ptr<NutritionPredictor> predictor;
    QString copyAssetToAppData(const QString &resourcePath);
};


#endif // APPCONTROLLER_H
