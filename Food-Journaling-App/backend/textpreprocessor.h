#ifndef TEXTPREPROCESSOR_H
#define TEXTPREPROCESSOR_H

#pragma once
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <unordered_map>
#include <vector>

class TextPreprocessor {
public:
    explicit TextPreprocessor(const QString& vocabPath);

    // Convert input text into numeric feature vector
    std::vector<float> transform(const QString& text) const;

private:
    std::unordered_map<QString, int> vocab;
    int vocabSize = 0;
};

#endif // TEXTPREPROCESSOR_H
