#include "textpreprocessor.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QDebug>

TextPreprocessor::TextPreprocessor(const QString& vocabPath) {
    QFile file(vocabPath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open vocab file:" << vocabPath;
        return;
    }

    QByteArray data = file.readAll();
    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(data, &err);

    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        qWarning() << "Invalid vocab JSON format:" << err.errorString();
        return;
    }

    QJsonObject obj = doc.object();
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        vocab[it.key()] = it.value().toInt();
    }

    vocabSize = vocab.size();
    qDebug() << "Loaded vocab with size:" << vocabSize;
}

std::vector<float> TextPreprocessor::transform(const QString& text) const {
    std::vector<float> features(vocabSize, 0.0f);
    QString lower = text.toLower();

    // Match sklearn's token pattern more precisely: (?u)\b[a-zA-Z][a-zA-Z]+\b
    // This means: word boundary + letter + one or more letters + word boundary
    QRegularExpression re("\\b[a-zA-Z][a-zA-Z]+\\b");

    // Collect all tokens first to match TF-IDF behavior
    QStringList tokens;
    QRegularExpressionMatchIterator it = re.globalMatch(lower);
    while (it.hasNext()) {
        QString token = it.next().captured();
        if (token.length() >= 2) { // sklearn's pattern ensures this, but double-check
            tokens.append(token);
        }
    }

    // Apply stop words filtering (basic English stop words)
    QSet<QString> stopWords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "the", "this", "but", "they", "have",
        "had", "what", "said", "each", "which", "she", "do", "how", "their",
        "if", "up", "out", "many", "then", "them", "these", "so", "some",
        "her", "would", "make", "like", "into", "him", "time", "two", "more",
        "go", "no", "way", "could", "my", "than", "first", "been", "call",
        "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
        "come", "made", "may", "part"
    };

    // Filter out stop words and count term frequencies
    QMap<QString, int> termFreq;
    for (const QString& token : tokens) {
        if (!stopWords.contains(token) && vocab.contains(token)) {
            termFreq[token]++;
        }
    }

    // Fill the feature vector with term frequencies
    // Note: This is simpler than full TF-IDF, but for bag-of-words it should work
    for (auto it = termFreq.constBegin(); it != termFreq.constEnd(); ++it) {
        const QString& token = it.key();
        int freq = it.value();

        auto vocabIt = vocab.find(token);
        if (vocabIt != vocab.end()) {
            features[vocabIt->second] = static_cast<float>(freq);
        }
    }

    // Debug: print non-zero features
    QStringList activeTokens;
    for (auto it = termFreq.constBegin(); it != termFreq.constEnd(); ++it) {
        if (vocab.find(it.key()) != vocab.end()) {
            activeTokens.append(QString("%1:%2").arg(it.key()).arg(it.value()));
        }
    }
    if (!activeTokens.isEmpty()) {
        qDebug() << "Active tokens:" << activeTokens.join(", ");
    }

    return features;
}

